import logging
import warnings
from datetime import timedelta
from typing import List, Optional, Tuple, Union

import PIL
import torch
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from packaging import version
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.api.visual_payload import (
    KIND_IMAGE_SEQUENCE,
    KIND_LEGACY_ITEMS,
    KIND_VIDEO_PATH,
    normalize_visual_payloads,
)

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
eval_logger = logging.getLogger("lmms-eval")

# Enable TF32 for CUDA
torch.backends.cuda.matmul.allow_tf32 = True


# Determine best attention implementation
if version.parse(torch.__version__) >= version.parse("2.1.2"):
    best_fit_attn_implementation = "sdpa"
else:
    best_fit_attn_implementation = "eager"


@register_model("videochat_flash")
class VideoChat_Flash(lmms):
    """
    VideoChat Flash
    """

    def __init__(
        self,
        pretrained: str = "OpenGVLab/VideoChat-Flash-Qwen2-7B_res448",
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        device_map: Optional[str] = "cuda:0",
        use_cache: Optional[bool] = True,
        max_num_frames: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now

        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        assert torch.cuda.device_count() > 0, torch.cuda.device_count()
        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        # `-1` is the model-side sentinel for "do not cap sampled frames".
        self.max_num_frames = -1 if max_num_frames is None else int(max_num_frames)

        self._tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)
        self._model = AutoModel.from_pretrained(pretrained, trust_remote_code=True).half().cuda()

        # modify here to use video-level compress
        self.model.config.mm_llm_compress = False
        self.model.config.llm_compress_type = "attention"
        self.model.config.llm_compress_layer_list = [24]
        self.model.config.llm_image_token_ratio_list = [1.0, 0.5]

        self._config = self._model.config
        self.model.eval()

        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        assert self.batch_size_per_gpu == 1

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")

            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes

        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with tensor parallelism")
            self._rank = 0
            self._world_size = 1

        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        """ """
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        try:
            return self.tokenizer.decode(tokens)
        except:
            return self.tokenizer.decode([tokens])

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def _resolve_visual_inputs(self, visual):
        video_path = None
        media_dict = {"video_read_type": "decord"}
        extra_images = []

        for payload in normalize_visual_payloads(visual):
            kind = payload.get("kind")
            metadata = payload.get("metadata") or {}

            if kind == KIND_VIDEO_PATH:
                if video_path is not None:
                    raise NotImplementedError("videochat_flash only supports a single video payload per request.")
                video_path = payload.get("video_path")
                media_dict = {"video_read_type": metadata.get("video_read_type", "decord")}
                if "start" in metadata and "end" in metadata:
                    media_dict["start"] = metadata["start"]
                    media_dict["end"] = metadata["end"]
            elif kind == KIND_IMAGE_SEQUENCE:
                extra_images.extend(payload.get("images") or [])
            elif kind == KIND_LEGACY_ITEMS:
                for item in payload.get("items") or []:
                    if isinstance(item, str):
                        if video_path is not None:
                            raise NotImplementedError("videochat_flash only supports a single video path per request.")
                        video_path = item
                    elif isinstance(item, dict):
                        media_dict.update(item)
                    elif isinstance(item, PIL.Image.Image):
                        extra_images.append(item)
                    else:
                        raise NotImplementedError(f"Unsupported legacy visual item: {type(item)}")
            else:
                raise NotImplementedError(f"Unsupported visual payload kind: {kind}")

        return video_path, media_dict, extra_images

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for request in requests:
            metadata = request.metadata or {}
            context, gen_kwargs, doc_to_visual, doc_id, task, split = request.args
            batched_visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            gen_kwargs = dict(gen_kwargs)
            if "until" in gen_kwargs:
                gen_kwargs.pop("until")

            # preconfigure gen_kwargs with defaults
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "do_sample" not in gen_kwargs:
                gen_kwargs["do_sample"] = False
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            for visual in batched_visuals:
                if len(visual) > 1 or "image_aspect_ratio" not in self._config.__dict__:  # for multi image case, we treat per image aspect ratio as "pad" by default.
                    self._config.image_aspect_ratio = gen_kwargs.get("image_aspect_ratio", "pad")
                    eval_logger.info(f"Setting image aspect ratio: {self._config.image_aspect_ratio}")

                video_path, media_dict, extra_images = self._resolve_visual_inputs(visual)
                if extra_images:
                    raise NotImplementedError("videochat_flash does not support Cambrian-W frame-recall requests with extra question images.")
                if not video_path:
                    raise NotImplementedError(f"Missing video payload for request: task={task}, metadata={metadata}")

                if isinstance(video_path, str):  # For video task
                    response = self.model.chat(
                        video_path,
                        self.tokenizer,
                        context,
                        chat_history=None,
                        return_history=False,
                        max_num_frames=self.max_num_frames,
                        media_dict=media_dict,
                        generation_config={
                            "max_new_tokens": gen_kwargs["max_new_tokens"],
                            "temperature": gen_kwargs["temperature"],
                            "do_sample": gen_kwargs["do_sample"],
                            "top_p": gen_kwargs["top_p"],
                            "num_beams": gen_kwargs["num_beams"],
                        },
                    )
                    response = response.strip()
                    res.append(response)
                    self.cache_hook.add_partial("generate_until", (context, gen_kwargs), [response])

                else:
                    raise NotImplementedError(f"Unsupported resolved video input: {type(video_path)}")

            pbar.update(1)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests):
        return self.generate_until(requests)
