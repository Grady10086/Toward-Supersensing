from datetime import timedelta
from typing import List, Optional, Tuple, Union

import av
import numpy as np
import torch
from PIL import Image
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from loguru import logger
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

eval_logger = logger

from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor

from lmms_eval.models.model_utils.load_video import read_video
from lmms_eval.api.visual_payload import flatten_visual_inputs, normalize_visual_payloads


@register_model("video_llava")
class VideoLLaVA(lmms):
    def __init__(
        self,
        pretrained: str = "LanguageBind/Video-LLaVA-7B-hf",
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda:0",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        trust_remote_code: Optional[bool] = False,
        revision=None,
        attn_implementation=(
            "sdpa" if torch.__version__ > "2.1.2" else "eager"
        ),  # inference implementation for attention, can be "sdpa", "eager", "flash_attention_2". Seems FA2 is not effective during inference: https://discuss.huggingface.co/t/flash-attention-has-no-effect-on-inference/73453/5
        device_map="cuda:0",
        conv_template="llava_v1",
        use_cache=True,
        truncate_context=False,
        num_frames: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__()
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

        self.pretrained = pretrained
        self._model = VideoLlavaForConditionalGeneration.from_pretrained(pretrained)
        self._processor = VideoLlavaProcessor.from_pretrained(pretrained)
        self.prompt = "USER: <video>{}? ASSISTANT:"
        self.num_frames = num_frames
        # self.model_name = get_model_name_from_path(pretrained)
        # self._tokenizer, self._model, self.processor, self._max_length = load_pretrained_model(pretrained, None, self.model_name, device_map=self.device_map)
        # self.video_processor = self.processor["video"]
        self._config = self._model.config
        self.model.eval()
        self.model.tie_weights()
        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        self.use_cache = use_cache
        self.truncate_context = truncate_context
        # assert self.batch_size_per_gpu == 1, "Llava currently does not support batched generation. See https://github.com/haotian-liu/LLaVA/issues/754. HF Llava also has this issue."
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

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

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

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        return super().loglikelihood(requests)

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def _probe_video_metadata(self, video_path):
        container = av.open(video_path)
        try:
            stream = container.streams.video[0]
            total_frames = int(stream.frames or 0)
            frame_rate = float(stream.average_rate) if stream.average_rate is not None else None
            if total_frames <= 0:
                total_frames = 0
                for total_frames, _ in enumerate(container.decode(video=0), start=1):
                    pass
            return total_frames, frame_rate
        finally:
            container.close()

    def _resolve_video_frame_count(self, video_path, fps=None, max_frames=None):
        total_frames, frame_rate = self._probe_video_metadata(video_path)
        if total_frames <= 0:
            fallback = max_frames if max_frames is not None else self.num_frames
            return max(1, int(fallback or 1))

        target = total_frames
        if fps is not None and frame_rate and frame_rate > 0:
            target = max(1, int(total_frames / frame_rate * float(fps)))
        if max_frames is not None:
            target = min(target, int(max_frames))
        elif self.num_frames is not None:
            target = min(target, int(self.num_frames))
        return max(1, target)

    def _decode_video_payload(self, payload):
        metadata = payload.get("metadata") or {}
        video_path = payload["video_path"]
        fps = metadata.get("fps")
        max_frames = metadata.get("max_num_frames")
        if max_frames is None:
            max_frames = metadata.get("max_frames")
        target_frames = self._resolve_video_frame_count(video_path, fps=fps, max_frames=max_frames)
        return read_video(video_path, num_frm=target_frames, fps=fps)

    def _build_clip(self, visuals):
        payloads = normalize_visual_payloads(visuals)
        frames = []
        for payload in payloads:
            kind = payload.get("kind")
            if kind == "video_path":
                decoded = self._decode_video_payload(payload)
                frames.extend(list(decoded))
                continue
            if kind == "image_sequence":
                for image in payload.get("images", []):
                    frames.append(np.asarray(image.convert("RGB"), dtype=np.uint8))
                continue
            if kind == "legacy_items":
                for visual in payload.get("items", []):
                    if isinstance(visual, Image.Image):
                        frames.append(np.asarray(visual.convert("RGB"), dtype=np.uint8))
                    elif isinstance(visual, np.ndarray) and visual.ndim == 3:
                        frames.append(visual.astype(np.uint8, copy=False))
                    elif isinstance(visual, str):
                        decoded = read_video(visual, num_frm=self._resolve_video_frame_count(visual, fps=None, max_frames=self.num_frames))
                        frames.extend(list(decoded))
                    else:
                        raise TypeError(f"Unsupported visual type for Video-LLaVA: {type(visual).__name__}")
                continue
            raise ValueError(f"Unsupported visual payload kind for Video-LLaVA: {kind}")

        if not frames:
            raise ValueError("No visuals provided to Video-LLaVA")

        if self.num_frames is not None:
            if len(frames) >= self.num_frames:
                sample_idx = np.linspace(0, len(frames) - 1, self.num_frames, dtype=int)
                frames = [frames[idx] for idx in sample_idx]
            else:
                while len(frames) < self.num_frames:
                    frames.append(frames[-1].copy())

        return np.stack(frames, axis=0)

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            # encode, pad, and truncate contexts for this batch
            visual_payloads = normalize_visual_payloads(doc_to_visual(self.task_dict[task][split][doc_id]))
            clip = self._build_clip(visual_payloads)

            inputs = self._processor(text=self.prompt.format(contexts), videos=clip, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            generate_ids = self.model.generate(**inputs, max_new_tokens=gen_kwargs["max_new_tokens"], temperature=gen_kwargs["temperature"])

            outputs = self._processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].split("ASSISTANT:")[-1].strip()
            res.append(outputs)
            pbar.update(1)
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
