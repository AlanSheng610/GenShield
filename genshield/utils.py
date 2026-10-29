import os
import random

import numpy as np
import torch
import yaml


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device=None):
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_sd_pipeline(model_id, device, dtype=None, scheduler_cls=None):
    """Stable Diffusion with its built-in safety checker disabled, using a DDIM scheduler by default.
    ``dtype`` defaults to fp16 on GPU and fp32 on CPU."""
    from diffusers import DDIMScheduler, StableDiffusionPipeline

    device = torch.device(device)
    dtype = dtype or (torch.float16 if device.type == "cuda" else torch.float32)
    scheduler_cls = scheduler_cls or DDIMScheduler
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False
    )
    pipe.scheduler = scheduler_cls.from_pretrained(model_id, subfolder="scheduler")
    pipe.set_progress_bar_config(disable=True)
    return pipe.to(device)
