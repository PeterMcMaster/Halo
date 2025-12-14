"""Embedding utilities for Halo."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import BinaryIO, Tuple, Union

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPModel, CLIPProcessor

from .config import get_config


def _device() -> torch.device:
    cfg = get_config()
    if cfg.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if cfg.device == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@lru_cache(maxsize=1)
def load_clip_model(model_name: str | None = None) -> Tuple[CLIPModel, CLIPProcessor]:
    cfg = get_config()
    name = model_name or cfg.clip_model_name
    model = CLIPModel.from_pretrained(name)
    processor = CLIPProcessor.from_pretrained(name)
    model.to(_device())
    model.eval()
    return model, processor


def _load_image(image_source: Union[str, Path, Image.Image, BinaryIO]) -> Image.Image:
    if isinstance(image_source, Image.Image):
        return image_source.convert("RGB")
    if isinstance(image_source, (str, Path)):
        return Image.open(image_source).convert("RGB")
    return Image.open(image_source).convert("RGB")


def embed_image(image_source: Union[str, Path, Image.Image, BinaryIO]) -> torch.Tensor:
    model, processor = load_clip_model()
    image = _load_image(image_source)
    inputs = processor(images=image, return_tensors="pt").to(_device())
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    norm = features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    normalized = features / norm
    return normalized.cpu().squeeze(0)


def embed_text(text: str) -> torch.Tensor:
    model, processor = load_clip_model()
    inputs = processor(text=text, return_tensors="pt", padding=True).to(_device())
    with torch.no_grad():
        features = model.get_text_features(**inputs)
    norm = features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    normalized = features / norm
    return normalized.cpu().squeeze(0)


@lru_cache(maxsize=1)
def load_blip_model(model_name: str | None = None) -> Tuple[BlipForConditionalGeneration, BlipProcessor]:
    cfg = get_config()
    name = model_name or cfg.blip_model_name
    processor = BlipProcessor.from_pretrained(name)
    model = BlipForConditionalGeneration.from_pretrained(name)
    model.to(_device())
    model.eval()
    return model, processor


def generate_caption(image_source: Union[str, Path, Image.Image, BinaryIO], max_new_tokens: int = 40) -> str:
    cfg = get_config()
    if not cfg.enable_blip:
        return ""
    model, processor = load_blip_model()
    image = _load_image(image_source)
    inputs = processor(images=image, return_tensors="pt").to(_device())
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    caption = processor.tokenizer.decode(out[0], skip_special_tokens=True)
    return caption.strip()
