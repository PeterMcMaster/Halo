"""Configuration helpers for Halo."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
import os

BASE_DIR = Path(__file__).resolve().parent.parent.parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH)


@dataclass
class AppConfig:
    photo_root: Path = BASE_DIR / "photos"
    chroma_db_dir: Path = BASE_DIR / "vector_store"
    device: str = "cuda" if os.environ.get("USE_CUDA") == "1" else "mps"
    openai_api_key: Optional[str] = os.environ.get("OPENAI_API_KEY")
    openai_model: str = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    gemini_api_key: Optional[str] = os.environ.get("GEMINI_API_KEY")
    # Default to a model name that works with current Gemini generateContent (free tier compatible).
    gemini_model: str = os.environ.get("GEMINI_MODEL", "models/gemini-2.5-flash")
    llm_provider: Literal["openai", "gemini", "none"] = os.environ.get("LLM_PROVIDER", "gemini").lower()
    clip_model_name: str = os.environ.get("CLIP_MODEL", "openai/clip-vit-base-patch32")
    blip_model_name: str = os.environ.get("BLIP_MODEL", "Salesforce/blip-image-captioning-base")
    enable_blip: bool = os.environ.get("ENABLE_BLIP", "1") not in {"0", "false", "False"}
    caption_weight: float = float(os.environ.get("CAPTION_WEIGHT", "0.2"))
    default_top_k: int = int(os.environ.get("DEFAULT_TOP_K", "12"))
    image_collection: str = os.environ.get("IMAGE_COLLECTION", "halo-images")
    caption_collection: str = os.environ.get("CAPTION_COLLECTION", "halo-captions")

    def ensure_dirs(self) -> None:
        self.photo_root.mkdir(exist_ok=True, parents=True)
        self.chroma_db_dir.mkdir(exist_ok=True, parents=True)


def get_config() -> AppConfig:
    cfg = AppConfig()
    if cfg.llm_provider not in {"openai", "gemini", "none"}:
        cfg.llm_provider = "gemini"
    cfg.ensure_dirs()
    return cfg
