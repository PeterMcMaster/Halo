"""Photo ingestion and embedding storage."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import exifread
from chromadb import PersistentClient
import hashlib

from .config import get_config
from .embeddings import embed_image, embed_text, generate_caption


SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _ratio_to_float(value) -> Optional[float]:
    try:
        return float(value.num) / float(value.den)
    except AttributeError:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


def _convert_gps(tag_value, ref_tag) -> Optional[float]:
    if not tag_value:
        return None
    values = getattr(tag_value, "values", tag_value)
    if not values:
        return None
    ratios = [_ratio_to_float(v) for v in values]
    if len(ratios) < 3 or any(r is None for r in ratios):
        return None
    degrees, minutes, seconds = ratios[:3]
    decimal = degrees + minutes / 60 + seconds / 3600
    ref = str(ref_tag) if ref_tag else ""
    if ref.strip().upper() in {"S", "W"}:
        decimal = -decimal
    return decimal


def _parse_datetime(raw_value: str) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    if not raw_value:
        return None, None, None
    for pattern in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(raw_value, pattern)
            return dt.isoformat(), dt.year, dt.month
        except ValueError:
            continue
    return None, None, None


def _read_metadata(image_path: Path) -> dict:
    try:
        with image_path.open("rb") as stream:
            tags = exifread.process_file(stream, details=False)
    except Exception:  # noqa: BLE001 - metadata parsing best effort
        tags = {}

    dt_raw = str(tags.get("EXIF DateTimeOriginal", ""))
    dt_iso, year, month = _parse_datetime(dt_raw)
    lat = _convert_gps(tags.get("GPS GPSLatitude"), tags.get("GPS GPSLatitudeRef"))
    lon = _convert_gps(tags.get("GPS GPSLongitude"), tags.get("GPS GPSLongitudeRef"))

    return {
        "datetime_raw": dt_raw,
        "datetime_iso": dt_iso,
        "year": year,
        "month": month,
        "latitude": lat,
        "longitude": lon,
    }


@dataclass
class IngestionResult:
    indexed: int
    skipped: int


class PhotoIndexer:
    def __init__(self, enable_captions: Optional[bool] = None) -> None:
        cfg = get_config()
        self.cfg = cfg
        self.enable_captions = cfg.enable_blip if enable_captions is None else enable_captions
        self.client = PersistentClient(path=str(cfg.chroma_db_dir))
        self.image_collection = self.client.get_or_create_collection(
            cfg.image_collection,
            metadata={"hnsw:space": "cosine"},
        )
        self.caption_collection = None
        if self.enable_captions:
            self.caption_collection = self.client.get_or_create_collection(
                cfg.caption_collection,
                metadata={"hnsw:space": "cosine"},
            )

    def _iter_images(self, root: Path) -> Iterable[Path]:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXT:
                yield path

    def _make_id(self, image_path: Path) -> str:
        return hashlib.md5(image_path.as_posix().encode("utf-8")).hexdigest()

    def index_folder(self, folder: str | Path) -> IngestionResult:
        folder_path = Path(folder).resolve()
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")

        documents: List[str] = []
        embeddings: List[List[float]] = []
        metadatas: List[dict] = []
        ids: List[str] = []
        caption_docs: List[str] = []
        caption_embeddings: List[List[float]] = []
        caption_metadata: List[dict] = []
        caption_ids: List[str] = []
        skipped = 0

        for image_path in self._iter_images(folder_path):
            try:
                vector = embed_image(str(image_path)).tolist()
            except Exception:  # noqa: BLE001 - log later
                skipped += 1
                continue

            metadata = _read_metadata(image_path)
            caption_text = ""
            if self.enable_captions:
                try:
                    caption_text = generate_caption(str(image_path))
                except Exception:
                    caption_text = ""
            if caption_text:
                metadata["caption"] = caption_text

            doc_path = str(image_path)
            documents.append(doc_path)
            embeddings.append(vector)
            metadatas.append(metadata)
            ids.append(self._make_id(image_path))

            if caption_text and self.caption_collection:
                caption_vector = embed_text(caption_text).tolist()
                caption_docs.append(doc_path)
                caption_embeddings.append(caption_vector)
                caption_metadata.append(metadata)
                caption_ids.append(self._make_id(image_path) + "-caption")

        if embeddings:
            self.image_collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )

        if caption_embeddings and self.caption_collection:
            self.caption_collection.upsert(
                ids=caption_ids,
                embeddings=caption_embeddings,
                documents=caption_docs,
                metadatas=caption_metadata,
            )

        return IngestionResult(indexed=len(embeddings), skipped=skipped)
