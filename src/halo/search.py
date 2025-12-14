"""Search utilities for Halo."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import BinaryIO, Dict, Iterable, List, Optional, Tuple, Union

from chromadb import PersistentClient
from PIL import Image

from .config import get_config
from .embeddings import embed_image, embed_text
from .llm_utils import expand_query


@dataclass
class SearchResult:
    path: str
    score: float
    metadata: dict


class PhotoSearcher:
    def __init__(self, caption_weight: Optional[float] = None) -> None:
        cfg = get_config()
        self.cfg = cfg
        self.caption_weight = caption_weight if caption_weight is not None else cfg.caption_weight
        self.client = PersistentClient(path=str(cfg.chroma_db_dir))
        collection_kwargs = {"metadata": {"hnsw:space": "cosine"}}
        self.image_collection = self.client.get_or_create_collection(
            cfg.image_collection,
            metadata=collection_kwargs["metadata"],
        )
        self.caption_collection = None
        if cfg.enable_blip:
            self.caption_collection = self.client.get_or_create_collection(
                cfg.caption_collection,
                metadata=collection_kwargs["metadata"],
            )

    # ---------------------------------------------------------------------
    # Public search APIs
    # ---------------------------------------------------------------------
    def search_text(
        self,
        query: str,
        *,
        k: Optional[int] = None,
        expand: bool = True,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        geo_box: Optional[Tuple[float, float, float, float]] = None,
    ) -> List[SearchResult]:
        max_hits = k or self.cfg.default_top_k
        enriched = expand_query(query) if expand else query
        query_vector = embed_text(enriched).tolist()
        return self._hybrid_search(
            query_vector=query_vector,
            max_hits=max_hits,
            start_date=start_date,
            end_date=end_date,
            geo_box=geo_box,
        )

    def search_by_image(
        self,
        image_source: Union[str, Image.Image, BinaryIO],
        *,
        k: Optional[int] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        geo_box: Optional[Tuple[float, float, float, float]] = None,
    ) -> List[SearchResult]:
        max_hits = k or self.cfg.default_top_k
        vector = embed_image(image_source).tolist()
        return self._hybrid_search(
            query_vector=vector,
            max_hits=max_hits,
            start_date=start_date,
            end_date=end_date,
            geo_box=geo_box,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _hybrid_search(
        self,
        *,
        query_vector: List[float],
        max_hits: int,
        start_date: Optional[date],
        end_date: Optional[date],
        geo_box: Optional[Tuple[float, float, float, float]],
    ) -> List[SearchResult]:
        image_hits = self._query_collection(self.image_collection, query_vector, max_hits * 3)
        caption_hits: List[SearchResult] = []
        if self.caption_collection and self.caption_weight > 0:
            caption_hits = self._query_collection(self.caption_collection, query_vector, max_hits * 3)

        combined: Dict[str, Dict[str, object]] = {}

        def accumulate(hits: Iterable[SearchResult], weight: float) -> None:
            for hit in hits:
                entry = combined.setdefault(hit.path, {"score": 0.0, "metadata": hit.metadata})
                entry["score"] = float(entry["score"]) + weight * hit.score
                if hit.metadata and hit.metadata.get("caption"):
                    entry["metadata"] = hit.metadata

        image_weight = 1.0 - min(max(self.caption_weight, 0.0), 0.95)
        accumulate(image_hits, image_weight if image_hits else 1.0)
        accumulate(caption_hits, min(max(self.caption_weight, 0.0), 1.0))

        merged = [
            SearchResult(path=path, score=values["score"], metadata=values.get("metadata", {}))
            for path, values in combined.items()
        ]

        filtered = [hit for hit in merged if self._passes_filters(hit.metadata, start_date, end_date, geo_box)]
        filtered.sort(key=lambda h: h.score, reverse=True)
        return filtered[:max_hits]

    def _query_collection(self, collection, query_vector: List[float], limit: int) -> List[SearchResult]:
        if not collection:
            return []
        results = collection.query(query_embeddings=[query_vector], n_results=max(1, limit))
        if not results["ids"]:
            return []
        hits: List[SearchResult] = []
        distances = results.get("distances", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        for idx, doc in enumerate(docs):
            distance = distances[idx]
            score = max(0.0, 1.0 - distance)  # cosine distance -> similarity
            hits.append(SearchResult(path=doc, score=score, metadata=metas[idx] or {}))
        return hits

    @staticmethod
    def _passes_filters(
        metadata: dict,
        start_date: Optional[date],
        end_date: Optional[date],
        geo_box: Optional[Tuple[float, float, float, float]],
    ) -> bool:
        if start_date or end_date:
            dt_iso = metadata.get("datetime_iso")
            if not dt_iso:
                return False
            try:
                dt_value = datetime.fromisoformat(dt_iso)
            except ValueError:
                return False
            if start_date and dt_value.date() < start_date:
                return False
            if end_date and dt_value.date() > end_date:
                return False

        if geo_box:
            min_lat, max_lat, min_lon, max_lon = geo_box
            lat = metadata.get("latitude")
            lon = metadata.get("longitude")
            if lat is None or lon is None:
                return False
            if not (min_lat <= lat <= max_lat and min_lon <= lon <= max_lon):
                return False

        return True
