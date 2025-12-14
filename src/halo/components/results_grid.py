"""Optional React-powered results grid component."""
from __future__ import annotations

import os
import base64
import mimetypes
from pathlib import Path
from typing import Iterable, List, Optional

import streamlit.components.v1 as components


def _component_path() -> Optional[Path]:
    """Return built component directory if present (defensive)."""
    here = Path(__file__).resolve()
    candidates = [
        here.parent.parent.parent.parent / "react_components" / "result-grid" / "dist",
        here.parent.parent.parent / "react_components" / "result-grid" / "dist",
        here.parent.parent / "react_components" / "result-grid" / "dist",
    ]
    for dist in candidates:
        if dist.exists() and dist.is_dir():
            return dist
    return None


def _declare_component():
    dev_url = os.environ.get("RESULT_GRID_DEV_URL")
    if dev_url:
        return components.declare_component("result_grid", url=dev_url)

    dist = _component_path()
    if dist:
        return components.declare_component("result_grid", path=str(dist))

    return None


_RESULT_GRID = None


def _to_data_url(path: str) -> Optional[str]:
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        return None
    try:
        data = file_path.read_bytes()
    except OSError:
        return None
    mime, _ = mimetypes.guess_type(file_path.name)
    mime = mime or "image/jpeg"
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _serialize_results(results: Iterable[object]) -> List[dict]:
    payload: List[dict] = []
    for item in results:
        path = getattr(item, "path", None) or (item.get("path") if isinstance(item, dict) else None)
        metadata = getattr(item, "metadata", None) or (item.get("metadata") if isinstance(item, dict) else {}) or {}
        url = metadata.get("url") or _to_data_url(path) if path else None
        payload.append(
            {
                "path": path or "",
                "url": url or "",
                "score": getattr(item, "score", None) if not isinstance(item, dict) else item.get("score"),
                "caption": metadata.get("caption") or getattr(item, "caption", None) or Path(path).name if path else "",
                "metadata": metadata,
            }
        )
    return payload


def render_results_grid(
    results: Iterable[object],
    *,
    columns: int = 3,
    thumb_height: int = 220,
    show_score: bool = True,
    initial_selection: Optional[List[str]] = None,
    key: str = "results-grid",
):
    """Render the React grid if the component build is available; otherwise return None."""
    global _RESULT_GRID
    if _RESULT_GRID is None:
        _RESULT_GRID = _declare_component()
    if not _RESULT_GRID:
        return None

    serialized = _serialize_results(results)
    return _RESULT_GRID(
        results=serialized,
        columns=columns,
        thumbHeight=thumb_height,
        showScore=show_score,
        initialSelection=initial_selection or [],
        key=key,
        default=[],
    )
