"""Download a richer placeholder photo set for Halo testing."""
from __future__ import annotations

import argparse
import re
from pathlib import Path
import shutil
from typing import List, Tuple

import requests

HEADERS = {"User-Agent": "HaloSampleDownloader/2.1 (https://example.com)"}
PICSUM_LIST_ENDPOINT = "https://picsum.photos/v2/list"


def slugify(value: str) -> str:
    slug = re.sub(r"_+", "_", "".join(ch.lower() if ch.isalnum() else "_" for ch in value))
    slug = slug.strip("_")
    return slug or "image"


def fetch_catalog(count: int) -> List[Tuple[int, str]]:
    params = {"page": 1, "limit": max(1, min(count, 100))}
    response = requests.get(PICSUM_LIST_ENDPOINT, params=params, timeout=60, headers=HEADERS)
    response.raise_for_status()
    items = response.json()
    catalog: List[Tuple[int, str]] = []
    for item in items:
        image_id = int(item["id"])
        author = item.get("author", "image")
        catalog.append((image_id, slugify(author)))
    return catalog


def build_dataset(limit: int, width: int, height: int) -> List[Tuple[str, str]]:
    catalog = fetch_catalog(limit)
    return [
        (f"https://picsum.photos/id/{image_id}/{width}/{height}", f"{slug}_{image_id}.jpg")
        for image_id, slug in catalog
    ]


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"✔ Already exists: {dest.name}")
        return
    print(f"↓ Downloading {url} -> {dest.name}")
    response = requests.get(url, timeout=60, headers=HEADERS)
    response.raise_for_status()
    dest.write_bytes(response.content)
    size_kb = len(response.content) / 1024
    print(f"  Saved {dest.name} ({size_kb:.1f} KB)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=24,
        help="Number of photos to download from Picsum (1-100).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=900,
        help="Width in pixels for each placeholder image.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=600,
        help="Height in pixels for each placeholder image.",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "photos" / "sample_dataset",
        help="Destination folder for the dataset.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing files in the target directory before downloading.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = build_dataset(args.limit, args.width, args.height)

    target_dir = args.target
    if args.clean and target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    for url, filename in dataset:
        dest = target_dir / filename
        _download(url, dest)

    print(f"Dataset ready in {target_dir}")


if __name__ == "__main__":
    main()
