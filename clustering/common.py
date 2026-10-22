from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


load_dotenv(Path(__file__).resolve().parents[1] / '.env', override=False, encoding='utf-8-sig')

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
IMAGE_INPUT = "full_page_original_size"


def list_images(input_dir: Path, limit: int = -1) -> list[Path]:
    roots = [input_dir]
    if all((input_dir / split / "metadata.csv").is_file() for split in ("train", "test")):
        roots = [input_dir / "train", input_dir / "test"]
    paths = sorted(
        path
        for root in roots
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    paths = paths if limit < 0 else paths[:limit]
    if len({path.name for path in paths}) != len(paths):
        raise ValueError('Clustering requires unique screenshot filenames')
    return paths


def file_hash(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def validate_sources(paths, inventory_path, resume):
    inventory = [{'image': path.name, 'sha256': file_hash(path)} for path in paths]
    if len({item['image'] for item in inventory}) != len(inventory):
        raise ValueError('Duplicate screenshot filenames')
    if resume:
        if not inventory_path.is_file() or json.loads(inventory_path.read_text(encoding='utf-8')) != inventory:
            raise ValueError('Clustering images changed; use a fresh output directory')
    else:
        write_json(inventory_path, inventory)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def clustering_summary(
    labels: np.ndarray,
    features: np.ndarray | None = None,
    distances: np.ndarray | None = None,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int32)
    non_noise = labels >= 0
    cluster_ids, counts = np.unique(labels[non_noise], return_counts=True)
    summary: dict[str, Any] = {
        "samples": int(labels.size),
        "clusters": int(cluster_ids.size),
        "noise_samples": int((~non_noise).sum()),
        "noise_fraction": float((~non_noise).mean()) if labels.size else 0.0,
        "cluster_sizes": {
            str(int(cluster_id)): int(count)
            for cluster_id, count in zip(cluster_ids, counts, strict=True)
        },
    }

    kept_labels = labels[non_noise]
    valid = cluster_ids.size >= 2 and kept_labels.size > cluster_ids.size
    if not valid:
        return summary

    if distances is not None:
        kept_distances = distances[np.ix_(non_noise, non_noise)]
        summary["silhouette"] = float(
            silhouette_score(kept_distances, kept_labels, metric="precomputed")
        )
    elif features is not None:
        kept_features = features[non_noise]
        summary["silhouette"] = float(silhouette_score(kept_features, kept_labels))
        summary["davies_bouldin"] = float(
            davies_bouldin_score(kept_features, kept_labels)
        )
        summary["calinski_harabasz"] = float(
            calinski_harabasz_score(kept_features, kept_labels)
        )
    return summary


def write_cluster_csv(
    path: Path,
    image_names: Sequence[str],
    labels: np.ndarray,
    coordinates: np.ndarray | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        fieldnames = ["image", "cluster", "is_noise"]
        if coordinates is not None:
            fieldnames.extend(["x", "y"])
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for index, (name, label) in enumerate(zip(image_names, labels, strict=True)):
            row: dict[str, Any] = {
                "image": name,
                "cluster": int(label),
                "is_noise": int(label < 0),
            }
            if coordinates is not None:
                row["x"] = float(coordinates[index, 0])
                row["y"] = float(coordinates[index, 1])
            writer.writerow(row)
