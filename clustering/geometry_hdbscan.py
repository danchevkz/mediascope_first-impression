from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from common import (
    clustering_summary,
    write_cluster_csv,
    write_json,
)


GRID_ROWS = 8
GRID_COLUMNS = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster screenshots using only normalized OCR-box geometry."
    )
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--pca-components", type=int, default=30)
    parser.add_argument("--min-cluster-size", type=int, default=10)
    parser.add_argument("--min-samples", type=int, default=5)
    return parser.parse_args()


def read_records(path: Path, limit: int) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            if line.strip():
                record = json.loads(line)
                records[str(record["image"])] = record
    ordered = [records[name] for name in sorted(records)]
    return ordered if limit < 0 else ordered[:limit]


def safe_statistics(values: np.ndarray) -> list[float]:
    if values.size == 0:
        return [0.0] * 7
    return [
        float(values.mean()),
        float(values.std()),
        float(values.min()),
        float(np.quantile(values, 0.25)),
        float(np.median(values)),
        float(np.quantile(values, 0.75)),
        float(values.max()),
    ]


def box_features(record: dict[str, Any]) -> np.ndarray:
    width = max(1.0, float(record["width"]))
    height = max(1.0, float(record["height"]))
    normalized: list[list[float]] = []
    for item in record.get("boxes", []):
        if float(item.get("score", 1.0)) < 0.35:
            continue
        x1, y1, x2, y2 = (float(value) for value in item["bbox"])
        normalized.append([x1 / width, y1 / height, x2 / width, y2 / height])
    boxes = np.asarray(normalized, dtype=np.float32).reshape(-1, 4)

    if len(boxes):
        box_widths = np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
        box_heights = np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
        areas = box_widths * box_heights
        centers_x = (boxes[:, 0] + boxes[:, 2]) * 0.5
        centers_y = (boxes[:, 1] + boxes[:, 3]) * 0.5
        aspects = box_widths / np.maximum(box_heights, 1e-6)
        confidences = np.asarray(
            [
                float(item.get("score", 1.0))
                for item in record.get("boxes", [])
                if float(item.get("score", 1.0)) >= 0.35
            ],
            dtype=np.float32,
        )
    else:
        box_widths = box_heights = areas = centers_x = centers_y = aspects = np.empty(0)
        confidences = np.empty(0)

    center_grid, _, _ = np.histogram2d(
        centers_y,
        centers_x,
        bins=(GRID_ROWS, GRID_COLUMNS),
        range=((0.0, 1.0), (0.0, 1.0)),
    )
    if len(boxes):
        center_grid /= len(boxes)

    coverage_grid = np.zeros((GRID_ROWS, GRID_COLUMNS), dtype=np.float32)
    cell_height = 1.0 / GRID_ROWS
    cell_width = 1.0 / GRID_COLUMNS
    for x1, y1, x2, y2 in boxes:
        first_row = max(0, min(GRID_ROWS - 1, int(y1 * GRID_ROWS)))
        last_row = max(0, min(GRID_ROWS - 1, int(np.nextafter(y2, 0.0) * GRID_ROWS)))
        first_column = max(0, min(GRID_COLUMNS - 1, int(x1 * GRID_COLUMNS)))
        last_column = max(
            0,
            min(GRID_COLUMNS - 1, int(np.nextafter(x2, 0.0) * GRID_COLUMNS)),
        )
        for row in range(first_row, last_row + 1):
            top = row * cell_height
            bottom = top + cell_height
            overlap_height = max(0.0, min(y2, bottom) - max(y1, top))
            for column in range(first_column, last_column + 1):
                left = column * cell_width
                right = left + cell_width
                overlap_width = max(0.0, min(x2, right) - max(x1, left))
                coverage_grid[row, column] += (
                    overlap_width * overlap_height / (cell_width * cell_height)
                )
    coverage_grid = np.clip(coverage_grid, 0.0, 1.0)

    sorted_y = np.sort(centers_y)
    vertical_gaps = np.diff(sorted_y) if sorted_y.size >= 2 else np.empty(0)
    global_features = [
        float(np.log1p(len(boxes))),
        float(np.clip(areas.sum(), 0.0, 1.0)),
        *safe_statistics(box_widths),
        *safe_statistics(box_heights),
        *safe_statistics(areas),
        *safe_statistics(centers_x),
        *safe_statistics(centers_y),
        *safe_statistics(aspects),
        *safe_statistics(vertical_gaps),
        *safe_statistics(confidences),
    ]
    projections = np.concatenate(
        [
            center_grid.sum(axis=1),
            center_grid.sum(axis=0),
            coverage_grid.mean(axis=1),
            coverage_grid.mean(axis=0),
        ]
    )
    return np.concatenate(
        [
            np.asarray(global_features, dtype=np.float32),
            center_grid.reshape(-1).astype(np.float32),
            coverage_grid.reshape(-1),
            projections.astype(np.float32),
        ]
    )


def main() -> None:
    args = parse_args()
    records = read_records(args.records, args.limit)
    if len(records) < 3:
        raise RuntimeError("At least three OCR records are required")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    features = np.stack([box_features(record) for record in records])
    scaled = StandardScaler().fit_transform(features)
    component_count = min(args.pca_components, scaled.shape[1], len(records) - 1)
    reduced = PCA(n_components=component_count, random_state=42).fit_transform(scaled)
    cluster_size = min(args.min_cluster_size, max(2, len(records) - 1))
    min_samples = min(args.min_samples, cluster_size)
    labels = HDBSCAN(
        min_cluster_size=cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        n_jobs=1,
        copy=True,
    ).fit_predict(reduced)

    np.save(args.output_dir / "geometry_features.npy", features)
    np.save(args.output_dir / "geometry_pca.npy", reduced)
    names = [str(record["image"]) for record in records]
    write_cluster_csv(
        args.output_dir / "clusters.csv",
        names,
        labels,
        reduced[:, :2],
    )
    summary = clustering_summary(labels, features=reduced)
    summary.update(
        {
            "method": "ocr_box_geometry_pca_hdbscan",
            "feature_dimensions": int(features.shape[1]),
            "pca_components": component_count,
            "min_cluster_size": cluster_size,
            "min_samples": min_samples,
            "grid_rows": GRID_ROWS,
            "grid_columns": GRID_COLUMNS,
        }
    )
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
