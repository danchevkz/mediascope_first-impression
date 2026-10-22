from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import HDBSCAN

from common import clustering_summary, file_hash, write_cluster_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply HDBSCAN directly to OCR-visual MinHash distances."
    )
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--distances", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-cluster-size", type=int, default=10)
    parser.add_argument("--min-samples", type=int, default=5)
    return parser.parse_args()


def read_names(path: Path) -> list[str]:
    names: set[str] = set()
    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            if line.strip():
                names.add(str(json.loads(line)["image"]))
    return sorted(names)


def load_distances(records, path):
    names = read_names(records)
    inventory = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    expected = {'images': names, 'records_sha256': file_hash(records), 'distances_sha256': file_hash(path)}
    if inventory != expected:
        raise ValueError('Distance matrix and OCR record identities differ')
    distances = np.load(path, mmap_mode='r', allow_pickle=False)
    if distances.shape != (len(names), len(names)) or not np.isfinite(distances).all():
        raise ValueError('Invalid distance matrix')
    if not np.allclose(distances, distances.T) or not np.allclose(np.diag(distances), 0) or (distances < 0).any():
        raise ValueError('Invalid pairwise distances')
    return names, distances


def main() -> None:
    args = parse_args()
    names, distances = load_distances(args.records, args.distances)
    labels = HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric="precomputed",
        cluster_selection_method="eom",
        n_jobs=1,
        copy=True,
    ).fit_predict(distances)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_cluster_csv(args.output_dir / "clusters.csv", names, labels)
    summary = clustering_summary(labels, distances=np.asarray(distances))
    summary.update(
        {
            "method": "ocr_visual_minhash_hdbscan",
            "min_cluster_size": args.min_cluster_size,
            "min_samples": args.min_samples,
            "clustering_input": "combined_minhash_distance",
        }
    )
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
