from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA

from common import (
    IMAGE_INPUT,
    clustering_summary,
    list_images,
    validate_sources,
    write_cluster_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster DINOv2 screenshot embeddings with PCA and HDBSCAN."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="facebook/dinov2-small")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--pca-components", type=int, default=50)
    parser.add_argument("--min-cluster-size", type=int, default=10)
    parser.add_argument("--min-samples", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def save_mask(path: Path, mask: np.ndarray) -> None:
    temporary = path.with_suffix(".tmp.npy")
    np.save(temporary, mask)
    temporary.replace(path)


def initialize_storage(
    output_dir: Path,
    names: list[str],
    dimension: int,
    resume: bool,
    model_id: str,
) -> tuple[np.memmap, np.ndarray]:
    names_path = output_dir / "filenames.json"
    embeddings_path = output_dir / "embeddings.npy"
    completed_path = output_dir / "completed.npy"
    settings_path = output_dir / "extraction.json"
    settings = {"input": IMAGE_INPUT, "model": model_id}
    if resume and embeddings_path.exists():
        if not settings_path.exists() or json.loads(settings_path.read_text(encoding="utf-8")) != settings:
            raise ValueError("Embedding extraction settings differ; use a fresh output directory")
    if resume and names_path.exists() and embeddings_path.exists() and completed_path.exists():
        stored_names = json.loads(names_path.read_text(encoding="utf-8"))
        if stored_names != names:
            raise RuntimeError("Input image list changed; cannot resume safely")
        embeddings = np.lib.format.open_memmap(embeddings_path, mode="r+")
        completed = np.load(completed_path)
        if embeddings.shape != (len(names), dimension) or completed.shape != (len(names),):
            raise RuntimeError("Stored embedding shape does not match the current run")
        return embeddings, completed.astype(bool)

    write_json(settings_path, settings)
    names_path.write_text(
        json.dumps(names, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    embeddings = np.lib.format.open_memmap(
        embeddings_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(names), dimension),
    )
    completed = np.zeros(len(names), dtype=bool)
    save_mask(completed_path, completed)
    return embeddings, completed


def extract_embeddings(args: argparse.Namespace, paths: list[Path]) -> np.ndarray:
    from transformers import AutoImageProcessor, AutoModel

    validate_sources(paths, args.output_dir / 'sources.json', args.resume and (args.output_dir / 'embeddings.npy').exists())
    torch.set_num_threads(args.threads)
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).eval().to("cpu")
    dimension = int(model.config.hidden_size)
    names = [path.name for path in paths]
    embeddings, completed = initialize_storage(
        args.output_dir,
        names,
        dimension,
        args.resume,
        args.model,
    )
    pending = np.flatnonzero(~completed)

    for start in range(0, len(pending), args.batch_size):
        indices = pending[start : start + args.batch_size]
        images: list[Image.Image] = []
        for index in indices:
            with Image.open(paths[int(index)]) as source:
                images.append(source.convert("RGB"))
        inputs = processor(images=images, return_tensors="pt")
        with torch.inference_mode():
            output = model(**inputs).last_hidden_state[:, 0, :]
            output = torch.nn.functional.normalize(output, dim=1)
        embeddings[indices] = output.cpu().numpy().astype(np.float32)
        completed[indices] = True
        embeddings.flush()
        save_mask(args.output_dir / "completed.npy", completed)
        print(
            f"embed {int(completed.sum())}/{len(paths)} {paths[int(indices[-1])].name}",
            flush=True,
        )
    return np.asarray(embeddings, dtype=np.float32)


def write_representatives(
    path: Path,
    names: list[str],
    labels: np.ndarray,
    features: np.ndarray,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=["cluster", "rank", "image", "distance_to_centroid"],
        )
        writer.writeheader()
        for cluster_id in sorted(set(labels.tolist()) - {-1}):
            indices = np.flatnonzero(labels == cluster_id)
            centroid = features[indices].mean(axis=0)
            distances = np.linalg.norm(features[indices] - centroid, axis=1)
            for rank, local_index in enumerate(np.argsort(distances)[:5], start=1):
                image_index = int(indices[int(local_index)])
                writer.writerow(
                    {
                        "cluster": int(cluster_id),
                        "rank": rank,
                        "image": names[image_index],
                        "distance_to_centroid": float(distances[int(local_index)]),
                    }
                )


def main() -> None:
    args = parse_args()
    paths = list_images(args.input_dir, args.limit)
    if len(paths) < 3:
        raise RuntimeError("At least three images are required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    embeddings = extract_embeddings(args, paths)

    component_count = min(args.pca_components, embeddings.shape[1], len(paths) - 1)
    reduced = PCA(n_components=component_count, random_state=42).fit_transform(embeddings)
    cluster_size = min(args.min_cluster_size, max(2, len(paths) - 1))
    min_samples = min(args.min_samples, cluster_size)
    labels = HDBSCAN(
        min_cluster_size=cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        n_jobs=1,
        copy=True,
    ).fit_predict(reduced)

    coordinates = reduced[:, :2]
    names = [path.name for path in paths]
    write_cluster_csv(args.output_dir / "clusters.csv", names, labels, coordinates)
    write_representatives(
        args.output_dir / "representatives.csv",
        names,
        labels,
        reduced,
    )
    summary = clustering_summary(labels, features=reduced)
    summary.update(
        {
            "method": "dinov2_pca_hdbscan",
            "model": args.model,
            "pca_components": component_count,
            "min_cluster_size": cluster_size,
            "min_samples": min_samples,
        }
    )
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
