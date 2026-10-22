from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

from common import (
    IMAGE_INPUT,
    clustering_summary,
    list_images,
    file_hash,
    validate_sources,
    write_cluster_csv,
    write_json,
)


PRIME = np.uint64(4_294_967_291)
TOKEN_RE = re.compile(r"[a-z0-9]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster screenshots using OCR-text and visual MinHash signatures."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--num-perm", type=int, default=128)
    parser.add_argument("--eps", type=float, default=0.42)
    parser.add_argument("--min-samples", type=int, default=3)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--text-recognition-batch-size", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def result_dict(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        value = item
    else:
        value = getattr(item, "json", None)
        if callable(value):
            value = value()
        if not isinstance(value, dict):
            value = getattr(item, "res", {})
    if isinstance(value, dict) and isinstance(value.get("res"), dict):
        return value["res"]
    return value if isinstance(value, dict) else {}


def clamp_box(box: Any, width: int, height: int) -> list[int] | None:
    array = np.asarray(box, dtype=np.float32)
    if array.size < 4:
        return None
    if array.ndim >= 2 and array.shape[-1] == 2:
        values = [
            float(np.min(array[..., 0])),
            float(np.min(array[..., 1])),
            float(np.max(array[..., 0])),
            float(np.max(array[..., 1])),
        ]
    else:
        values = array.reshape(-1)[:4].tolist()
    x1, y1, x2, y2 = values
    result = [
        max(0, min(width - 1, round(float(x1)))),
        max(0, min(height - 1, round(float(y1)))),
        max(1, min(width, round(float(x2)))),
        max(1, min(height, round(float(y2)))),
    ]
    return result if result[2] > result[0] and result[3] > result[1] else None


def text_tokens(texts: Iterable[str]) -> set[str]:
    words = TOKEN_RE.findall(" ".join(texts).lower())
    tokens = {f"u:{word}" for word in words if len(word) >= 4}
    tokens.update(
        "b:" + "_".join(words[index : index + 2])
        for index in range(max(0, len(words) - 1))
    )
    tokens.update(
        "t:" + "_".join(words[index : index + 3])
        for index in range(max(0, len(words) - 2))
    )
    return tokens


def visual_tokens(image: Image.Image) -> set[str]:
    grid = np.asarray(
        image.convert("RGB").resize((32, 18), Image.Resampling.BOX),
        dtype=np.uint8,
    )
    quantized = grid // 32
    gray = (
        0.299 * grid[..., 0] + 0.587 * grid[..., 1] + 0.114 * grid[..., 2]
    )
    horizontal = np.pad(np.abs(np.diff(gray, axis=1)), ((0, 0), (0, 1)))
    vertical = np.pad(np.abs(np.diff(gray, axis=0)), ((0, 1), (0, 0)))
    edge = np.minimum(7, ((horizontal + vertical) // 32).astype(np.uint8))

    tokens: set[str] = set()
    for row in range(18):
        for column in range(32):
            red, green, blue = quantized[row, column]
            tokens.add(
                f"c:{row}:{column}:{int(red)}:{int(green)}:{int(blue)}:{int(edge[row, column])}"
            )

    coarse = np.asarray(
        image.convert("L").resize((16, 9), Image.Resampling.BOX),
        dtype=np.uint8,
    ) // 32
    for row in range(8):
        for column in range(15):
            patch = coarse[row : row + 2, column : column + 2].reshape(-1)
            tokens.add(
                f"p:{row}:{column}:" + ":".join(str(int(value)) for value in patch)
            )
    return tokens


def permutation_coefficients(num_perm: int) -> tuple[np.ndarray, np.ndarray]:
    random = np.random.default_rng(20260921)
    a = random.integers(1, int(PRIME), size=num_perm, dtype=np.uint64)
    b = random.integers(0, int(PRIME), size=num_perm, dtype=np.uint64)
    return a, b


def minhash_signature(
    tokens: set[str],
    a: np.ndarray,
    b: np.ndarray,
) -> list[int]:
    if not tokens:
        return [int(PRIME)] * len(a)
    hashes = np.fromiter(
        (
            int.from_bytes(
                hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(),
                "little",
            )
            % int(PRIME)
            for token in sorted(tokens)
        ),
        dtype=np.uint64,
    )
    signature = np.full(len(a), PRIME, dtype=np.uint64)
    for start in range(0, len(hashes), 1024):
        values = hashes[start : start + 1024, None]
        candidates = (values * a[None, :] + b[None, :]) % PRIME
        signature = np.minimum(signature, candidates.min(axis=0))
    return [int(value) for value in signature]


def load_completed(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    with path.open("r", encoding="utf-8") as input_file:
        return {
            str(json.loads(line)["image"])
            for line in input_file
            if line.strip()
        }


def extract_records(
    paths: list[Path],
    records_path: Path,
    num_perm: int,
    device: str,
    text_recognition_batch_size: int,
    resume: bool,
) -> None:
    settings_path = records_path.with_suffix(".settings.json")
    settings = {"input": IMAGE_INPUT, "num_perm": num_perm,
                "detector": "PP-OCRv6_medium_det", "recognizer": "PP-OCRv6_medium_rec"}
    if resume and records_path.exists():
        if not settings_path.exists() or json.loads(settings_path.read_text(encoding="utf-8")) != settings:
            raise ValueError("OCR extraction settings differ; use a fresh output directory")
    validate_sources(paths, records_path.with_suffix('.sources.json'), resume and records_path.exists())
    write_json(settings_path, settings)
    from paddleocr import PaddleOCR
    completed = load_completed(records_path) if resume else set()
    pending = [path for path in paths if path.name not in completed]
    records_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if resume and records_path.exists() else "w"
    a, b = permutation_coefficients(num_perm)
    model = PaddleOCR(
        text_detection_model_name="PP-OCRv6_medium_det",
        text_recognition_model_name="PP-OCRv6_medium_rec",
        device="gpu:0" if device == "gpu" else "cpu",
        engine="paddle_dynamic",
        enable_mkldnn=False,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_recognition_batch_size=text_recognition_batch_size,
    )

    with records_path.open(mode, encoding="utf-8") as output_file:
        for index, path in enumerate(pending, start=1):
            with Image.open(path) as source:
                source_width, source_height = source.size
                image = source.convert("RGB")
                width, height = image.size
                visual = minhash_signature(visual_tokens(image), a, b)

            boxes: list[dict[str, Any]] = []
            texts: list[str] = []
            for result in model.predict(np.asarray(image)):
                data = result_dict(result)
                recognized = list(data.get("rec_texts", []))
                scores = list(data.get("rec_scores", []))
                polygons = data.get("rec_polys", data.get("dt_polys", []))
                for item_index, polygon in enumerate(polygons):
                    box = clamp_box(polygon, width, height)
                    if box is None:
                        continue
                    text = (
                        str(recognized[item_index])
                        if item_index < len(recognized)
                        else ""
                    )
                    score = (
                        float(scores[item_index])
                        if item_index < len(scores)
                        else 1.0
                    )
                    boxes.append({"bbox": box, "score": score, "text": text})
                    if text.strip() and score >= 0.35:
                        texts.append(text)

            tokens = text_tokens(texts)
            record = {
                "image": path.name,
                "width": width,
                "height": height,
                "source_width": source_width,
                "source_height": source_height,
                "texts": texts,
                "boxes": boxes,
                "text_token_count": len(tokens),
                "text_signature": minhash_signature(tokens, a, b),
                "visual_signature": visual,
            }
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_file.flush()
            print(
                f"extract {index}/{len(pending)} {path.name} "
                f"boxes={len(boxes)} text_tokens={len(tokens)}",
                flush=True,
            )


def read_records(path: Path, names: set[str]) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            record = json.loads(line)
            if record["image"] in names:
                records[record["image"]] = record
    missing = names.difference(records)
    if missing:
        raise RuntimeError(f"Missing OCR records for {len(missing)} images")
    return [records[name] for name in sorted(names)]


def compact_records(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(".tmp.jsonl")
    with temporary.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def combined_distances(records: list[dict[str, Any]]) -> np.ndarray:
    text = np.asarray([record["text_signature"] for record in records], dtype=np.uint64)
    visual = np.asarray(
        [record["visual_signature"] for record in records], dtype=np.uint64
    )
    has_text = np.asarray(
        [record["text_token_count"] > 0 for record in records], dtype=bool
    )
    count = len(records)
    distances = np.empty((count, count), dtype=np.float32)
    for start in range(0, count, 128):
        stop = min(count, start + 128)
        visual_similarity = (
            visual[start:stop, None, :] == visual[None, :, :]
        ).mean(axis=2)
        text_similarity = (
            text[start:stop, None, :] == text[None, :, :]
        ).mean(axis=2)
        both_have_text = has_text[start:stop, None] & has_text[None, :]
        similarity = np.where(
            both_have_text,
            0.55 * text_similarity + 0.45 * visual_similarity,
            visual_similarity,
        )
        distances[start:stop] = 1.0 - similarity.astype(np.float32)
    distances = (distances + distances.T) * 0.5
    np.fill_diagonal(distances, 0.0)
    return distances


def write_nearest_pairs(
    path: Path,
    names: list[str],
    distances: np.ndarray,
    limit: int = 250,
) -> None:
    row_indices, column_indices = np.triu_indices(len(names), k=1)
    values = distances[row_indices, column_indices]
    order = np.argsort(values)[:limit]
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=["image_a", "image_b", "combined_similarity"],
        )
        writer.writeheader()
        for index in order:
            writer.writerow(
                {
                    "image_a": names[int(row_indices[index])],
                    "image_b": names[int(column_indices[index])],
                    "combined_similarity": float(1.0 - values[index]),
                }
            )


def main() -> None:
    args = parse_args()
    paths = list_images(args.input_dir, args.limit)
    if len(paths) < 2:
        raise RuntimeError("At least two images are required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / "ocr_minhash_records.jsonl"
    extract_records(
        paths,
        records_path,
        args.num_perm,
        args.device,
        args.text_recognition_batch_size,
        args.resume,
    )

    records = read_records(records_path, {path.name for path in paths})
    compact_records(records_path, records)
    names = [str(record["image"]) for record in records]
    distances = combined_distances(records)
    np.save(args.output_dir / "combined_distances.npy", distances)
    write_json(args.output_dir / 'combined_distances.json', {
        'images': names, 'records_sha256': file_hash(records_path),
        'distances_sha256': file_hash(args.output_dir / 'combined_distances.npy')})
    labels = DBSCAN(
        eps=args.eps,
        min_samples=args.min_samples,
        metric="precomputed",
        n_jobs=1,
    ).fit_predict(distances)

    coordinates = PCA(n_components=2, random_state=42).fit_transform(distances)
    write_cluster_csv(args.output_dir / "clusters.csv", names, labels, coordinates)
    write_nearest_pairs(args.output_dir / "nearest_pairs.csv", names, distances)
    summary = clustering_summary(labels, distances=distances)
    summary.update(
        {
            "method": "ocr_visual_minhash_dbscan",
            "num_perm": args.num_perm,
            "text_weight": 0.55,
            "visual_weight": 0.45,
            "eps": args.eps,
            "min_samples": args.min_samples,
            "extraction_device": args.device,
            "text_recognition_batch_size": args.text_recognition_batch_size,
        }
    )
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
