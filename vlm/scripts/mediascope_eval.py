from __future__ import annotations

import argparse
import csv
import os
import json
import time
import traceback
import uuid
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mediascope.data import read_manifest

from run_config import (IMAGE_DEFAULTS, THINKING_MODES, environment_info, json_hash,
                        load_adapter_run_config, resolve_image_settings, sha256_file, source_info)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it
    tqdm.write = lambda *a, **k: print(*a)


FACTUALITY_ORDER = ["LOW", "MIXED", "HIGH"]

BIAS_ORDER = [
    "LEFT",
    "LEFT-CENTER",
    "LEAST BIASED",
    "RIGHT-CENTER",
    "RIGHT",
]


FACT_COLLAPSE = {
    "VERY LOW": "LOW", "LOW": "LOW",
    "MIXED": "MIXED",
    "MOSTLY FACTUAL": "HIGH", "HIGH": "HIGH", "VERY HIGH": "HIGH",
}
BIAS_COLLAPSE = {
    "EXTREME LEFT": "LEFT", "LEFT": "LEFT",
    "LEFT-CENTER": "LEFT-CENTER",
    "LEAST BIASED": "LEAST BIASED",
    "RIGHT-CENTER": "RIGHT-CENTER",
    "RIGHT": "RIGHT", "EXTREME RIGHT": "RIGHT",
}

TASKS = {
    "factuality": {"order": FACTUALITY_ORDER, "extra": [], "collapse": FACT_COLLAPSE},
    "bias": {"order": BIAS_ORDER, "extra": [], "collapse": BIAS_COLLAPSE},
}


@dataclass
class Sample:
    media_name: str
    url: str
    image_path: str
    country: str
    factuality: str
    bias: str
    image: str = ''
    group_id: str = ''


def load_samples(csv_path: str, image_root: str | None, tasks=("factuality", "bias"), expected_split=None):
    rows = read_manifest(csv_path, image_root, expected_split=expected_split)
    samples = [Sample(row['media_name'].strip(), row.get('url', '').strip(), row['image_path'],
                      row.get('country', '').strip(), row['true_label_factuality'], row['true_label_bias'],
                      row['image'], row.get('group_id', '')) for row in rows]
    return samples, []


def check_training_overlap(training, samples):
    for field in ('image_path', 'image_sha256', 'media_name', 'url', 'image', 'group_id'):
        trained = {row[field] for row in training if row.get(field)}
        evaluated = {sha256_file(s.image_path) if field == 'image_sha256' else
                     str(Path(s.image_path).resolve()) if field == 'image_path' else getattr(s, field, '')
                     for s in samples}
        if trained & (evaluated - {''}):
            raise ValueError(f'Training/evaluation overlap in {field}')


def training_inventory(samples, task):
    return [{'image_path': str(Path(s.image_path).resolve()), 'image_sha256': sha256_file(s.image_path),
             'label': getattr(s, task), 'media_name': s.media_name, 'url': s.url,
             'image': s.image, 'group_id': s.group_id} for s in samples]


def normalize_to_label(raw: str, allowed, mode="strict") -> str | None:
    if mode != "strict":
        raise ValueError(f"Unknown parser mode: {mode}")
    if not isinstance(raw, str):
        return None
    text = raw.strip().upper()
    if text in allowed:
        return text
    return None


def macro_prf(y_true, y_pred, labels):

    precs, recs, f1s = [], [], []
    for lab in labels:
        tp = sum(t == lab and p == lab for t, p in zip(y_true, y_pred))
        fp = sum(t != lab and p == lab for t, p in zip(y_true, y_pred))
        fn = sum(t == lab and p != lab for t, p in zip(y_true, y_pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        precs.append(prec); recs.append(rec); f1s.append(f1)
    n = len(labels)
    return sum(precs) / n, sum(recs) / n, sum(f1s) / n


def balanced_accuracy(y_true, y_pred, labels):

    recs = []
    for lab in labels:
        support = sum(t == lab for t in y_true)
        if support:
            recs.append(sum(t == lab and p == lab
                            for t, p in zip(y_true, y_pred)) / support)
    return sum(recs) / len(recs) if recs else 0.0


def ordinal_errors(y_true, y_pred, order):


    idx = {lab: i for i, lab in enumerate(order)}
    hi = len(order) - 1
    abs_err, sq_err = [], []
    for t, p in zip(y_true, y_pred):
        if t not in idx:
            continue
        ti = idx[t]
        if p in idx:
            d = abs(ti - idx[p])
        else:
            d = max(ti, hi - ti)
        abs_err.append(d)
        sq_err.append(d * d)
    if not abs_err:
        return None, None, None, 0
    n = len(abs_err)
    mae = sum(abs_err) / n
    mse = sum(sq_err) / n
    off_by_one = sum(d <= 1 for d in abs_err) / n
    return mae, mse, off_by_one, n


def print_confusion(y_true, y_pred, labels):
    counts = defaultdict(Counter)
    for t, p in zip(y_true, y_pred):
        counts[t][p or "(none)"] += 1
    cols = labels + ["(none)"]
    w = max(len(c) for c in cols + labels)
    print("    true\\pred  " + " ".join(f"{c[:6]:>6}" for c in cols))
    for t in labels:
        row = " ".join(f"{counts[t][p]:>6}" for p in cols)
        print(f"    {t:<{w}}  {row}")


def evaluate(task_key, y_true, y_pred, print_report=True):
    task = TASKS[task_key]
    labels = task["order"]
    if len(y_true) != len(y_pred) or set(y_true) - set(labels):
        raise ValueError('Targets must be valid and match the prediction count')
    n = len(y_true)
    if n == 0:
        return {"task": task_key, "n": 0, "macro_f1": None, "accuracy": None,
                "per_class": {}, "confusion_matrix": None}
    acc = sum(t == p for t, p in zip(y_true, y_pred)) / n
    majority = Counter(y_true).most_common(1)[0]
    maj_acc = majority[1] / n
    prec, rec, f1 = macro_prf(y_true, y_pred, labels)
    bal_acc = balanced_accuracy(y_true, y_pred, labels)
    mae, mse, obo, n_ord = ordinal_errors(y_true, y_pred, task["order"])

    if print_report:
        print(f"\n=== {task_key}  ({len(labels)}-class, n={n}) ===")
        print(f"  macro-F1          {f1:6.3f}   <- primary")
        print(f"  macro precision   {prec:6.3f}")
        print(f"  macro recall      {rec:6.3f}")
        print(f"  accuracy          {acc:6.3f}   (test-majority prevalence '{majority[0]}' = {maj_acc:.3f})")
        print(f"  balanced acc      {bal_acc:6.3f}   (mean per-class recall)")
        print(f"  MAE               {mae:6.3f}   (ordinal 0..{len(labels)-1})")
        print(f"  MSE               {mse:6.3f}")
        print(f"  off-by-one acc    {obo:6.3f}")
        print_confusion(y_true, y_pred, labels)
    per_class = {}
    for label in labels:
        support = sum(t == label for t in y_true)
        predicted = sum(p == label for p in y_pred)
        tp = sum(t == p == label for t, p in zip(y_true, y_pred))
        lp = tp / predicted if predicted else 0.0
        lr = tp / support if support else 0.0
        per_class[label] = {"support": support, "predicted": predicted, "precision": lp,
                            "recall": lr, "f1": 2 * lp * lr / (lp + lr) if lp + lr else 0.0}
    columns = labels + ["(invalid)"]
    matrix = [[sum(t == lab and (p == col if col != "(invalid)" else p not in labels)
                   for t, p in zip(y_true, y_pred)) for col in columns] for lab in labels]
    return {
        "task": task_key, "n": n,
        "macro_f1": f1, "macro_precision": prec, "macro_recall": rec,
        "accuracy": acc, "majority_acc": maj_acc, "balanced_accuracy": bal_acc,
        "mae": mae, "mse": mse, "off_by_one": obo,
        "per_class": per_class,
        "confusion_matrix": {"rows": labels, "columns": columns, "counts": matrix},
    }


class Predictor:


    def predict(self, sample: Sample, task_key: str) -> str | None:
        raise NotImplementedError


class MajorityPredictor(Predictor):


    def __init__(self, samples):
        if not samples:
            raise ValueError('Majority baseline requires training examples')
        self.const = {}
        for task, settings in TASKS.items():
            counts = Counter(getattr(sample, task) for sample in samples)
            self.const[task] = max(settings['order'], key=lambda label: counts[label])

    def predict(self, sample, task_key):
        return self.const[task_key]

    def run_metadata(self):
        return {"backend": "majority", "constants": self.const, "fit_method": "MEFI training majority"}


def audit_metrics(audit, print_report=False):
    results = {}
    for task in audit.identity["tasks"]:
        rows = [r for r in audit.records.values() if r["task"] == task]
        scored = [r for r in rows if r["status"] != "not_applicable"]
        result = evaluate(task, [r["true_label"] for r in scored],
                          [r["predicted_label"] for r in scored], print_report=print_report)
        result["status_counts"] = dict(Counter(r["status"] for r in rows))
        result["pending"] = len(audit.identity["samples"]) - len(rows)
        result["valid_output_rate"] = sum(r["status"] == "ok" for r in scored) / len(scored) if scored else None
        result["truncated_images"] = sum(r["details"].get("image", {}).get("truncated_px", 0) > 0 for r in scored)
        result["elapsed_prediction_seconds"] = sum(r["elapsed_seconds"] for r in scored)
        results[task] = result
    return results


def run(predictor: Predictor, samples, out_csv=None, tasks=("factuality", "bias"),
        *, resume=False, retry_errors=False, parser="strict", audit_context=None):
    from eval_audit import EvaluationAudit

    if not samples:
        raise ValueError("No samples to evaluate")
    if parser != "strict":
        raise ValueError(f"Unknown parser: {parser}")
    if resume and out_csv is None:
        raise ValueError("Resuming requires an explicit output CSV")
    if out_csv is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_csv = f"artifacts/evaluations/{stamp}-{uuid.uuid4().hex[:8]}/predictions.csv"
    sample_manifest = []
    for sample in samples:
        path = str(Path(sample.image_path).resolve())
        sample_manifest.append({
            "sample_id": json_hash([sample.media_name, sample.url, path]),
            "media_name": sample.media_name, "url": sample.url,
            "image_path": path, "image_sha256": sha256_file(path),
            "labels": {task: getattr(sample, task) for task in tasks},
        })
    if len({s["sample_id"] for s in sample_manifest}) != len(sample_manifest):
        raise ValueError("Duplicate evaluation sample IDs")
    identity = {
        "tasks": list(tasks), "samples": sample_manifest,
        "parser": {"mode": parser, "version": 1},
        "predictor": predictor.run_metadata() if hasattr(predictor, "run_metadata") else {"class": type(predictor).__name__},
        "context": audit_context or {}, "source": source_info(), "environment": environment_info(),
    }
    with EvaluationAudit(out_csv, identity, resume=resume) as audit:
        print(f"Evaluation journal: {audit.directory / 'predictions.jsonl'}")
        audit.save_metrics(audit_metrics(audit), "running")
        complete = False
        try:
            for sample, info in tqdm(list(zip(samples, sample_manifest)), desc="+".join(tasks), unit="img"):
                for task in tasks:
                    if not audit.pending(info["sample_id"], task, retry_errors):
                        continue
                    true_label = getattr(sample, task)
                    record = {
                        **{k: v for k, v in info.items() if k != "labels"}, "task": task,
                        "true_label": true_label, "raw_text": None, "predicted_label": None,
                        "correct": None, "status": "not_applicable", "error": None,
                        "elapsed_seconds": 0.0, "details": {},
                    }
                    if true_label in TASKS[task]["order"]:
                        start = time.perf_counter()
                        predictor.last_details = {}
                        try:
                            raw = predictor.predict(sample, task)
                            pred = normalize_to_label(raw, TASKS[task]["order"], mode=parser)
                            record.update(raw_text=raw, predicted_label=pred, correct=pred == true_label,
                                          status="ok" if pred is not None else "unparseable")
                        except Exception as exc:
                            record.update(status="error", correct=False, error={
                                "type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(),
                            })
                            tqdm.write(f"[error] {sample.media_name} {task}: {type(exc).__name__}: {exc}")
                        record["elapsed_seconds"] = time.perf_counter() - start
                        record["details"] = getattr(predictor, "last_details", {})
                    audit.append(record)
                    if len(audit.records) % 10 == 0:
                        audit.save_metrics(audit_metrics(audit), "running")
            complete = True
        finally:
            audit.export_csv()
            results = audit_metrics(audit, print_report=complete)
            has_failures = any(r["status"] in ("error", "unparseable") for r in audit.records.values())
            status = ("complete_with_failures" if has_failures else "complete") if complete else "interrupted"
            audit.save_metrics(results, status)
        print(f"Predictions: {out_csv}\nManifest and metrics: {audit.directory}")
        return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--image-root")
    ap.add_argument("--predictor", default="majority",
                    choices=["majority", "openrouter", "local"])
    ap.add_argument("--model",
                    help="openrouter slug (--predictor openrouter) or a "
                         "train_peft.py registry key / raw HF id (--predictor local)")
    ap.add_argument("--adapter-dir", help="path to a final_adapter/ dir from train_peft.py "
                                            "(required for fine-tuned --predictor local)")
    ap.add_argument("--base-only", action="store_true",
                    help="--predictor local: evaluate the untuned base model without an adapter")
    ap.add_argument("--processor-dir",
                    help="--predictor local: processor assets to use without attaching an adapter")
    ap.add_argument("--quant", choices=["4bit", "16bit"],
                    help="--predictor local: quantization of the base model (match training)")
    ap.add_argument("--tile-height", type=int)
    ap.add_argument("--max-tiles", type=int)
    ap.add_argument("--max-width", type=int)
    ap.add_argument("--max-new-tokens", type=int)
    ap.add_argument("--thinking", choices=THINKING_MODES,
                    help="inherit the adapter's template settings")
    ap.add_argument("--revision", help="base-model HF commit/tag; inherits saved revision when available")
    ap.add_argument("--parser", choices=["strict"], default="strict")
    ap.add_argument("--task", default="both",
                    choices=["factuality", "bias", "both"],
                    help="run a single task or both")
    ap.add_argument("--limit", type=int, help="only run the first N samples (smoke test)")
    ap.add_argument("--base-url", default="https://openrouter.ai/api/v1",
                    help="OpenAI-compatible endpoint; point at a local server "
                         "(e.g. http://localhost:11434/v1) for local models")
    ap.add_argument("--api-key", help="API key; omit for local servers")
    ap.add_argument("--out")
    ap.add_argument("--resume", action="store_true", help="resume --out using its matching audit journal")
    ap.add_argument("--retry-errors", action="store_true", help="with --resume, retry failed inference calls")
    ap.add_argument("--report-only", action="store_true", help="rebuild CSV/metrics from --out's journal without loading a model")
    args = ap.parse_args()

    if not args.report_only and (not args.csv or not args.image_root):
        ap.error('--csv and --image-root are required for inference')
    if (args.resume or args.report_only) and not args.out:
        ap.error("--resume/--report-only requires --out")
    if args.retry_errors and not args.resume:
        ap.error("--retry-errors requires --resume")
    if args.limit is not None and args.limit <= 0:
        ap.error("--limit must be positive")
    if args.max_new_tokens is not None and args.max_new_tokens <= 0:
        ap.error("--max-new-tokens must be positive")
    if args.report_only:
        from eval_audit import EvaluationAudit
        directory = Path(args.out).with_suffix(".audit")
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        with EvaluationAudit(args.out, manifest["identity"], resume=True) as audit:
            audit.export_csv()
            metrics = audit_metrics(audit, print_report=True)
            expected = len(audit.identity["samples"]) * len(audit.identity["tasks"])
            failures = any(r["status"] in ("error", "unparseable") for r in audit.records.values())
            status = "interrupted" if len(audit.records) < expected else ("complete_with_failures" if failures else "complete")
            audit.save_metrics(metrics, status)
        return

    tasks = ("factuality", "bias") if args.task == "both" else (args.task,)
    if args.base_only and args.predictor != "local":
        ap.error("--base-only requires --predictor local")
    if args.base_only and args.adapter_dir:
        ap.error("--base-only and --adapter-dir are mutually exclusive")
    if args.processor_dir and args.predictor != "local":
        ap.error("--processor-dir requires --predictor local")
    saved = load_adapter_run_config(args.adapter_dir) if args.predictor == "local" and args.adapter_dir else None
    settings = saved["settings"] if saved else {}
    image_settings = resolve_image_settings(vars(args), settings.get("preprocessing"))
    from qwen_predictor import PROMPTS
    prompts = settings.get("prompts", PROMPTS)
    thinking = args.thinking or settings.get("thinking", "template-default" if args.predictor == "local" else "off")
    quant = args.quant or settings.get("quant", "4bit")
    revision = args.revision or (saved.get("model_revision") if saved else None)


    found, missing = load_samples(args.csv, args.image_root, tasks=tasks, expected_split='test')
    if missing:
        ap.error(f"{len(missing)} expected screenshots are missing; fix inputs before evaluation")

    print(f"Loaded {len(found)+len(missing)} rows: "
          f"{len(found)} with images, {len(missing)} missing")
    if args.limit:
        found = found[:args.limit]
        print(f"--limit: evaluating first {len(found)} samples only")

    if args.predictor == "majority":
        training, _ = load_samples(Path(args.csv).parent / 'train.csv', args.image_root, expected_split='train')
        check_training_overlap(training_inventory(training, 'factuality'), found)
        predictor = MajorityPredictor(training)
    elif args.predictor == "local":
        if not args.base_only and not args.adapter_dir:
            ap.error("--adapter-dir is required unless --base-only is selected")
        if not args.base_only and args.task == "both":
            ap.error("A local adapter is task-specific; select --task factuality or --task bias")
        if not args.base_only and settings.get("task", args.task) != args.task:
            ap.error(f"Adapter was trained for {settings['task']}, not {args.task}")
        from train_peft import resolve_model_id
        from qwen_predictor import TransformersLoraPredictor
        model_arg = args.model or settings.get("model_id")
        if not model_arg:
            ap.error("--model is required for adapters without run_config.json")
        model_id, skip_vision_modules = resolve_model_id(model_arg)
        if settings.get("model_id", model_id) != model_id:
            ap.error("Requested base model does not match the adapter's saved configuration")
        if settings:
            skip_vision_modules = settings.get("skip_vision_modules", skip_vision_modules)
        elif not args.base_only:
            raise ValueError("Adapter directory must include training prompt and image settings in run_config.json")
        if saved:
            if not saved.get('training_samples'):
                raise ValueError('Adapter has no training-image inventory')
            check_training_overlap(saved['training_samples'], found)
        predictor = TransformersLoraPredictor(
            model_id, None if args.base_only else args.adapter_dir,
            processor_dir=args.processor_dir,
            quant=quant, skip_vision_modules=skip_vision_modules,
            **image_settings, max_new_tokens=args.max_new_tokens or 16,
            thinking=thinking, prompts=prompts, revision=revision,
        )
    else:
        from qwen_predictor import OpenRouterQwenPredictor
        if not args.model:
            ap.error("--model is required for OpenRouter inference")
        predictor = OpenRouterQwenPredictor(
            model=args.model, base_url=args.base_url, api_key=args.api_key,
            **image_settings, disable_reasoning=thinking == "off",
            max_new_tokens=args.max_new_tokens or 512, prompts=prompts)
    print(f"predictor={args.predictor} tasks={tasks} preprocessing={image_settings} thinking={thinking}")
    context = {"csv_path": str(Path(args.csv).resolve()), "csv_sha256": sha256_file(args.csv),
               "adapter_training_config": saved,
               "requested_overrides": {k: getattr(args, k) for k in
                                       (*IMAGE_DEFAULTS, "thinking", "quant", "revision", "max_new_tokens")}}
    run(predictor, found, out_csv=args.out, tasks=tasks, resume=args.resume,
        retry_errors=args.retry_errors, parser=args.parser, audit_context=context)


if __name__ == "__main__":
    main()
