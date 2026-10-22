from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parents[2] / '.env', override=False, encoding='utf-8-sig')

IMAGE_DEFAULTS = {"tile_height": 1400, "max_tiles": 6, "max_width": 1024}
THINKING_MODES = ("off", "template-default")
LORA_EXCLUDED_MODULES = r"(?:.*\.)?(?:visual|vision_model|vision_tower|multi_modal_projector|lm_head)(?:\..*)?"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path, value):

    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def environment_info():
    versions = {"python": platform.python_version()}
    for package in ("torch", "torchvision", "transformers", "peft", "accelerate",
                    "bitsandbytes", "pillow", "requests", "huggingface-hub", "filelock"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


def source_info():
    scripts = Path(__file__).resolve().parent
    names = ("run_config.py", "eval_audit.py", "mediascope_eval.py", "qwen_predictor.py", "train_peft.py")
    hashes = {name: sha256_file(scripts / name) for name in names if (scripts / name).exists()}
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=scripts.parent, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"git_commit": commit, "file_sha256": hashes}


def render_generation_prompt(processor, messages, thinking="off"):
    if thinking not in THINKING_MODES:
        raise ValueError(f"Unknown thinking mode: {thinking}")
    template = getattr(processor, "chat_template", None)


    kwargs = {"enable_thinking": False} if thinking == "off" and "enable_thinking" in str(template) else {}
    return processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False, **kwargs
    )


def load_adapter_run_config(adapter_dir):

    if not adapter_dir:
        return None
    root = Path(adapter_dir)
    for path in (root / "run_config.json", root.parent / "run_config.json"):
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def resolve_image_settings(overrides, saved=None):
    settings = {**IMAGE_DEFAULTS, **(saved or {})}
    for key in IMAGE_DEFAULTS:
        if overrides.get(key) is not None:
            settings[key] = overrides[key]
        if not isinstance(settings[key], int) or settings[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    return settings


def adapter_fingerprint(adapter_dir):
    root = Path(adapter_dir)


    return {p.name: sha256_file(p) for p in sorted(root.iterdir())
            if p.is_file() and p.suffix in (".safetensors", ".json", ".jinja", ".model")}
