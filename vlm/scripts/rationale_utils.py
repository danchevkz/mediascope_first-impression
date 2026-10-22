import csv
import re
from mediascope_eval import TASKS
from qwen_predictor import PROMPTS

def rationale_prompt(task: str) -> str:
    base = PROMPTS[task].rsplit("\n\nOutput ONLY", 1)[0]
    labels = ", ".join(TASKS[task]["order"])
    return (
        f"{base}\n\nBriefly explain the basis for your judgment in 1-3 sentences. "
        f"Finish with a separate final line in the form `LABEL: <label>`, where "
        f"<label> is one of: {labels}."
    )


def parse_label(text: str, labels: list[str]) -> str | None:
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    if stripped.upper() in labels:
        return stripped.upper()
    alternatives = "|".join(re.escape(label) for label in sorted(labels, key=len, reverse=True))
    match = re.search(
        rf"(?im)^\s*(?:\*\*)?LABEL\s*:\s*({alternatives})\s*[.`]*(?:\*\*)?\s*$",
        stripped,
    )
    return match.group(1).upper() if match else None


def selected_keys(path: str, task: str) -> set[tuple[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        fields = set(rows.fieldnames or ())
        selected = set()
        for row in rows:
            if {"model", "task", "status", "readable"}.issubset(fields):
                include = (row["model"] == "opus" and row["task"] == task
                           and row["status"] == "ok" and row["readable"] == "True")
            elif "task" in fields:
                include = row["task"] == task
            else:
                include = True
            if include:
                selected.add((row["media_name"], row["url"]))
        return selected
