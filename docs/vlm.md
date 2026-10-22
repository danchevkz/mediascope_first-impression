# VLM Experiments

Install `vlm/requirements-inference.txt` in a dedicated environment. For local
VLMs, install a compatible CUDA PyTorch/torchvision pair and then
`vlm/requirements-training.txt`. The supplied run configurations record the exact
model revisions and software versions. Training requires substantially more
memory than the classical classifiers; the supplied VLM runs used 96 GB GPUs.

## Frontier Classification

Prepare manifests once with `python -m mediascope.cli prepare` from the repository
root. The commands below use those manifests and the supplied screenshot folders.

| Selector | Model | Credential |
|---|---|---|
| `gpt56` | GPT-5.6 Sol | OPENAI_API_KEY |
| `opus5` | Claude Opus 5 | ANTHROPIC_API_KEY |
| `kimi` | Kimi K3 | OPENROUTER_API_KEY |
| `qwen` | Qwen3.8 Max Prime | OPENROUTER_API_KEY |

```bash
python vlm/run.py prepare --dataset main --csv work/dataset/test.csv --images ../data/mediascope-clustered --output work/api-inputs
python vlm/run.py api --model opus5 --dataset main --inputs work/api-inputs --output work/vlm-opus5 --dry-run
```

Remove `--dry-run` to send paid requests. Each task has its own prompt. The
runner records inputs, responses, usage, failures, and metrics. It uses native
OpenAI/Anthropic endpoints and OpenRouter for the two other frontier models.
`--resume` continues a matching run. Use `--dataset masked` or `unmasked` with
the corresponding manifest and prepared images for pair evaluation.

## QLoRA and Local Inference

```bash
python vlm/run.py train --model qwen35 --task factuality --dataset main --csv work/dataset/train.csv --images ../data/mediascope-clustered --output work/qlora/qwen35-factuality --dry-run
python vlm/run.py local --model qwen35 --variant finetuned --task factuality --dataset main --csv work/dataset/test.csv --images ../data/mediascope-clustered --adapters /models/adapters --output work/local/qwen35-factuality.csv --dry-run
```

Choose `qwen35` or `ministral`, and `factuality` or `bias`. For base evaluation,
set `--variant base`. Base inference uses the matching processor assets without
loading adapter weights. A newly trained `final_adapter` directory must be placed
under the corresponding key in the adapter root, as listed in
`vlm/configs/local_evaluation.json`. The root contains `qwen35_factuality`,
`qwen35_bias`, `ministral_factuality`, and `ministral_bias`. Each task-specific
adapter is shared across main, masked, and unmasked evaluation; do not train an
adapter on the masked or unmasked test subset.

Training uses NF4, double quantization, BF16 compute, rank 16, alpha 32, dropout
0.05, three epochs, batch size one, accumulation 16, and seed 42. AdamW uses a
learning rate of 2e-4, a linear schedule, and 16 warmup steps. Generation limits
come from the task/condition configurations.
LoRA targets language-model linear layers only; vision modules, the multimodal
projector, and the output head are explicitly excluded, with no trainable bias.

Before local inference, the wrapper checks the adapter's `run_config.json`
against the task's training configuration, including its learning rate, training
budget, preprocessing, prompt, and base-model revision. An incompatible adapter
requires training with the specified configuration, not editing its metadata.
It also checks every training image, label, source, and group against
`work/dataset/train.csv`. Use `--train-csv` and `--train-images` only when this
same prepared dataset is stored elsewhere. Evaluation rejects train/test overlaps.
Evaluation records the actual adapter fingerprint; the configuration does not
prescribe a weight hash before training.

## Naive Baselines

```bash
python vlm/scripts/run_naive_baselines.py --dataset-root work/dataset --seed 42 --random-repetitions 10000 --out work/naive.json
```

This reads `train.csv` and `test.csv`, fits the majority baseline from train
labels, uses the middle ordinal class, and averages uniform-random predictions
over 10,000 repetitions.

## Outlet-Rating Recall

The recall probe elicits structured source-recognition reports and provider
reasoning summaries, not verified evidence of training-set membership.
Use the same reviewed subset for every model.

```bash
python vlm/scripts/make_interpretability_subset.py --dataset-root work/dataset --output-dir work/recall-subset
python vlm/run.py prepare-api-inputs --inputs work/api-inputs --output work/recall-inputs
python vlm/scripts/eval_opus_summary_parallel.py --dataset-root work/api-inputs --subset-csv work/recall-subset/subset.csv --input-run-dir work/recall-inputs --output-dir work/recall-opus5 --workers 1
python vlm/scripts/eval_gpt56_summary_parallel.py --dataset-root work/api-inputs --subset-csv work/recall-subset/subset.csv --input-run-dir work/recall-inputs --output-dir work/recall-gpt56 --workers 1 --stream --record-missing-summaries
python vlm/scripts/eval_openrouter_summary_parallel.py --model kimi --reference-plan work/recall-opus5/plan.json --subset-csv work/recall-subset/subset.csv --output-dir work/recall-kimi --workers 1 --record-missing-summaries --record-provider-refusals --record-output-limits --record-invalid-reports
```

Select `qwen` for the other OpenRouter model. Subset construction requires
`clusters_13.csv` alongside `test.csv`, with `media_name,url,cluster,split`.
It samples 50% proportionally across template-group/factuality/bias strata,
using seed 42. Reviews hide ground-truth labels and correctness.

```bash
python vlm/scripts/review_recognition_probe_v2.py freeze --run work/recall-opus5 --subset-csv work/recall-subset/subset.csv --root work/review-opus5
python vlm/scripts/review_recognition_probe_v2.py read --root work/review-opus5 --task factuality
python vlm/scripts/review_recognition_probe_v2.py aggregate --root work/review-opus5
```

Before aggregation, provide adjudications under `annotations/` following
`vlm/scripts/recognition_probe_judge_v2.md`. Each JSON batch contains `task`,
`reviewed_ranges` (inclusive one-based pairs), and `decisions` keyed by the
one-based report number. Decisions include `category`, `rating_scope`,
`family_link`, exact `evidence` quotes, and `note`. The aggregator rejects
overlapping reviews, missing reviews, unsupported evidence, and invalid categories.
Judge outputs are external annotations, not inferred from correctness or filled
from missing reports. This package includes the rubric and aggregation code,
not an implementation that silently substitutes a different judge for GPT-6 Astra.
