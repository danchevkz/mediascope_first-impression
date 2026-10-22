# MeFI: Media Profiling from First Impressions

Code for **A Picture Is Worth a Hundred Articles: Media Profiling from First
Impressions**.

## Introduction

Readers form impressions of news outlets before reading their articles. We study
whether models can predict an outlet's factuality and political bias from a
single full-page screenshot of its landing page. The input contains the headlines,
images, advertisements, and layout that readers encounter on arrival, without
article collections, external source metadata, browsing, or retrieved evidence.
The predictions estimate source-level ratings, not the truth of individual claims.

**MeFI (Media profiling from First Impressions)** pairs landing-page screenshots
from **2,115 news outlets** with ratings from Media Bias/Fact Check (MBFC). The
benchmark has three factuality classes (LOW, MIXED, HIGH) and five political bias
classes (LEFT, LEFT-CENTER, LEAST BIASED, RIGHT-CENTER, RIGHT). Its fixed,
template-aware split contains **1,692 training** and **423 test** outlets.

We evaluate zero-shot vision-language models, QLoRA-tuned compact models, and
supervised classifiers using OCR text and visual embeddings. The repository also
includes template clustering, outlet-rating recall probes, and paired
masked/unmasked evaluation to examine whether predictions depend on explicit
outlet identifiers.

## 1. Clone the Repository

Clone this repository and run all commands from its root with Python 3.12.

```bash
python -m pip install -r requirements.txt
```

## 2. Set API Keys

Create `.env` from `env-example.txt` and fill in `HF_TOKEN` and
`OPENROUTER_API_KEY`. Set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` only for
experiments that use those providers directly.

## 3. Download the Dataset

```bash
dotenv run --no-override -- hf download Media-scope/mediascope-clustered --repo-type dataset --revision ea1f7ec37e90411502385ebb7fc5a1106f58a1c2 --local-dir ../data/mediascope-clustered
```

This uses the supplied train/test split without regenerating it.

## 4. Run Experiments

Prepare the input tables once:

```bash
python -m mediascope.cli prepare
```

Install the additional requirements and run the commands for the selected experiment:

- [OCR and image classifiers; GPT with OCR](docs/classification.md)
- [VLM inference, fine-tuning, and rating recall](docs/vlm.md)
- [Template clustering](docs/clustering.md)
- [Masking and paired evaluation](docs/masking.md)
