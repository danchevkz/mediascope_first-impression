# Classification

```bash
python -m pip install -r requirements/encoders.txt -r requirements/ocr.txt
```

The four published classifiers have stable selectors:

| Selector | System | Configuration |
|---|---|---|
| `ocr_svm` | SVM w/ OCR-TFIDF | Word 1-2 grams, min_df=2, sublinear TF, 15,000 terms, balanced linear SVM C=0.7 |
| `image_svm` | SVM w/ image embeddings | SigLIP, standardization, 24 whitened PCA components, standardization, balanced linear SVM C=0.2 |
| `image_ridge` | Ridge w/ image embeddings | CLIP + DINOv2 + SigLIP, separate standardization and 24 whitened PCA components per encoder, concatenation, standardization, balanced Ridge alpha=3 |
| `fusion_svm` | SVM w/ OCR & image embeddings | 72 image components + 48 standardized OCR SVD components + four OCR statistics, standardization, balanced linear SVM C=0.2 |

OCR statistics are word count, unique-word ratio, mean recognition confidence,
and token count. Word count and token count use the same tokenization and are
identical. The implementation preserves both columns. All learned transformations
fit training rows only. Feature metadata and outlet names are not classifier inputs.

## Extract

Prepare the supplied dataset once with `python -m mediascope.cli prepare`.

```bash
python -m mediascope.cli extract --manifest work/dataset/train.csv --images ../data/mediascope-clustered --output work/features-train
python -m mediascope.cli extract --manifest work/dataset/test.csv --images ../data/mediascope-clustered --output work/features-test
```

Use `--encoders` with no values for OCR only. Use `--encoders siglip --no-ocr` for
the visual SVM alone. Use matching extraction settings for train and test.
Extraction checks image hashes before reusing an output directory.

`--device` selects the image-encoder device. `--ocr-device` can select a different
OCR device, for example `--device cpu --ocr-device gpu:0`. GPU OCR requires the
GPU Paddle runtime instead of the CPU runtime. Use the same device settings for
training and test extraction.

For GPU OCR with CPU image encoders, use a fresh environment and install:

```bash
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements/encoders.txt paddleocr==3.7.0
python -m pip install paddlepaddle-gpu==3.2.1 --index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/
```

The encoder IDs are `openai/clip-vit-base-patch32`,
`google/siglip-base-patch16-224`, and `facebook/dinov2-small`.
CLIP returns projected image features, SigLIP returns pooled features, and DINOv2
returns an L2-normalized CLS embedding. OCR uses the PP-OCRv6 medium detector and
recognizer, with orientation correction and unwarping disabled. Accepted text has
confidence >=0.35. The numeric confidence statistic includes all detected boxes.

## Train and Evaluate

```bash
python -m mediascope.cli train --train work/dataset/train.csv --images ../data/mediascope-clustered --features work/features-train --output work/classifiers
python -m mediascope.cli predict --test work/dataset/test.csv --images ../data/mediascope-clustered --features work/features-test --model work/classifiers/factuality/ocr_svm/model.joblib --output work/test/factuality/ocr_svm
```

Training produces one artifact for each method and task. Repeat `predict` for
each artifact. Prediction never fits a transformation or classifier. Only load
trusted joblib artifacts: loading an untrusted pickle can execute code.

Each evaluation writes `predictions.csv` and `metrics.json`. P/R/F1 and accuracy
are fractions; MAE and MSE use the raw ordinal distances, not percentages.

## Hyperparameter Search

Add `--search --folds 5 --jobs 1` to training. Search uses macro-F1 and the grids
in Appendix B.3. Every fold fits its own TF-IDF, PCA, SVD, scalers, and classifier.
If all training rows include `group_id`, validation uses StratifiedGroupKFold;
otherwise it uses StratifiedKFold. The test manifest never enters search.
The selected estimator then refits on all supplied training rows.

Search emits every candidate's mean and standard deviation of validation macro-F1,
the selected parameters, and the fitted artifact. The published configuration is
the default without `--search`; running a new search does not overwrite it.

## GPT-5.6 Sol with OCR

Set `OPENROUTER_API_KEY` in the environment, then run:

```bash
python -m mediascope.ocr_gpt --test work/dataset/test.csv --images ../data/mediascope-clustered --features work/features-test --output work/gpt-ocr
```

This sends text only, one request per screenshot returning both task labels. This
is the request format associated with the OCR row in the supplied manuscript.
The exact prompts live in `mediascope/ocr_gpt.py`. Requests use a 256-token output
cap, strict JSON, and reasoning effort `none`. The model is fixed to GPT-5.6 Sol;
there is no model substitution. Execution is sequential. Responses and usage
metadata persist before parsing, and a resumed run does not repeat completed calls.
