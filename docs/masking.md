# Identity Masking and Paired Evaluation

```bash
python -m pip install -r requirements/encoders.txt -r requirements/ocr.txt
```

## Create Candidate Pairs

Provide a manifest selected exclusively from `work/dataset/test.csv`. Set
`OPENROUTER_API_KEY`. The masking model is GPT-6 Sol:

```bash
python -m mediascope.masking --manifest /data/masking-candidates.csv --images ../data/mediascope-clustered --model openai/gpt-6-sol --output work/masking
```

The pipeline keeps full image dimensions. PaddleOCR locates text on overlapping
vertical tiles. A VLM receives the screenshot, an OCR-box overlay, and OCR text,
then selects publisher names, logos, domains, slogans, and branded handles.
Opaque masks cover those boxes. The prompt preserves headlines, article images,
ordinary navigation, and advertisements. OCR for mask localization uses
PP-OCRv5-server; classifier OCR uses PP-OCRv6-medium. These are distinct operations.
Both generation and pair verification validate candidates against
`work/dataset/test.csv`; use `--test` to specify its location if needed.
Generation checks image hashes and labels. Verification also compares each
unmasked image pixel-for-pixel with the corresponding original test screenshot.

The output contains `pairs.csv`, `masked/`, `unmasked/`, box coordinates, OCR
records, and model responses. Inspect all candidates: automatic masks are not
equivalent to the human-reviewed paper subset. Responses must identify the
requested model; a model mismatch or incomplete response stops the run.

## Human Review and Pixel Verification

Two reviewers independently mark each candidate `accept` or `reject` in a CSV
with columns `image,reviewer_1,reviewer_2`. Rejecting by either reviewer excludes
the pair. Do not treat the model's review flag as a human approval.

```bash
python -m mediascope.paired verify --pairs work/masking/pairs.csv --images work/masking --reviews /data/reviews.csv --output work/verified
```

This validates image dimensions, mask coordinates, exact pixel equality outside
the masks, mask color, labels, and review completeness. It exports separate
masked/unmasked manifests, which reference the same source IDs.

## Frozen Classifiers

Extract each condition separately, using exactly the training settings:

```bash
python -m mediascope.cli extract --manifest work/verified/masked.csv --images work/masking --output work/features-masked
python -m mediascope.cli extract --manifest work/verified/unmasked.csv --images work/masking --output work/features-unmasked
python -m mediascope.cli predict --test work/verified/masked.csv --images work/masking --features work/features-masked --model work/classifiers/factuality/ocr_svm/model.joblib --output work/eval-masked
python -m mediascope.cli predict --test work/verified/unmasked.csv --images work/masking --features work/features-unmasked --model work/classifiers/factuality/ocr_svm/model.joblib --output work/eval-unmasked
python -m mediascope.paired compare --masked work/eval-masked/predictions.csv --unmasked work/eval-unmasked/predictions.csv --output work/paired-comparison.json
```

Repeat for each trained artifact and task. Use the OCR-GPT command on both
feature directories, or the VLM commands with the two condition manifests.
No training or hyperparameter selection takes place on either condition.
The comparison requires identical image IDs and targets; it reports masked minus
unmasked changes for accuracy, macro-P/R/F1, MAE, and MSE.
