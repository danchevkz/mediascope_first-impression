# Data and Protocol

## Manifests

Run `python -m mediascope.cli prepare` after placing the dataset as shown in the
README. It joins each split's CSV with its `metadata.csv` by exact outlet name
and URL, obtains filenames from metadata, and checks that labels and cluster
membership agree. It rejects missing images, duplicate sources, and cross-split
overlaps in source identity, group membership, or file hashes. It never reassigns
sources between train and test.

Use the resulting `work/dataset/train.csv` and `work/dataset/test.csv`:

| Column | Meaning |
|---|---|
| `image_path` | Resolved screenshot path; preparation writes absolute local paths |
| `image` | Optional stable ID; defaults to image_path |
| `true_label_factuality` | LOW, MIXED, or HIGH |
| `true_label_bias` | LEFT, LEFT-CENTER, LEAST BIASED, RIGHT-CENTER, or RIGHT |
| `media_name`, `url` | Required by the VLM experiment tools, never sent as model inputs |
| `group_id` | Supplied group ID or a unique deterministic singleton ID |

Use the released MEFI train/test assignments. Do not create a random split for a
paper comparison. Normalize raw factuality labels by mapping VERY LOW to LOW and
MOSTLY FACTUAL/VERY HIGH to HIGH. Map EXTREME LEFT/RIGHT to LEFT/RIGHT and exclude
outlets without a bias rating from the joint benchmark. Classifier manifests
require the final labels, not raw aliases.

The dataset contains 1,692 training and 423 test outlets for both tasks.
Factuality counts are 256/959/477 in train and 64/243/116 in test, in LOW/MIXED/HIGH
order. Bias counts are 108/210/183/776/415 in train and 26/52/45/199/101 in test,
in LEFT/LEFT-CENTER/LEAST BIASED/RIGHT-CENTER/RIGHT order.
Preparation records source CSV hashes, screenshot hashes, and split-membership
signatures in its outputs. These counts describe the supplied split, not a rule
for moving or dropping rows.
Regenerate the prepared manifests after moving the dataset to another location.
All classifier, local-VLM, API-VLM, and naive-baseline loaders validate both task
labels. Invalid labels or missing images stop the run instead of removing rows.
Prepared split signatures detect changed membership or labels; screenshot hashes
detect changed input files.

## Image Processing

VLM inference and QLoRA use vertical tiles covering the full page, with up to six
tiles, tile height 1,400, and maximum width 1,024. The VLM input preparation rejects
truncation instead of silently discarding the lower page.

The classical classifiers, GPT OCR extraction, and template clustering receive
the complete supplied screenshot without an additional crop or resize. OCR runs
on the original-size RGB array; image encoders use their checkpoint processors.
Those processors can resize or crop internally according to the checkpoint.
MinHash pools the full image into its fixed grids. Mask generation also preserves
the full screenshot dimensions.

Feature extraction records `full_page_original_size` in its protocol. Training
and inference reject features with a different input policy. Extract train and
test features in fresh directories and fit the classifiers using those features.

## Evaluation

Classical inference rejects train/test overlaps in IDs, file paths, outlet names,
URLs, provided group IDs, and exact file hashes. PCA, SVD, TF-IDF, and scalers are
part of the fitted artifact. Pair evaluation uses the same artifact for both
conditions. Local VLM evaluation also uses the same task-specific adapter across
main, masked, and unmasked conditions and checks its training configuration.
The local VLM wrapper also matches the adapter's training-image inventory and
labels to `work/dataset/train.csv`. It rejects an adapter trained on a different
split, even if its hyperparameters match. Training never substitutes a failed
image with a randomly selected example.
No model receives the target label, source metadata, or the other
condition's prediction.

Ordinal scales are LOW=0, MIXED=1, HIGH=2 and LEFT=0 through RIGHT=4. Macro metrics
average across the full fixed label set, with zero division assigned zero.
Classical and OCR-GPT exports require complete, valid predictions. The VLM audit
also reports invalid outputs and request failures; they count as incorrect, with
maximum possible ordinal error for the target class. Do not compare valid-only
VLM scores with complete-set scores.

## Reproduction Boundary

Dataset manifests, screenshots, manual cluster decisions, human masking reviews,
and trained adapter weights remain external experiment inputs. Matching code and
hyperparameters alone does not establish that two reported scores use identical
examples or checkpoints. Preserve the generated manifests, input hashes, request
journals, dependency versions, and model revisions with each execution.
