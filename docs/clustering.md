# Template Clustering

```bash
python -m pip install -r requirements/encoders.txt -r requirements/ocr.txt
```

All four methods use the full supplied screenshot without a 16:9 crop. Inputs are
PNG/JPEG/WebP/TIFF images. When the input root contains the supplied train/test
metadata, extraction scans only those two screenshot folders, excluding plots.
Filenames must be unique.

```bash
python clustering/ocr_visual_minhash.py --input-dir ../data/mediascope-clustered --output-dir work/clusters/minhash-dbscan
python clustering/minhash_hdbscan.py --records work/clusters/minhash-dbscan/ocr_minhash_records.jsonl --distances work/clusters/minhash-dbscan/combined_distances.npy --output-dir work/clusters/minhash-hdbscan
python clustering/visual_hdbscan.py --input-dir ../data/mediascope-clustered --output-dir work/clusters/visual
python clustering/geometry_hdbscan.py --records work/clusters/minhash-dbscan/ocr_minhash_records.jsonl --output-dir work/clusters/geometry
```

The MinHash representation combines lowercase OCR unigrams of at least four
characters, bigrams and trigrams with position-aware visual tokens. Visual tokens
encode quantized RGB and edge strength on a 32x18 grid, plus 2x2 grayscale patches
on a 16x9 grid. Both signatures have 128 components and permutation seed 20260921.
Similarity is 0.55 text + 0.45 visual, or visual only if either text set is empty.
DBSCAN uses distance 1-similarity, eps=0.42, min_samples=3.

The visual method L2-normalizes DINOv2-Small CLS features and applies 50-component
PCA. Geometry uses a 290-dimensional OCR-box representation, standardization,
and 30-component PCA. It filters boxes below confidence 0.35 and includes
normalized geometry, confidence statistics, 8x12 occupancy/coverage maps, and
projection profiles. PCA uses seed 42. Both reduced spaces use Euclidean distance.
All HDBSCAN variants use min_cluster_size=10, min_samples=5, and EOM selection.

Each method exports `clusters.csv` and `summary.json`; noise has cluster -1.
Cluster IDs have meaning only within their method. Review candidate groups before
using them for splitting. The dense MinHash matrix uses O(N^2) storage.
OCR is sequential by default. `--device gpu` explicitly opts into GPU OCR;
the DINOv2 extraction script uses CPU. No command controls unrelated processes.

## Dataset Utilities

`data_tools/` contains code for dataset statistics and template-aware split
construction, not dataset files. These utilities are optional: classification
uses the train/test split downloaded from Hugging Face without regenerating it.

### Statistics

Use the manifests produced by `python -m mediascope.cli prepare`:

```bash
python data_tools/statistics.py --split-dir work/dataset --images-dir ../data/mediascope-clustered --out-dir work/statistics
```

### Split Construction

For the template-based split construction experiment, use the separate inputs
below. Its output is not substituted for the supplied classification split.

Use a separate Python environment with `requirements/split.txt`.

```bash
python data_tools/make_split.py --data-dir /data/split-inputs --out-dir work/split --seed 42
python data_tools/statistics.py --split-dir work/split --images-dir /data/screenshots --out-dir work/statistics
```

Required split inputs:

- `mbfc_labels.csv`: `image`, `media_name`, `url`, `timestamp`, `country`,
  `factuality`, `trustworthiness`, `bias_rating`, and `MBFC_source`.
- `curated_clusters.csv`: `method`, `folder`, `image`, and boolean
  `used_for_split`. Use method names `ocr_visual_minhash_dbscan`,
  `ocr_visual_minhash_hdbscan`, `dinov2_pca_hdbscan`, `geometry_pca_hdbscan`.
- `dinov2_vectors.npz`: `keys` and `vectors`; keys follow `screenshot_key` in
  `data_tools/make_split.py`. Vectors must have nonzero norm.

The split links reviewed clusters, identical registered domains, matching screenshot
identities, and DINOv2 cosine similarities >=0.95. Whole connected components stay
on one side. The search targets a 20% test partition while matching factuality and
bias distributions. It removes UNRATED outlets after assignment without moving
the remaining outlets. Manual cluster curation is an input, not an automatic
claim of template identity. Keep the released split for reproducing paper scores.
