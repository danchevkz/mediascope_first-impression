import argparse
import struct
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
TASKS = {
    "Factuality": ("true_label_factuality", ["LOW", "MIXED", "HIGH"]),
    "Political bias": ("true_label_bias", ["LEFT", "LEFT-CENTER", "LEAST BIASED", "RIGHT-CENTER", "RIGHT"]),
}


def png_size(path):

    with open(path, "rb") as f:
        return struct.unpack(">II", f.read(24)[16:24])


def find_image(images_dir, image, split):
    for path in (images_dir / image, images_dir / split / image):
        if path.exists():
            return path
    raise FileNotFoundError(f"{image} is not in {images_dir} or {images_dir / split}")


def main():
    parser = argparse.ArgumentParser(description="Dataset numbers for the paper (Table 1, section 3.2).")
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=HERE / "output")
    args = parser.parse_args()
    split_dir, out_dir = args.split_dir.resolve(), args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(split_dir / "train.csv", keep_default_na=False)
    test = pd.read_csv(split_dir / "test.csv", keep_default_na=False)
    pages = pd.read_csv(split_dir / "split_assignments.csv", keep_default_na=False)
    both = pd.concat([train, test])
    if args.images_dir:
        images_dir = args.images_dir.resolve()
        sizes = pd.DataFrame([(image, *png_size(find_image(images_dir, image, split)))
                              for image, split in zip(pages["image"], pages["split"])],
                             columns=["image", "width", "height"])
    else:
        sizes = pd.read_csv(HERE / "data" / "page_sizes.csv")

    n = len(both)
    dates = pd.to_datetime(both["timestamp"])
    widths = sizes["width"].value_counts()
    common_width = widths.index[0]
    tallest = sizes["height"].max()
    template = pages[pages["group_id"].str.startswith("cluster_")]
    linked = pages[pages["group_id"].str.startswith("D")]
    single = pages[pages["group_id"] == ""]
    test_groups = template.loc[template["split"] == "test", "group_id"].unique()

    lines = [
        "# MEFI in numbers", "",
        f"- Pages: {n:,} ({len(train):,} train, {len(test):,} test, {100 * len(test) / n:.1f}% test)",
        f"- Captured: {dates.min().day} {dates.min():%B %Y} to {dates.max().day} {dates.max():%B %Y}",
        f"- Width: {widths.iloc[0]:,} pages are {common_width:,} px wide; the other {n - widths.iloc[0]} vary",
        f"- Height: median {sizes['height'].median():,.0f} px, tallest {tallest:,} px "
        f"({(sizes['height'] == tallest).sum()} pages cut at that height)",
        f"- Groups: {template['group_id'].nunique()} template groups ({len(template):,} pages), "
        f"{linked['group_id'].nunique()} small groups of linked pages ({len(linked)} pages), "
        f"{len(single):,} pages linked to no other page",
        f"- Test set: {len(test_groups)} template groups "
        f"({(template['split'] == 'test').sum()} pages: {', '.join(test_groups)}), "
        f"{(linked['split'] == 'test').sum()} pages from small groups, {(single['split'] == 'test').sum()} single pages",
        "",
        "## Table 1: label distribution", "",
        "| Label | Train | Test | Total | % |", "|---|---|---|---|---|",
    ]
    for task, (column, classes) in TASKS.items():
        lines.append(f"| **{task}** | | | | |")
        for label in classes:
            n_train, n_test = (train[column] == label).sum(), (test[column] == label).sum()
            lines.append(f"| {label} | {n_train:,} | {n_test:,} | {n_train + n_test:,} | "
                         f"{100 * (n_train + n_test) / n:.2f} |")
    lines.append(f"| Total | {len(train):,} | {len(test):,} | {n:,} | |")

    text = "\n".join(lines) + "\n"
    (out_dir / "dataset_stats.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
