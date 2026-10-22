import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
import tldextract

HERE = Path(__file__).resolve().parent


METHODS = (
    "ocr_visual_minhash_dbscan",
    "ocr_visual_minhash_hdbscan",
    "dinov2_pca_hdbscan",
    "geometry_pca_hdbscan",
)


FACTUALITY_MAP = {
    "VERY LOW": "LOW", "LOW": "LOW",
    "MIXED": "MIXED",
    "MOSTLY FACTUAL": "HIGH", "HIGH": "HIGH", "VERY HIGH": "HIGH",
}
BIAS_MAP = {
    "EXTREME LEFT": "LEFT", "LEFT": "LEFT",
    "LEFT-CENTER": "LEFT-CENTER",
    "LEAST BIASED": "LEAST BIASED",
    "RIGHT-CENTER": "RIGHT-CENTER",
    "RIGHT": "RIGHT", "EXTREME RIGHT": "RIGHT",
}
UNRATED = "UNRATED"
FACTUALITY_CLASSES = ("LOW", "MIXED", "HIGH")
BIAS_CLASSES = ("LEFT", "LEFT-CENTER", "LEAST BIASED", "RIGHT-CENTER", "RIGHT")


SPLIT_COLUMNS = [
    "media_name", "url", "timestamp", "country", "factuality",
    "trustworthiness", "bias_rating", "MBFC_source", "true_label_factuality", "true_label_bias",
]


FAMILY_MIN_SHARED = 3
FAMILY_MIN_OVERLAP = 0.5

NEAR_DUPLICATE_COSINE = 0.95


BIG_GROUP = 10
COMBINATIONS_KEPT = 50
JOINT_WEIGHT = 0.25
CLUSTERED_WEIGHT = 0.5


NETWORK_NAMES = {
    "Ada_Reporter.png": "pink_slime_local_A",
    "Suffolk_Reporter.png": "pink_slime_local_B",
    "Illinois_Business_Daily.png": "business_daily",
    "Amur_News.png": "pravda_network_ru",
    "Donetsk-News.png": "pravda_network_ua",
    "Biased.News.png": "natural_news_network",
    "Glasgow_Times.png": "uk_local_papers",
    "Kentucky_Lantern.png": "states_newsroom",
    "Castlegar_News.png": "black_press_canada",
    "Sudbury_Star.png": "postmedia_canada",
    "HigherEd_Dive.png": "industry_dive",
    "KKOH_-_780_AM_-_Reno.png": "talk_radio_stations",
    "Mexico_Daily_Post.png": "mexico_post_network",
    "Iowa_Star.png": "star_news_network",
    "Catholic_Tribune_Arizona.png": "catholic_tribune",
    "Current_Affairs.png": "mixed_ocr_cluster",
}


CELLS = [(f, b) for f in FACTUALITY_CLASSES for b in (*BIAS_CLASSES, UNRATED)]
CELL_INDEX = {cell: i for i, cell in enumerate(CELLS)}
FACTUALITY_TOTALS = np.array([[float(c[0] == f) for c in CELLS] for f in FACTUALITY_CLASSES])
BIAS_TOTALS = np.array([[float(c[1] == b) for c in CELLS] for b in BIAS_CLASSES])


BACKGROUND = "#fcfcfb"
TEXT = "#0b0b0b"
SOFT_TEXT = "#52514e"
AXIS = "#898781"
GRID = "#e1e0d9"
TRAIN_GREEN = "#2f9e44"
TEST_YELLOW = "#e0b400"

TIMESTAMP = re.compile(r"_\d{8}_\d{6}$")
TLD = tldextract.TLDExtract(suffix_list_urls=())


def screenshot_key(image):

    stem = TIMESTAMP.sub("", Path(image).stem.lower())
    return re.sub(r"[^a-z0-9]", "", stem)


def website(url):

    if not isinstance(url, str) or not url.strip():
        return ""
    url = url.strip()
    host = (urlsplit(url if "//" in url else "//" + url).hostname or "").lower()
    host = host.removeprefix("www.")
    parts = TLD(host)
    domain = getattr(parts, "top_domain_under_public_suffix", None) or parts.registered_domain
    return domain or host


class UnionFind:
    def __init__(self, items):
        self.parent = {item: item for item in items}

    def find(self, item):
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, a, b):
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self.parent[max(root_a, root_b)] = min(root_a, root_b)

    def groups(self):
        members = defaultdict(list)
        for item in self.parent:
            members[self.find(item)].append(item)
        return [sorted(group) for group in members.values()]


@dataclass
class Group:
    pages: list
    cells: np.ndarray
    clustered: int
    families: list
    group_id: str = ""
    name: str = ""
    split: str = "train"

    @property
    def size(self):
        return len(self.pages)


def load_pages(path):

    pages = pd.read_csv(path)
    if pages["image"].duplicated().any():
        raise ValueError(f"pages listed twice in {path}")
    pages["fact3"] = pages["factuality"].map(FACTUALITY_MAP)
    pages["bias5"] = pages["bias_rating"].map(BIAS_MAP)
    if pages["fact3"].isna().any():
        raise ValueError(f"unknown factuality ratings: {pages.loc[pages['fact3'].isna(), 'factuality'].unique()}")
    unknown = set(pages.loc[pages["bias5"].isna(), "bias_rating"]) - {UNRATED}
    if unknown:
        raise ValueError(f"unknown bias ratings: {unknown}")
    pages["cell"] = [(f, b if isinstance(b, str) else UNRATED) for f, b in zip(pages["fact3"], pages["bias5"])]
    pages["site"] = pages["url"].map(website)
    return pages.sort_values("image").reset_index(drop=True)


def load_clusters(path, images):


    table = pd.read_csv(path, dtype={"folder": str})
    table = table[table["used_for_split"] & table["method"].isin(METHODS) & table["image"].isin(set(images))]
    clusters = {}
    for method in METHODS:
        rows = table[table["method"] == method]
        for cluster in sorted(rows["folder"].unique()):
            clusters[(method, cluster)] = set(rows.loc[rows["folder"] == cluster, "image"])
    return clusters


def find_families(clusters):

    union = UnionFind(list(clusters))
    for (a, pages_a), (b, pages_b) in combinations(clusters.items(), 2):
        if a[0] == b[0]:
            continue
        shared = len(pages_a & pages_b)
        if shared >= FAMILY_MIN_SHARED and shared / min(len(pages_a), len(pages_b)) >= FAMILY_MIN_OVERLAP:
            union.union(a, b)
    families = sorted(union.groups(), key=lambda keys: (-len(set().union(*(clusters[k] for k in keys))), keys))
    family_of = {key: number for number, keys in enumerate(families) for key in keys}
    votes = defaultdict(Counter)
    for key, members in clusters.items():
        for page in members:
            votes[page][family_of[key]] += 1
    main_family = {page: min(v, key=lambda f: (-v[f], f)) for page, v in votes.items()}
    names = {}
    for number in range(len(families)):
        tags = [name for page, name in NETWORK_NAMES.items() if main_family.get(page) == number]
        names[number] = "+".join(tags) if tags else f"family_{number:02d}"
    return family_of, votes, main_family, names


def load_vectors(path, images):

    data = np.load(path)
    row_of = {key: i for i, key in enumerate(data["keys"])}
    keys = [screenshot_key(image) for image in images]
    missing = [image for image, key in zip(images, keys) if key not in row_of]
    if missing:
        raise KeyError(f"no DINOv2 vector for {missing[:5]} ({len(missing)} pages)")
    if len(row_of) != len(data["keys"]):
        raise ValueError("Duplicate DINOv2 feature keys")
    vectors = data["vectors"][[row_of[key] for key in keys]].astype(np.float64)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(vectors).all() or np.any(norms == 0):
        raise ValueError("DINOv2 features must be finite and nonzero")
    return vectors / norms


def find_links(pages, clusters, vectors, cosine):

    links = []

    def chain(members, reason):
        members = sorted(members)
        links.extend((members[0], other, reason) for other in members[1:])

    for members in clusters.values():
        chain(members, "cluster")
    for site, members in pages.groupby("site")["image"]:
        if site and len(members) > 1:
            chain(members, "same_website")
    for _, members in pages.groupby(pages["image"].map(screenshot_key))["image"]:
        if len(members) > 1:
            chain(members, "same_screenshot_key")
    images = pages["image"].tolist()
    rows, cols = np.where(np.triu(vectors @ vectors.T >= cosine, k=1))
    links.extend((images[r], images[c], "near_duplicate") for r, c in zip(rows, cols))
    return links


def make_groups(pages, links, main_family, family_names):
    union = UnionFind(pages["image"].tolist())
    reasons = defaultdict(set)
    for a, b, reason in links:
        union.union(a, b)
        reasons[a].add(reason)
        reasons[b].add(reason)

    cell_of = dict(zip(pages["image"], pages["cell"]))
    groups = []
    for members in union.groups():
        cells = np.zeros(len(CELLS))
        for page in members:
            cells[CELL_INDEX[cell_of[page]]] += 1
        family_sizes = Counter(main_family[p] for p in members if p in main_family)
        families = sorted(family_sizes, key=lambda f: (-family_sizes[f], f))
        groups.append(Group(members, cells, sum(family_sizes.values()), families))


    groups.sort(key=lambda g: (not g.families, -g.size, g.pages[0]))
    n_template = n_linked = 0
    for group in groups:
        if group.families:
            group.group_id = f"cluster_{n_template:02d}"
            group.name = "+".join(family_names[f] for f in group.families)
            n_template += 1
        elif group.size > 1:
            group.group_id = f"D{n_linked:02d}"
            group.name = "+".join(sorted(set().union(*(reasons[p] for p in group.pages))))
            n_linked += 1
    return groups, reasons


def link_pages(pages, clusters_file, vectors_file, cosine):

    clusters = load_clusters(clusters_file, pages["image"])
    family_of, votes, main_family, family_names = find_families(clusters)
    vectors = load_vectors(vectors_file, pages["image"].tolist())
    links = find_links(pages, clusters, vectors, cosine)
    groups, reasons = make_groups(pages, links, main_family, family_names)
    return SimpleNamespace(clusters=clusters, family_of=family_of, main_family=main_family,
                           family_names=family_names, vectors=vectors, links=links, groups=groups, reasons=reasons)


class SplitLoss:


    def __init__(self, all_cells, all_clustered, fraction):
        self.cells = fraction * all_cells
        self.factuality = FACTUALITY_TOTALS @ self.cells
        self.bias = BIAS_TOTALS @ self.cells
        self.clustered = fraction * all_clustered

    def __call__(self, cells, clustered):
        factuality = FACTUALITY_TOTALS @ cells
        bias = BIAS_TOTALS @ cells
        return float(
            np.sum((factuality - self.factuality) ** 2 / np.maximum(self.factuality, 1.0))
            + np.sum((bias - self.bias) ** 2 / np.maximum(self.bias, 1.0))
            + JOINT_WEIGHT * np.sum((cells - self.cells) ** 2 / np.maximum(self.cells, 1.0))
            + CLUSTERED_WEIGHT * (clustered - self.clustered) ** 2 / max(self.clustered, 1.0)
        )


def fill_with_singles(loss, base_cells, base_clustered, needed, available):

    if needed < 0 or needed > available.sum():
        return None
    take = np.clip(np.rint(loss.cells - base_cells), 0, available)
    while take.sum() != needed:
        step = 1 if take.sum() < needed else -1
        best, best_loss = None, np.inf
        for i in range(len(take)):
            if (step > 0 and take[i] >= available[i]) or (step < 0 and take[i] <= 0):
                continue
            take[i] += step
            value = loss(base_cells + take, base_clustered)
            take[i] -= step
            if value < best_loss:
                best, best_loss = i, value
        take[best] += step
    return take


def polish(loss, base_cells, base_clustered, take, available):

    current = loss(base_cells + take, base_clustered)
    improved = True
    while improved:
        improved = False
        for i in range(len(take)):
            for j in range(len(take)):
                if i == j or take[i] <= 0 or take[j] >= available[j]:
                    continue
                take[i] -= 1
                take[j] += 1
                candidate = loss(base_cells + take, base_clustered)
                if candidate < current - 1e-12:
                    current, improved = candidate, True
                else:
                    take[i] += 1
                    take[j] -= 1
    return take, current


def search_split(groups, fraction, seed):

    all_cells = sum(g.cells for g in groups)
    n_pages = int(all_cells.sum())
    test_size = int(round(fraction * n_pages))
    loss = SplitLoss(all_cells, sum(g.clustered for g in groups), fraction)


    big = [g for g in groups if BIG_GROUP <= g.size <= test_size]
    small = [g for g in groups if 1 < g.size < BIG_GROUP]
    singles = [g for g in groups if g.size == 1]
    available = sum((g.cells for g in singles), np.zeros(len(CELLS)))

    def score(cells, size, clustered):
        take = fill_with_singles(loss, cells, clustered, test_size - size, available)
        if take is None:
            return np.inf
        return loss(cells + take, clustered)


    big_cells = [g.cells for g in big]
    candidates = []
    for mask in range(1 << len(big)):
        chosen = [i for i in range(len(big)) if mask >> i & 1]
        cells = sum((big_cells[i] for i in chosen), np.zeros(len(CELLS)))
        size = sum(big[i].size for i in chosen)
        if size > test_size:
            continue
        clustered = sum(big[i].clustered for i in chosen)
        candidates.append((score(cells, size, clustered), mask))
    candidates.sort()


    best = None
    for _, mask in candidates[:COMBINATIONS_KEPT]:
        chosen = [big[i] for i in range(len(big)) if mask >> i & 1]
        in_test = [False] * len(small)

        def totals():
            members = chosen + [g for g, flag in zip(small, in_test) if flag]
            return (
                sum((g.cells for g in members), np.zeros(len(CELLS))),
                sum(g.size for g in members),
                sum(g.clustered for g in members),
            )

        current = score(*totals())
        improved = True
        while improved:
            improved = False
            for j in range(len(small)):
                in_test[j] = not in_test[j]
                candidate = score(*totals())
                if candidate < current - 1e-12:
                    current, improved = candidate, True
                else:
                    in_test[j] = not in_test[j]
        if best is None or current < best[0] - 1e-12:
            best = (current, mask, list(in_test))

    _, mask, in_test = best
    test_groups = [big[i] for i in range(len(big)) if mask >> i & 1]
    test_groups += [g for g, flag in zip(small, in_test) if flag]
    base_cells = sum((g.cells for g in test_groups), np.zeros(len(CELLS)))
    base_size = sum(g.size for g in test_groups)
    base_clustered = sum(g.clustered for g in test_groups)
    take = fill_with_singles(loss, base_cells, base_clustered, test_size - base_size, available)
    take, final_loss = polish(loss, base_cells, base_clustered, take, available)

    for group in groups:
        group.split = "train"
    for group in test_groups:
        group.split = "test"


    rng = np.random.default_rng(seed)
    by_cell = defaultdict(list)
    for group in singles:
        by_cell[int(np.argmax(group.cells))].append(group)
    for cell, members in sorted(by_cell.items()):
        members.sort(key=lambda g: g.pages[0])
        for k in rng.choice(len(members), size=int(take[cell]), replace=False):
            members[k].split = "test"
    return {"pages": n_pages, "test_size": test_size, "loss": final_loss, "big_groups": len(big)}


def keep_sides(groups, side_of):

    for group in groups:
        sides = {side_of[p] for p in group.pages}
        if len(sides) != 1:
            raise AssertionError(f"group {group.pages[:3]} would be split")
        group.split = sides.pop()


def split_loss(groups, fraction):

    all_cells = sum(g.cells for g in groups)
    loss = SplitLoss(all_cells, sum(g.clustered for g in groups), fraction)
    test = [g for g in groups if g.split == "test"]
    test_cells = sum((g.cells for g in test), np.zeros(len(CELLS)))
    n_pages = int(all_cells.sum())
    return {
        "pages": n_pages,
        "test_size": int(sum(g.size for g in test)),
        "loss": loss(test_cells, sum(g.clustered for g in test)),
        "big_groups": sum(BIG_GROUP <= g.size <= int(round(fraction * n_pages)) for g in groups),
    }


def check_split(pages, clusters, links, groups):

    side = {page: g.split for g in groups for page in g.pages}
    if set(side) != set(pages["image"]) or sum(g.size for g in groups) != len(pages):
        raise AssertionError("every page must be placed exactly once")
    for (method, cluster), members in clusters.items():
        if len({side[p] for p in members}) > 1:
            raise AssertionError(f"{method}/{cluster} is on both sides")
    for a, b, reason in links:
        if side[a] != side[b]:
            raise AssertionError(f"{a} and {b} ({reason}) are on different sides")
    return side


def clusters_on_both_sides(clusters, side):
    return sum(len({side[p] for p in members}) > 1 for members in clusters.values())


def closest_train_page(pages, vectors, side):

    is_test = np.array([side[p] == "test" for p in pages["image"]])
    best = (vectors[is_test] @ vectors[~is_test].T).max(axis=1)
    return {
        "test_pages": int(is_test.sum()),
        "at_0.999": int((best >= 0.999).sum()),
        "at_0.99": int((best >= 0.99).sum()),
        "at_0.95": int((best >= 0.95).sum()),
        "at_0.90": int((best >= 0.90).sum()),
        "median": float(np.median(best)),
    }


def label_distribution(pages, side):

    frame = pages.assign(split=pages["image"].map(side))
    rows = []
    for task, column, classes in (("factuality", "fact3", FACTUALITY_CLASSES), ("bias", "bias5", BIAS_CLASSES)):
        rated = frame[frame[column].notna()]
        for label in classes:
            row = {"task": task, "label": label}
            for split in ("train", "test"):
                part = rated[rated["split"] == split]
                row[f"{split}_n"] = int((part[column] == label).sum())
                row[f"{split}_pct"] = 100 * row[f"{split}_n"] / max(len(part), 1)
            rows.append(row)
    table = pd.DataFrame(rows)
    table["diff_pp"] = table["test_pct"] - table["train_pct"]
    return table


def tidy_axes(ax):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=SOFT_TEXT, labelsize=10)
    ax.set_facecolor(BACKGROUND)
    ax.set_axisbelow(True)


def plot_distribution(table, task, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    part = table[table["task"] == task]
    labels = part["label"].tolist()
    x = np.arange(len(labels))
    n_train, n_test = int(part["train_n"].sum()), int(part["test_n"].sum())
    fig, ax = plt.subplots(figsize=(max(7.5, 1.9 * len(labels) + 2), 6))
    fig.patch.set_facecolor(BACKGROUND)
    tidy_axes(ax)
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    for offset, split, color, n in ((-0.2, "train", TRAIN_GREEN, n_train), (0.2, "test", TEST_YELLOW, n_test)):
        bars = ax.bar(x + offset, part[f"{split}_pct"], 0.38, color=color, label=f"{split.title()} (n={n:,})", zorder=3)
        for bar, pct, count in zip(bars, part[f"{split}_pct"], part[f"{split}_n"]):
            ax.annotate(f"{pct:.1f}%\n{count:,}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                        fontsize=8.5, color=TEXT, linespacing=1.25)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Share of split (%)", color=SOFT_TEXT, fontsize=11)
    ax.set_ylim(0, part[["train_pct", "test_pct"]].to_numpy().max() * 1.22)
    ax.legend(frameon=False, loc="upper right", fontsize=10, labelcolor=SOFT_TEXT)
    fig.suptitle(title, x=0.06, y=0.98, ha="left", fontsize=14, color=TEXT, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150, facecolor=BACKGROUND)
    plt.close(fig)


def plot_groups(groups, pages, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fact_of = dict(zip(pages["image"], pages["fact3"]))
    bias_of = dict(zip(pages["image"], pages["bias5"]))
    shown = [g for g in groups if g.group_id.startswith("cluster_")]
    fig, ax = plt.subplots(figsize=(12, 0.62 * len(shown) + 1.8))
    fig.patch.set_facecolor(BACKGROUND)
    tidy_axes(ax)
    ax.xaxis.grid(True, color=GRID, linewidth=1)
    y = np.arange(len(shown))[::-1]
    sizes = [g.size for g in shown]
    ax.barh(y, sizes, height=0.62, color=[TRAIN_GREEN if g.split == "train" else TEST_YELLOW for g in shown], zorder=3)
    for yi, group in zip(y, shown):
        fact_text = ", ".join(f"{k} {v}" for k, v in Counter(fact_of[p] for p in group.pages).most_common())
        bias_counts = Counter(bias_of[p] for p in group.pages if isinstance(bias_of[p], str)).most_common()
        bias_text = ", ".join(f"{k} {v}" for k, v in bias_counts) if bias_counts else "-"
        ax.annotate(f"{group.size:,}\nfactuality: {fact_text}\nbias: {bias_text}",
                    (group.size, yi), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=TEXT, linespacing=1.6)
    ax.set_yticks(y)
    ax.set_yticklabels([g.group_id for g in shown], fontsize=9)
    ax.set_xlim(0, max(sizes) * 1.9)
    ax.set_xlabel("Pages", color=SOFT_TEXT)
    ax.legend(handles=[Patch(color=TRAIN_GREEN, label="Train"), Patch(color=TEST_YELLOW, label="Test")],
              frameon=False, loc="lower right", fontsize=10, labelcolor=SOFT_TEXT)
    fig.suptitle("Template groups and their split", x=0.02, y=0.99, ha="left", fontsize=14, color=TEXT,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150, facecolor=BACKGROUND)
    plt.close(fig)


def counts(values):
    return ", ".join(f"{k} {v}" for k, v in Counter(values).most_common())


def write_files(out_dir, pages, linked, side):

    group_of = {p: g for g in linked.groups for p in g.pages}
    cluster_of = {method: {} for method in METHODS}
    for (method, cluster), members in linked.clusters.items():
        for page in members:
            cluster_of[method][page] = cluster

    assignments = pages[["image", "media_name", "url", "site", "factuality", "fact3",
                         "bias_rating", "bias5"]].copy()
    assignments.insert(1, "split", assignments["image"].map(side))
    assignments.insert(2, "group_id", assignments["image"].map(lambda p: group_of[p].group_id))
    assignments.insert(3, "group_name", assignments["image"].map(lambda p: group_of[p].name))
    assignments.insert(4, "family", assignments["image"].map(
        lambda p: linked.family_names.get(linked.main_family.get(p), "")))
    assignments.insert(5, "linked_by", assignments["image"].map(lambda p: "+".join(sorted(linked.reasons[p]))))
    for method in METHODS:
        assignments[method] = assignments["image"].map(cluster_of[method]).fillna("")
    assignments.to_csv(out_dir / "split_assignments.csv", index=False)

    rows = pages.assign(true_label_factuality=pages["fact3"], true_label_bias=pages["bias5"],
                        split=pages["image"].map(side))
    for split in ("train", "test"):
        rows.loc[rows["split"] == split, SPLIT_COLUMNS].to_csv(out_dir / f"{split}.csv", index=False)

    fact_of = dict(zip(pages["image"], pages["fact3"]))
    bias_of = dict(zip(pages["image"], pages["bias5"]))
    group_rows = [{
        "group_id": g.group_id,
        "split": g.split,
        "n": g.size,
        "families": g.name,
        "factuality": counts(fact_of[p] for p in g.pages),
        "bias": counts(bias_of[p] if isinstance(bias_of[p], str) else UNRATED for p in g.pages),
        "members": ", ".join(p.removesuffix(".png") for p in g.pages) if not g.families else "",
    } for g in linked.groups if g.group_id]
    groups_table = pd.DataFrame(group_rows)
    groups_table.to_csv(out_dir / "groups.csv", index=False)

    family_rows = []
    for number, name in linked.family_names.items():
        members = sorted(p for p, f in linked.main_family.items() if f == number)
        keys = sorted(k for k, f in linked.family_of.items() if f == number)
        family_rows.append({
            "family_id": f"F{number:02d}",
            "family": name,
            "n": len(members),
            "group_id": group_of[members[0]].group_id,
            "split": side[members[0]],
            "method_clusters": "; ".join(f"{m}/{c}" for m, c in keys),
            "factuality": counts(fact_of[p] for p in members),
            "bias": counts(bias_of[p] if isinstance(bias_of[p], str) else UNRATED for p in members),
        })
    pd.DataFrame(family_rows).to_csv(out_dir / "families.csv", index=False)
    return groups_table


def parse_args():
    parser = argparse.ArgumentParser(description="Template-aware train/test split of the MEFI screenshots.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=HERE / "output" / "split")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--near-duplicate-cosine", type=float, default=NEAR_DUPLICATE_COSINE)
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir, out_dir = args.data_dir.resolve(), args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    clusters_file = data_dir / "curated_clusters.csv"
    vectors_file = data_dir / "dinov2_vectors.npz"


    pages = load_pages(data_dir / "mbfc_labels.csv")
    everything = link_pages(pages, clusters_file, vectors_file, args.near_duplicate_cosine)
    search = search_split(everything.groups, args.test_fraction, args.seed)
    side_of = {p: g.split for g in everything.groups for p in g.pages}


    unrated = pages[pages["bias5"].isna()]
    pages = pages[pages["bias5"].notna()].reset_index(drop=True)
    linked = link_pages(pages, clusters_file, vectors_file, args.near_duplicate_cosine)
    keep_sides(linked.groups, side_of)
    for group in linked.groups:
        if group.group_id.startswith("cluster_"):
            group.group_id = f"{group.group_id}_{group.split}"
    side = check_split(pages, linked.clusters, linked.links, linked.groups)

    groups_table = write_files(out_dir, pages, linked, side)
    unrated[["image", "media_name", "url"]].assign(reason="no MBFC bias rating").to_csv(
        out_dir / "excluded_pages.csv", index=False)

    table = label_distribution(pages, side)
    table.to_csv(out_dir / "label_distribution.csv", index=False)
    plot_distribution(table, "factuality", "Factuality distribution: train vs test",
                      out_dir / "factuality_distribution.png")
    plot_distribution(table, "bias", "Bias distribution: train vs test", out_dir / "bias_distribution.png")
    plot_groups(linked.groups, pages, out_dir / "groups_by_split.png")


    comparisons = (("template-aware", side),)
    leakage = pd.DataFrame([
        {"split": name, **closest_train_page(pages, linked.vectors, assignment),
         "clusters_on_both_sides": clusters_on_both_sides(linked.clusters, assignment)}
        for name, assignment in comparisons
    ])
    leakage.to_csv(out_dir / "leakage_check.csv", index=False)

    groups = linked.groups
    summary = {
        "pages": len(pages),
        "train": sum(s == "train" for s in side.values()),
        "test": sum(s == "test" for s in side.values()),
        "search": search,
        "final_split": split_loss(groups, args.test_fraction),
        "pages_without_bias_rating": unrated["image"].tolist(),
        "clusters": len(linked.clusters),
        "families": len(linked.family_names),
        "template_groups": sum(g.group_id.startswith("cluster_") for g in groups),
        "pages_in_template_groups": sum(g.size for g in groups if g.group_id.startswith("cluster_")),
        "other_linked_groups": sum(g.group_id.startswith("D") for g in groups),
        "pages_in_other_linked_groups": sum(g.size for g in groups if g.group_id.startswith("D")),
        "single_pages": sum(g.size == 1 for g in groups),
        "pages_per_link_type": {r: sum(r in v for v in linked.reasons.values())
                                for r in ("cluster", "same_website", "same_screenshot_key", "near_duplicate")},
        "largest_class_gap_pp": {t: float(table.loc[table["task"] == t, "diff_pp"].abs().max())
                                 for t in ("factuality", "bias")},
        "methods": list(METHODS),
        "near_duplicate_cosine": args.near_duplicate_cosine,
        "seed": args.seed,
        "test_fraction": args.test_fraction,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Searched over {search['pages']} pages (test size {search['test_size']}, loss {search['loss']:.4f}).")
    print(f"Dropped {len(unrated)} pages without a bias rating: {', '.join(unrated['image'])}.")
    print(f"Final split: {summary['train']} train / {summary['test']} test.\n")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    print()
    print(groups_table.drop(columns=["members"]).to_string(index=False))
    print()
    print(leakage.to_string(index=False))
    print(f"\nWrote {out_dir}")


if __name__ == "__main__":
    main()
