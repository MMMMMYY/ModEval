"""
Section 4.1 — Modifier Category Harmonization
==============================================
Encodes the 16 raw category descriptions with Sentence-BERT (all-mpnet-base-v2),
computes a 16×16 cosine-similarity matrix, then determines the optimal merging
threshold via silhouette analysis over [0.50, 0.90] (Rousseeuw, 1987).

Unlike prior work that applies a fixed threshold for near-duplicate detection
(Chakrabarty, 2022), cross-source category alignment requires a lower threshold
because equivalent concepts are expressed with varying terminology across sources.

Outputs
-------
outputs/4_1/
  similarity_matrix.npy          – raw 16×16 similarity values
  figure_silhouette.pdf          – silhouette score vs threshold curve
  figure3_heatmap.pdf            – 16×16 heatmap with cluster boundaries
  unified_taxonomy.json          – cluster assignments + unified names
  table5_mapping.csv             – Table 5: source → unified mapping
"""

import json
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import fcluster, linkage
from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

from taxonomy import RAW_CATEGORIES

# ── Config ────────────────────────────────────────────────────────────────────
SBERT_MODEL     = "all-mpnet-base-v2"
OUTPUT_DIR      = "outputs/4_1"
SILHOUETTE_LO   = 0.20
SILHOUETTE_HI   = 0.90
SILHOUETTE_STEP = 0.01

# Unified names — mapped by *member IDs*, not by volatile cluster IDs.
# Keys: frozenset of category IDs in a cluster → unified name.
# This is stable across re-runs regardless of scipy's internal cluster numbering.
CLUSTER_NAME_BY_MEMBERS: dict[frozenset, str] = {
    frozenset({"CI_Artist", "OP_Style", "LC_ArtisticStyle", "HA_Stylistic"}): "Artist",
    frozenset({"CI_Medium", "LC_TechMedium"}):                                "Medium",
    frozenset({"LC_AestheticDesc", "HA_Thematic"}):                          "Atmosphere",
    frozenset({"CI_Flavor", "OP_Quality", "LC_QualityTerms", "HA_Quality"}): "Quality",
    frozenset({"CI_Movement"}):                                               "Movement",
    frozenset({"CI_Trending"}):                                               "Trending",
    frozenset({"OP_Magic"}):                                                  "Magic",
    frozenset({"OP_Repeating"}):                                              "Repeating",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Encoding & similarity ─────────────────────────────────────────────────────

def encode_descriptions(model: SentenceTransformer, categories: list[dict]) -> np.ndarray:
    descriptions = [c["description"] for c in categories]
    return model.encode(descriptions, convert_to_numpy=True, show_progress_bar=True)


def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    return cosine_similarity(embeddings)


# ── Silhouette analysis ───────────────────────────────────────────────────────

def silhouette_sweep(
    dist_matrix: np.ndarray,
    Z: np.ndarray,
    lo: float = SILHOUETTE_LO,
    hi: float = SILHOUETTE_HI,
    step: float = SILHOUETTE_STEP,
) -> tuple[list[float], list[float], list[int], float, int]:
    """
    Sweep cosine-similarity thresholds in [lo, hi].
    Returns (thresholds, scores, n_clusters_list, optimal_threshold, optimal_k).
    """
    thresholds, scores, n_clusters_list = [], [], []
    best_thr, best_score, best_k = lo, -1.0, 2

    for thr in np.arange(lo, hi + step / 2, step):
        labels = fcluster(Z, t=1.0 - thr, criterion="distance")
        k = len(set(labels))
        if k < 2 or k >= len(labels):          # silhouette undefined for k=1 or k=N
            continue
        score = silhouette_score(dist_matrix, labels, metric="precomputed")
        thresholds.append(round(float(thr), 4))
        scores.append(float(score))
        n_clusters_list.append(int(k))
        if score > best_score:
            best_score, best_thr, best_k = score, round(float(thr), 4), int(k)

    return thresholds, scores, n_clusters_list, best_thr, best_k


def plot_silhouette_curve(
    thresholds: list[float],
    scores: list[float],
    n_clusters_list: list[int],
    optimal_thr: float,
    optimal_k: int,
    out_path: str,
) -> None:
    fig, ax1 = plt.subplots(figsize=(7, 3.8))

    # Silhouette score (primary y-axis)
    color_score = "#2c7bb6"
    ax1.plot(thresholds, scores, color=color_score, linewidth=2, zorder=3, label="Silhouette score")
    ax1.set_xlabel("Cosine similarity threshold", fontsize=11)
    ax1.set_ylabel("Silhouette score", color=color_score, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=color_score)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    # Number of clusters (secondary y-axis)
    ax2 = ax1.twinx()
    color_k = "#d7191c"
    ax2.step(thresholds, n_clusters_list, color=color_k, linewidth=1.4,
             linestyle="--", alpha=0.7, where="mid", label="# clusters")
    ax2.set_ylabel("Number of clusters", color=color_k, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=color_k)
    ax2.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # Mark optimal point
    opt_score = scores[thresholds.index(optimal_thr)]
    ax1.axvline(optimal_thr, color="gray", linewidth=1.2, linestyle=":", zorder=2)
    ax1.scatter([optimal_thr], [opt_score], color=color_score, s=80, zorder=5)
    ax1.annotate(
        f"  threshold = {optimal_thr}\n  k = {optimal_k}\n  score = {opt_score:.3f}",
        xy=(optimal_thr, opt_score),
        xytext=(optimal_thr + 0.03, opt_score - 0.015),
        fontsize=8.5,
        color="black",
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.9),
    )

    # Combined legend
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, loc="upper left", fontsize=8.5,
               framealpha=0.85)

    ax1.set_xlim(min(thresholds) - 0.01, max(thresholds) + 0.01)
    ax1.grid(axis="y", linestyle=":", alpha=0.4)
    fig.suptitle(
        "Silhouette analysis for threshold selection\n"
        "(SBERT all-mpnet-base-v2, average linkage)",
        fontsize=10, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[4.1] Silhouette curve saved → {out_path}")


# ── Clustering ────────────────────────────────────────────────────────────────

def cluster_categories(
    sim_matrix: np.ndarray, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    dist_matrix = np.clip(1.0 - sim_matrix, 0, None)
    n = dist_matrix.shape[0]
    condensed = dist_matrix[np.triu_indices(n, k=1)]
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=1.0 - threshold, criterion="distance")
    return labels, Z, dist_matrix


# ── Heatmap ───────────────────────────────────────────────────────────────────

# Source → colour for row/col label ticks (4 clearly distinct colours)
SOURCE_COLORS = {
    "CLIP Interrogator": "#2166ac",   # strong blue
    "Oppenlaender":      "#e31a1c",   # red
    "Liu & Chilton":     "#33a02c",   # green
    "Hao et al.":        "#ff7f00",   # orange
}


def plot_heatmap(
    sim_matrix: np.ndarray,
    categories: list[dict],
    labels: np.ndarray,
    unified_names: dict[int, str],
    optimal_thr: float,
    optimal_k: int,
    out_path: str,
) -> None:
    # Sort so same-cluster entries are adjacent
    order = np.argsort(labels, kind="stable")
    sim_sorted  = sim_matrix[np.ix_(order, order)]
    cats_sorted = [categories[i] for i in order]
    lbls_sorted = labels[order]

    tick_labels = [f"{c['id']}" for c in cats_sorted]
    n_cats = len(cats_sorted)

    # Compute cluster boundaries once
    boundaries = [
        i for i in range(1, len(lbls_sorted))
        if lbls_sorted[i] != lbls_sorted[i - 1]
    ]

    # ── Collect cluster spans (start, end, name) ──────────────────────────────
    cluster_spans = []
    prev_lbl, start = lbls_sorted[0], 0
    for i in range(1, n_cats + 1):
        cur_lbl = lbls_sorted[i] if i < n_cats else -1
        if cur_lbl != prev_lbl:
            name = unified_names.get(int(prev_lbl), f"C{prev_lbl}")
            cluster_spans.append((start, i, name))
            start, prev_lbl = i, cur_lbl

    # ── Figure layout: small top strip for cluster labels + main heatmap ──────
    fig = plt.figure(figsize=(11, 10))
    # gridspec: top row = cluster label strip, bottom row = heatmap
    gs = fig.add_gridspec(
        2, 1,
        height_ratios=[0.12, 1],
        hspace=0.02,
    )
    ax_top = fig.add_subplot(gs[0])
    ax     = fig.add_subplot(gs[1])

    # ── Top strip: cluster name bars ──────────────────────────────────────────
    CLUSTER_COLORS = [
        "#a6cee3", "#1f78b4", "#b2df8a", "#33a02c",
        "#fb9a99", "#e31a1c", "#fdbf6f", "#ff7f00",
    ]
    ax_top.set_xlim(0, n_cats)
    ax_top.set_ylim(0, 1)
    ax_top.axis("off")

    for idx, (s, e, name) in enumerate(cluster_spans):
        color = CLUSTER_COLORS[idx % len(CLUSTER_COLORS)]
        ax_top.barh(
            0.5, e - s - 0.1, left=s + 0.05,
            height=0.82, color=color, align="center",
        )
        ax_top.text(
            (s + e) / 2, 0.5, name,
            ha="center", va="center",
            fontsize=8, fontweight="bold", color="black",
            rotation=30,
            bbox=None,
        )
    # Vertical dividers between clusters
    for b in boundaries:
        ax_top.axvline(b, color="white", linewidth=2)

    # ── Heatmap ───────────────────────────────────────────────────────────────
    sns.heatmap(
        sim_sorted,
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        vmin=0.0, vmax=1.0,
        cmap="YlOrRd",
        linewidths=0.4,
        linecolor="white",
        annot=True, fmt=".2f",
        annot_kws={"size": 6.5},
        ax=ax,
        cbar_kws={"shrink": 0.55, "label": "Cosine similarity", "pad": 0.01},
    )

    # Cluster boundary lines on heatmap
    for b in boundaries:
        ax.axhline(b, color="white", linewidth=2.5)
        ax.axvline(b, color="white", linewidth=2.5)

    # Colour tick labels by source
    for tick, cat in zip(ax.get_xticklabels(), cats_sorted):
        tick.set_color(SOURCE_COLORS.get(cat["source"], "black"))
        tick.set_fontsize(7.5)
    for tick, cat in zip(ax.get_yticklabels(), cats_sorted):
        tick.set_color(SOURCE_COLORS.get(cat["source"], "black"))
        tick.set_fontsize(7.5)

    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)

    # Source legend (bottom-right, outside heatmap)
    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor=SOURCE_COLORS[s], label=s)
        for s in SOURCE_COLORS
    ]
    ax.legend(
        handles=legend_els, title="Source",
        loc="lower right", bbox_to_anchor=(1.28, 0.0),
        fontsize=7.5, title_fontsize=8, framealpha=0.9,
    )

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[4.1] Heatmap saved → {out_path}")


# ── Taxonomy & Table 5 ────────────────────────────────────────────────────────

def resolve_cluster_names(
    categories: list[dict], labels: np.ndarray
) -> dict[int, str]:
    """
    Map each cluster ID → unified name using CLUSTER_NAME_BY_MEMBERS.
    Falls back to the most descriptive member name if no override is found.
    """
    from collections import defaultdict
    cluster_ids: dict[int, list[str]] = defaultdict(list)
    for cat, lbl in zip(categories, labels):
        cluster_ids[int(lbl)].append(cat["id"])

    result: dict[int, str] = {}
    for cid, member_ids in cluster_ids.items():
        key = frozenset(member_ids)
        if key in CLUSTER_NAME_BY_MEMBERS:
            result[cid] = CLUSTER_NAME_BY_MEMBERS[key]
        else:
            # Fallback: CLIP Interrogator name > first member name
            ci_members = [c for c in categories
                          if c["id"] in member_ids and c["source"] == "CLIP Interrogator"]
            fallback = ci_members[0]["name"] if ci_members else member_ids[0]
            result[cid] = fallback
    return result


def build_taxonomy(categories: list[dict], labels: np.ndarray) -> dict:
    names = resolve_cluster_names(categories, labels)
    taxonomy: dict = {}
    for cat, lbl in zip(categories, labels):
        name = names[int(lbl)]
        if name not in taxonomy:
            taxonomy[name] = {"cluster_id": int(lbl), "members": []}
        taxonomy[name]["members"].append({
            "id": cat["id"],
            "source": cat["source"],
            "original_name": cat["name"],
            "description": cat["description"],
        })
    return taxonomy


def build_table5(taxonomy: dict) -> pd.DataFrame:
    rows = []
    for unified_name, info in taxonomy.items():
        for m in info["members"]:
            rows.append({
                "Unified Category": unified_name,
                "Source": m["source"],
                "Original Name": m["original_name"],
                "Description": m["description"],
            })
    return pd.DataFrame(rows).sort_values(["Unified Category", "Source"])


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("[4.1] Loading Sentence-BERT model …")
    model = SentenceTransformer(SBERT_MODEL)

    print("[4.1] Encoding 16 category descriptions …")
    embeddings = encode_descriptions(model, RAW_CATEGORIES)

    print("[4.1] Computing 16×16 cosine similarity matrix …")
    sim_matrix = compute_similarity_matrix(embeddings)
    np.save(os.path.join(OUTPUT_DIR, "similarity_matrix.npy"), sim_matrix)

    labels_tmp, Z, dist_matrix = cluster_categories(sim_matrix, threshold=0.70)

    print(f"[4.1] Silhouette sweep over [{SILHOUETTE_LO}, {SILHOUETTE_HI}] …")
    thresholds, scores, n_clusters_list, opt_thr, opt_k = silhouette_sweep(
        dist_matrix, Z
    )
    print(f"[4.1] → Optimal threshold = {opt_thr}  (k = {opt_k}, "
          f"silhouette = {scores[thresholds.index(opt_thr)]:.4f})")

    plot_silhouette_curve(
        thresholds, scores, n_clusters_list, opt_thr, opt_k,
        os.path.join(OUTPUT_DIR, "figure_silhouette.pdf"),
    )

    print(f"[4.1] Clustering at optimal threshold = {opt_thr} …")
    labels, Z, dist_matrix = cluster_categories(sim_matrix, opt_thr)
    n_clusters = len(set(labels))
    print(f"[4.1] → {n_clusters} unified categories")

    unified_names = resolve_cluster_names(RAW_CATEGORIES, labels)

    print("\n[4.1] Cluster assignments:")
    for cat, lbl in zip(RAW_CATEGORIES, labels):
        name = unified_names[int(lbl)]
        print(f"  [{name:12s}] Cluster {lbl:2d} | {cat['source']:20s} | {cat['name']}")

    plot_heatmap(
        sim_matrix, RAW_CATEGORIES, labels, unified_names, opt_thr, opt_k,
        os.path.join(OUTPUT_DIR, "figure3_heatmap.pdf"),
    )

    taxonomy = build_taxonomy(RAW_CATEGORIES, labels)
    taxonomy_path = os.path.join(OUTPUT_DIR, "unified_taxonomy.json")
    with open(taxonomy_path, "w") as f:
        json.dump(taxonomy, f, indent=2)
    print(f"[4.1] Unified taxonomy → {taxonomy_path}")

    df_table5 = build_table5(taxonomy)
    table5_path = os.path.join(OUTPUT_DIR, "table5_mapping.csv")
    df_table5.to_csv(table5_path, index=False)
    print(f"[4.1] Table 5 → {table5_path}")

    with open("outputs/unified_taxonomy.json", "w") as f:
        json.dump(taxonomy, f, indent=2)

    # Summary stats for paper
    print(f"\n[4.1] Summary for paper:")
    print(f"  Silhouette sweep range : [{SILHOUETTE_LO}, {SILHOUETTE_HI}], step={SILHOUETTE_STEP}")
    print(f"  Optimal threshold      : {opt_thr}")
    print(f"  Optimal k              : {opt_k}")
    print(f"  Peak silhouette score  : {scores[thresholds.index(opt_thr)]:.4f}")


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    main()
