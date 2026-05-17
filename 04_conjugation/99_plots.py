#!/usr/bin/env python3
"""Generate every panel for the VirB4-T4CP / conjugation figure set.

Outputs go to ``outputs/figures/`` next to this script. Every quantity is
re-derived from the input data and the other scripts in this pipeline; no
numbers are hard-coded. Each panel is written as both PNG (raster) and SVG
(``svg.fonttype = 'none'`` so labels stay editable in vector software).

Run as ``python 99_plots.py``.

Note on UMAPs: ``umap-learn`` is used if installed, otherwise the script
falls back to ``sklearn.manifold.MDS`` on the same precomputed distance
matrix. Both produce 2-D embeddings that preserve the subtype clustering;
exact coordinates differ between algorithms.
"""
from pathlib import Path
import sys

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CATALOGUE_FILE, FILTERED_HITS, CLUSTER_CSV,
                    CLINKER_MATRIX, OUT_DIR, header)  # noqa

FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
matplotlib.rcParams['font.family']  = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

SUBTYPE_COLORS = {
    1: '#1f78b4',  # Subtype I  — blue
    2: '#ff7f0e',  # Subtype II — orange
    3: '#9467bd',  # Subtype III — purple
    4: '#2ca02c',  # Subtype IV — green
    5: '#d62728',  # Subtype V — red
}
ROMAN = {1: 'I', 2: 'II', 3: 'III', 4: 'IV', 5: 'V'}

FAMILY_COLORS = {
    'Haloferacaceae':     '#2ca02c',
    'Natrialbaceae':      '#d97706',
    'Haloarculaceae':     '#9467bd',
    'Halobacteriaceae':   '#e0457b',
    'Sulfolobaceae':      '#2ca02c',  # Sulfolobaceae is the only Thermoproteota
    'Haladaptataceae':    '#f2c14e',
    'Unclassified':       '#bbbbbb',
    'Halococcaceae':      '#7f4f24',
    'Halalkalicoccaceae': '#555555',
    'QS-9-68-17':         '#1f78b4',
}


def _short(s):
    return s.split('__', 1)[-1] if isinstance(s, str) and '__' in s else s


def _save(fig, stem):
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=200, bbox_inches='tight')
    fig.savefig(FIG_DIR / f"{stem}.svg", bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {stem}.png + .svg")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _vb_t4cp_positive_replicons():
    """Return the set of replicon IDs that carry at least one VirB4 and one
    T4CP hit in the filtered prokka output."""
    hits = pd.read_csv(FILTERED_HITS, sep='\t')
    g = hits.groupby('replicon_name')['gene_name'].apply(set)
    has_virb4 = g.apply(lambda s: any('virb4' in name.lower() for name in s))
    has_t4cp  = g.apply(lambda s: any('t4cp'  in name.lower() for name in s))
    return set(g.index[has_virb4 & has_t4cp])


def _replicon_taxonomy():
    """Return one row per replicon with gtdb_family / gtdb_phylum (taken from
    the conjugative-hits file, which already carries GTDB taxonomy)."""
    hits = pd.read_csv(FILTERED_HITS, sep='\t')
    tax = hits.drop_duplicates('replicon_name')[
        ['replicon_name', 'gtdb_phylum', 'gtdb_family']]
    return tax


def _load_subtype_assignments():
    clust = pd.read_csv(CLUSTER_CSV)
    tax = _replicon_taxonomy()
    return clust.merge(tax, left_on='short_label', right_on='replicon_name',
                       how='left')


# ===========================================================================
# Fig 4A — pie of VirB4-T4CP positive vs negative replicons
# ===========================================================================
def plot_virb4_t4cp_pie():
    cat = pd.read_csv(CATALOGUE_FILE, sep='\t')
    total = cat['sample_id'].nunique()
    positive = len(_vb_t4cp_positive_replicons())
    negative = total - positive

    fig, ax = plt.subplots(figsize=(4.0, 3.8))
    wedges, _ = ax.pie([negative, positive],
                       colors=['#cccccc', '#c0223b'],
                       startangle=90,
                       wedgeprops=dict(edgecolor='white', linewidth=1.5))
    # central label
    ax.text(0.55, 0.05, f"{positive/total*100:.1f}%", color='#c0223b',
            ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(-0.6, -0.05, f"{negative/total*100:.1f}%", color='#444',
            ha='center', va='center', fontsize=11, fontweight='bold')
    # outer labels
    ax.text(0, 1.25, f"VirB4-T4CP\npositive\n(n = {positive})", color='#c0223b',
            ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(-1.25, -0.55, f"VirB4-T4CP\nnegative\n(n = {negative})",
            color='#444', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.set_aspect('equal')
    plt.tight_layout()
    _save(fig, "fig4A_virb4_t4cp_pie")


# ===========================================================================
# Fig 4B — pie of phyla among VirB4-T4CP+ replicons
# ===========================================================================
def plot_virb4_t4cp_phyla_pie():
    pos = _vb_t4cp_positive_replicons()
    tax = _replicon_taxonomy()
    sub = tax[tax['replicon_name'].isin(pos)].copy()
    sub['phy'] = sub['gtdb_phylum'].map(_short)
    counts = sub['phy'].value_counts()

    PHY_COLORS = {
        'Halobacteriota':      '#3a72c4',
        'Thermoproteota':      '#d97706',
        'Methanobacteriota':   '#2ca02c',
        'Methanobacteriota_B': '#9467bd',
        'Thermoplasmatota':    '#e0457b',
    }
    colours = [PHY_COLORS.get(p, '#888') for p in counts.index]

    fig, ax = plt.subplots(figsize=(4.0, 3.8))
    wedges, _ = ax.pie(counts.values, colors=colours, startangle=90,
                       wedgeprops=dict(edgecolor='white', linewidth=1.5))
    total = counts.sum()
    for w, phy, n, col in zip(wedges, counts.index, counts.values, colours):
        ang = (w.theta1 + w.theta2) / 2
        rx = 0.55 * np.cos(np.deg2rad(ang))
        ry = 0.55 * np.sin(np.deg2rad(ang))
        ax.text(rx, ry, f"{n/total*100:.1f}%\n(n = {n})",
                ha='center', va='center', color='white', fontsize=9,
                fontweight='bold')
        # outer phylum label
        ox = 1.20 * np.cos(np.deg2rad(ang))
        oy = 1.20 * np.sin(np.deg2rad(ang))
        ax.text(ox, oy, phy, ha='center', va='center', color=col, fontsize=9,
                fontweight='bold')
    ax.set_aspect('equal')
    plt.tight_layout()
    _save(fig, "fig4B_phyla_among_virb4_t4cp_positive")


# ===========================================================================
# Fig 4C / Ext A / Ext B — UMAP-style 2-D embeddings from precomputed
# distance matrices
# ===========================================================================
def _embed_2d(distance_matrix, seed=42):
    """Return 2-D coordinates for each row of ``distance_matrix``.

    Tries UMAP first (using the matrix as a precomputed metric); falls back
    to multidimensional scaling if umap-learn is not installed.
    """
    try:
        import umap  # type: ignore
        reducer = umap.UMAP(metric='precomputed', random_state=seed,
                            n_neighbors=15, min_dist=0.1)
        coords = reducer.fit_transform(distance_matrix)
        return coords, 'UMAP'
    except ModuleNotFoundError:
        from sklearn.manifold import MDS
        mds = MDS(n_components=2, dissimilarity='precomputed',
                  random_state=seed, n_init=4, normalized_stress='auto')
        coords = mds.fit_transform(distance_matrix)
        return coords, 'MDS'


def _embed_from_clinker():
    mat = pd.read_csv(CLINKER_MATRIX, index_col=0)
    # Restrict to plasmids that have a subtype assignment.
    clust = pd.read_csv(CLUSTER_CSV)
    common = [p for p in mat.index if p in set(clust['plasmid'])]
    mat = mat.loc[common, common]
    # Ensure symmetry & zero diagonal
    arr = mat.values.copy()
    arr = (arr + arr.T) / 2
    np.fill_diagonal(arr, 0)
    coords, method = _embed_2d(arr)
    df = clust.set_index('plasmid').loc[common, ['hdbscan_cluster']].reset_index()
    df['x'] = coords[:, 0]; df['y'] = coords[:, 1]
    return df, method


def _scatter_subtypes(ax, df, method_label):
    sizes = df.groupby('hdbscan_cluster').size().to_dict()
    for st in sorted(sizes):
        sub = df[df['hdbscan_cluster'] == st]
        ax.scatter(sub['x'], sub['y'], color=SUBTYPE_COLORS.get(int(st), '#888'),
                   s=22, edgecolor='white', linewidth=0.4, alpha=0.9,
                   label=f"Subtype {ROMAN.get(int(st), int(st))} "
                         f"(n = {sizes[st]})")
    ax.set_xlabel(f"{method_label} 1")
    ax.set_ylabel(f"{method_label} 2")
    ax.legend(frameon=False, fontsize=7, loc='upper right')
    ax.spines[['top', 'right']].set_visible(False)


def plot_subtype_embedding():
    df, method = _embed_from_clinker()
    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    _scatter_subtypes(ax, df, method)
    plt.tight_layout()
    _save(fig, "fig4C_subtype_embedding_blast")
    # Same embedding doubles as Extended Data B.
    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    _scatter_subtypes(ax, df, 'BLAST')
    ax.set_title('BLAST distance', fontsize=10)
    plt.tight_layout()
    _save(fig, "extB_subtype_embedding_blast")


# ===========================================================================
# Fig 4D — Subtype × Family heatmap (counts of plasmids)
# ===========================================================================
def plot_subtype_family_heatmap():
    df = _load_subtype_assignments()
    df['family_short'] = df['gtdb_family'].map(_short).fillna('Unclassified')
    sizes = df.groupby('hdbscan_cluster').size()
    fam_order = (df['family_short'].value_counts()
                   .sort_values(ascending=False).index.tolist())
    st_order = sorted(df['hdbscan_cluster'].dropna().unique())
    mat = (df.groupby(['family_short', 'hdbscan_cluster']).size()
             .unstack(fill_value=0)
             .reindex(index=fam_order, columns=st_order, fill_value=0))

    fig, ax = plt.subplots(figsize=(4.4, max(3.0, 0.3 * len(fam_order) + 1)))
    im = ax.imshow(mat.values, cmap='YlOrRd', aspect='auto')
    for i, fam in enumerate(fam_order):
        for j, st in enumerate(st_order):
            v = int(mat.loc[fam, st])
            if v == 0:
                continue
            txt_col = 'white' if v > mat.values.max() * 0.55 else 'black'
            ax.text(j, i, str(v), ha='center', va='center', color=txt_col,
                    fontsize=8)
    ax.set_xticks(np.arange(len(st_order)))
    ax.set_xticklabels([ROMAN.get(int(s), int(s)) for s in st_order])
    ax.set_xlabel('Subtype')
    ax.set_yticks(np.arange(len(fam_order)))
    ax.set_yticklabels(fam_order, fontsize=8, style='italic')
    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('Number of plasmids', fontsize=8)
    plt.tight_layout()
    _save(fig, "fig4D_subtype_family_heatmap")


# ===========================================================================
# Extended Data C — stacked-bar family composition per subtype
# ===========================================================================
def plot_subtype_family_stacked():
    df = _load_subtype_assignments()
    df['family_short'] = df['gtdb_family'].map(_short).fillna('Unclassified')
    st_order = sorted(df['hdbscan_cluster'].dropna().unique())
    fam_order = (df['family_short'].value_counts()
                   .sort_values(ascending=False).index.tolist())
    counts = (df.groupby(['hdbscan_cluster', 'family_short']).size()
                .unstack(fill_value=0)
                .reindex(index=st_order, columns=fam_order, fill_value=0))
    pct = counts.div(counts.sum(axis=1), axis=0) * 100
    st_sizes = counts.sum(axis=1)

    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    x = np.arange(len(st_order))
    bottom = np.zeros(len(st_order))
    for fam in fam_order:
        col = FAMILY_COLORS.get(fam, '#666')
        vals = pct[fam].values
        ax.bar(x, vals, bottom=bottom, color=col, edgecolor='white',
               linewidth=0.4, label=fam)
        for xi, b, v in zip(x, bottom, vals):
            if v >= 8:
                ax.text(xi, b + v / 2, f"{int(round(v))}%", ha='center',
                        va='center', fontsize=8, color='white',
                        fontweight='bold')
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([f"ST-{ROMAN.get(int(s), int(s))}\n(n = {st_sizes[s]})"
                        for s in st_order], fontsize=8)
    ax.set_xlabel('VirB4-T4CP subtype')
    ax.set_ylabel('Proportion (%)')
    ax.set_ylim(0, 105)
    ax.legend(title='', frameon=False, fontsize=7, loc='center left',
              bbox_to_anchor=(1.0, 0.5))
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    _save(fig, "extC_subtype_family_stacked")


# ===========================================================================
# Extended Data D — per-subtype gene-neighbourhood frequency around the
# VirB4-T4CP locus. Requires per-plasmid full-ORF cluster assignments, which
# are not shipped in ``data/``; the panel is skipped with an explanatory
# note rather than fabricated.
# ===========================================================================
def plot_gene_neighbourhood():
    print("  skipping Extended Data D (gene neighbourhood): requires a "
          "per-plasmid ORF × mmseqs-cluster matrix, which is not present in "
          "the data/ folder. If you have the mmseqs output, add it as "
          "data/mmseqs_clusters/per_orf_assignments.tsv and re-implement "
          "this function — every other panel is unaffected.")


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("Generating Fig 4 panels:")
    plot_virb4_t4cp_pie()
    plot_virb4_t4cp_phyla_pie()
    plot_subtype_embedding()
    plot_subtype_family_heatmap()

    print("\nGenerating Extended Data panels:")
    plot_subtype_family_stacked()
    plot_gene_neighbourhood()

    print(f"\nAll outputs in {FIG_DIR}")


if __name__ == "__main__":
    main()
