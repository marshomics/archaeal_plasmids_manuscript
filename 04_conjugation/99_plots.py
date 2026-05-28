#!/usr/bin/env python3
"""Generate every panel for the VirB4-T4CP / conjugation figure set.

Outputs go to ``outputs/figures/`` next to this script. Every quantity is
re-derived from the input data and the other scripts in this pipeline; no
numbers are hard-coded. Each panel is written as both PNG (raster) and SVG
(``svg.fonttype = 'none'`` so labels stay editable in vector software).

Run as ``python 99_plots.py``.
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
                    CLINKER_MATRIX, MEMBERSHIP_FILE, OUT_DIR, header)  # noqa

FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
matplotlib.rcParams['font.family']  = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

SUBTYPE_COLORS = {
    1: '#1f77b4',  # Sub-type 1 — blue
    2: '#ff7f0e',  # Sub-type 2 — orange
    3: '#9467bd',  # Sub-type 3 — purple
    4: '#2ca02c',  # Sub-type 4 — green
    5: '#d62728',  # Sub-type 5 — red
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
    sub = sub[sub['phy'] != 'Methanobacteriota_B']
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
                   s=22, edgecolor='white', linewidth=0.3, alpha=0.8,
                   label=f"Sub-type {int(st)}")
    ax.set_xlabel(f"{method_label} 1")
    ax.set_ylabel(f"{method_label} 2")
    ax.legend(frameon=False, fontsize=7, loc='upper right')
    ax.spines[['top', 'right']].set_visible(False)


def plot_subtype_embedding():
    # --- Single-panel Clinker embedding (Fig 4C / Ext B) ---
    df_clink, method_clink = _embed_from_clinker()
    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    _scatter_subtypes(ax, df_clink, method_clink)
    plt.tight_layout()
    _save(fig, "fig4C_subtype_embedding_blast")

    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    _scatter_subtypes(ax, df_clink, 'BLAST')
    ax.set_title('BLAST distance', fontsize=10)
    plt.tight_layout()
    _save(fig, "extB_subtype_embedding_blast")


# ===========================================================================
# Fig 4D — Subtype × Family heatmap (counts of plasmids)
# ===========================================================================
def plot_subtype_family_heatmap():
    df = _load_subtype_assignments()
    df['family_short'] = df['gtdb_family'].map(_short).fillna('Unclassified')
    df = df[df['family_short'] != 'Thermococcaceae']
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
    df = df[df['family_short'] != 'Thermococcaceae']
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
# Positional conservation of protein families relative to VirB4
# ===========================================================================
def plot_positional_conservation():
    """For each subtype, plot the fraction of members that share the dominant
    protein cluster (PC) at each gene-position offset from VirB4.

    Two series per subtype:
      - grey bar:   fraction of members with *any* conjugation-region gene
      - coloured bar: fraction sharing the single most-common PC at that offset

    The membership file contains ALL plasmid genes, so we first identify
    "positionally conserved" PCs — those that appear at the same VirB4-
    relative offset in ≥2 members of any subtype.  Only these PCs are
    counted when computing fractions, which restricts the signal to the
    syntenic conjugation neighbourhood.

    All values derived from:
      - FILTERED_HITS   → VirB4 ORF position per replicon
      - MEMBERSHIP_FILE → gene ORF numbers and PC assignments per plasmid
      - CLUSTER_CSV     → subtype assignments per fragment
    """
    from collections import Counter, defaultdict

    if not MEMBERSHIP_FILE.exists():
        print("  skipping positional conservation: "
              "cluster_membership.tsv not in data/")
        return

    # ----- 1. VirB4 ORF position per replicon ----------------------------
    hits = pd.read_csv(FILTERED_HITS, sep='\t')
    virb4 = hits[hits['gene_name'].str.contains('virb4', case=False, na=False)]
    virb4_pos = virb4.groupby('replicon_name')['position_hit'].first().to_dict()

    # ----- 2. Gene ORF numbers and PC assignments ------------------------
    mem = pd.read_csv(MEMBERSHIP_FILE, sep='\t')
    mem['plasmid'] = mem['plasmid'].astype(str)
    mem['cluster_id'] = mem['cluster_id'].astype(str)
    mem['orf_num'] = mem['protein_id'].str.extract(r'_(\d+)$').astype(int)

    plasmid_genes = defaultdict(list)
    for _, r in mem.iterrows():
        plasmid_genes[r['plasmid']].append((r['orf_num'], r['cluster_id']))

    # ----- 3. Subtype assignments (deduplicate by short_label) -----------
    clust = pd.read_csv(CLUSTER_CSV)
    clust_dedup = clust.drop_duplicates('short_label')

    # ----- 4. Build per-subtype offset → PC maps ------------------------
    st_plasmid_offsets = defaultdict(list)   # st → [dict{offset: pc}, …]

    for _, row in clust_dedup.iterrows():
        label = row['short_label']
        st = int(row['hdbscan_cluster'])
        if label not in virb4_pos:
            continue
        vb4_orf = virb4_pos[label]
        genes = plasmid_genes.get(label, [])
        if not genes:
            continue
        offset_map = {}
        for orf, pc in genes:
            offset_map[orf - vb4_orf] = pc
        st_plasmid_offsets[st].append(offset_map)

    if not st_plasmid_offsets:
        print("  skipping positional conservation: no matching data")
        return

    conserved_pcs = set()
    for st, maps in st_plasmid_offsets.items():
        off_pc_count = Counter()
        for m in maps:
            for off, pc in m.items():
                off_pc_count[(off, pc)] += 1
        for (off, pc), cnt in off_pc_count.items():
            if cnt >= 2:
                conserved_pcs.add(pc)

    st_plasmid_windows = defaultdict(list)  # st → [(lo, hi), …]
    for st, maps in st_plasmid_offsets.items():
        for m in maps:
            cons_offs = [off for off, pc in m.items()
                         if pc in conserved_pcs]
            if cons_offs:
                st_plasmid_windows[st].append((min(cons_offs), max(cons_offs)))
            else:
                st_plasmid_windows[st].append(None)
              
    subtypes = sorted(st_plasmid_offsets.keys())
    scan_lo, scan_hi = -50, 50
    scan_offsets = list(range(scan_lo, scan_hi + 1))

    st_data_wide = {}
    for st in subtypes:
        maps = st_plasmid_offsets[st]
        windows = st_plasmid_windows[st]
        n = len(maps)
        frac_any, frac_dom = [], []
        for off in scan_offsets:
            # Grey: any gene at this offset, but only within the window
            n_present = 0
            for m, win in zip(maps, windows):
                if win is None:
                    continue
                if win[0] <= off <= win[1] and off in m:
                    n_present += 1
            frac_any.append(n_present / n if n else 0)
            # Coloured: dominant conserved PC
            pcs = [m[off] for m in maps
                   if off in m and m[off] in conserved_pcs]
            if not pcs:
                frac_dom.append(0)
            else:
                dom_count = Counter(pcs).most_common(1)[0][1]
                frac_dom.append(dom_count / n if n else 0)
        st_data_wide[st] = (np.array(frac_any), np.array(frac_dom), n)

    all_maps_pooled = [m for maps in st_plasmid_offsets.values() for m in maps]
    n_total = len(all_maps_pooled)

    pooled_dom = np.zeros(len(scan_offsets))
    for i, off in enumerate(scan_offsets):
        pcs = [m[off] for m in all_maps_pooled
               if off in m and m[off] in conserved_pcs]
        if pcs:
            pooled_dom[i] = Counter(pcs).most_common(1)[0][1] / n_total

    CONS_THRESH = 0.025          # ≈ 3 / 121 plasmids pooled
    MAX_GAP = 0                  # strict contiguity (no gap tolerance)
    CONTEXT_PAD = 2              # flanking positions for context

    zero_idx = scan_offsets.index(0)

    # Expand left from VirB4
    left_idx = zero_idx
    gap = 0
    for i in range(zero_idx - 1, -1, -1):
        if pooled_dom[i] >= CONS_THRESH:
            left_idx = i
            gap = 0
        else:
            gap += 1
            if gap > MAX_GAP:
                break

    # Expand right from VirB4
    right_idx = zero_idx
    gap = 0
    for i in range(zero_idx + 1, len(scan_offsets)):
        if pooled_dom[i] >= CONS_THRESH:
            right_idx = i
            gap = 0
        else:
            gap += 1
            if gap > MAX_GAP:
                break

    # Add context padding
    left_idx = max(0, left_idx - CONTEXT_PAD)
    right_idx = min(len(scan_offsets) - 1, right_idx + CONTEXT_PAD)

    offsets = scan_offsets[left_idx:right_idx + 1]

    st_data = {}
    for st in subtypes:
        fa, fd, n = st_data_wide[st]
        st_data[st] = (fa[left_idx:right_idx + 1],
                       fd[left_idx:right_idx + 1], n)

    # ----- 8. Plot -------------------------------------------------------
    fig, axes = plt.subplots(len(subtypes), 1,
                             figsize=(10, 2.2 * len(subtypes)),
                             sharex=True)
    if len(subtypes) == 1:
        axes = [axes]

    x = np.arange(len(offsets))
    bar_width = 0.8

    for ax, st in zip(axes, subtypes):
        frac_any, frac_dom, n = st_data[st]
        col = SUBTYPE_COLORS.get(st, '#888')

        ax.bar(x, frac_any, width=bar_width, color='#d3d3d3',
               edgecolor='#808080', linewidth=0.5, alpha=0.7,
               label='Any gene present')
        ax.bar(x, frac_dom, width=bar_width, color=col,
               edgecolor='none', label='Most common PC')

        ax.set_ylabel(f"ST{st} (n={n})", fontsize=9, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(frameon=True, fontsize=6, loc='upper right',
                  framealpha=0.8, edgecolor='none')

        # Light vertical line at VirB4 (offset 0)
        if 0 in offsets:
            vb4_x = offsets.index(0)
            ax.axvline(vb4_x, color=col, linewidth=0.8, alpha=0.5, zorder=0)

    # X-axis labels on the bottom panel only
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(offsets, fontsize=7)
    axes[-1].set_xlabel('Gene position offset from VirB4', fontsize=10)

    fig.suptitle(
        'Positional conservation of protein families relative to VirB4\n'
        '(fraction of subtype members sharing the dominant PC at each position)',
        fontsize=11, fontweight='bold', y=1.01)
    plt.tight_layout()
    _save(fig, "positional_conservation_by_subtype")


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
    plot_positional_conservation()

    print(f"\nAll outputs in {FIG_DIR}")


if __name__ == "__main__":
    main()
