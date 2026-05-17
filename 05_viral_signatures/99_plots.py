#!/usr/bin/env python3
"""Generate every panel for the viral-signature figure set.

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
from common import load_data, OUT_DIR  # noqa


FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
matplotlib.rcParams['font.family']  = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']


# Category display order and colours match the manuscript palette.
CAT_ORDER = ['Integrase', 'Phage replication', 'Lysis & membrane',
             'Metabolic enzymes', 'Osmolyte transport',
             'Prophage maintenance', 'Tail & baseplate',
             'DNA packaging', 'Capsid & head']
CAT_COLOR = {
    'Integrase':            '#1f78b4',
    'Phage replication':    '#e0457b',
    'Lysis & membrane':     '#2ca02c',
    'Metabolic enzymes':    '#f9a13b',
    'Osmolyte transport':   '#3fa297',
    'Prophage maintenance': '#d04141',
    'Tail & baseplate':     '#9467bd',
    'DNA packaging':        '#f4d35e',
    'Capsid & head':        '#e8807a',
}

PHYLUM_COLOR = {
    'p__Halobacteriota':      '#d97070',
    'p__Thermoproteota':      '#1f78b4',
    'p__Methanobacteriota':   '#2ca02c',
    'p__Methanobacteriota_B': '#f4a07a',
    'p__Thermoplasmatota':    '#9467bd',
}


def _short(s):
    return s.split('__', 1)[-1] if isinstance(s, str) and '__' in s else s


def _save(fig, stem):
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=200, bbox_inches='tight')
    fig.savefig(FIG_DIR / f"{stem}.svg", bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {stem}.png + .svg")


# ===========================================================================
# Fig 5A — pie of plasmids with vs without viral proteins
# ===========================================================================
def plot_viral_prevalence_pie(cls, mob):
    total = mob['sample_id'].nunique()
    with_viral = cls['replicon'].nunique()
    without    = total - with_viral

    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    wedges, _ = ax.pie([with_viral, without],
                       colors=['#1f78b4', '#cccccc'],
                       startangle=90,
                       wedgeprops=dict(edgecolor='white', linewidth=1.5))
    # outside labels
    labels = [f"With viral proteins\n({with_viral}; {with_viral/total*100:.1f}%)",
              f"Without viral proteins\n({without}; {without/total*100:.1f}%)"]
    for w, lab in zip(wedges, labels):
        ang = (w.theta1 + w.theta2) / 2
        x = 1.25 * np.cos(np.deg2rad(ang))
        y = 1.25 * np.sin(np.deg2rad(ang))
        ax.text(x, y, lab, ha='center', va='center', fontsize=9,
                color='#333')
    ax.set_aspect('equal')
    plt.tight_layout()
    _save(fig, "fig5A_viral_prevalence_pie")


# ===========================================================================
# Fig 5B — per-category prevalence horizontal bar
# ===========================================================================
def plot_category_prevalence(cls):
    n_viral = cls['replicon'].nunique()
    counts = (cls.drop_duplicates(['replicon', 'new_category'])
                 .groupby('new_category')['replicon'].nunique())
    counts = counts.reindex(CAT_ORDER, fill_value=0)
    pct = 100 * counts / n_viral

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    y = np.arange(len(counts))[::-1]
    colors = [CAT_COLOR[c] for c in counts.index]
    ax.barh(y, counts.values, color=colors, edgecolor='white', linewidth=0.4)
    for yi, c, p in zip(y, counts.values, pct.values):
        ax.text(c + counts.max() * 0.012, yi, f"{int(c)} ({p:.1f}%)",
                va='center', fontsize=8, color='#222')
    ax.set_yticks(y)
    ax.set_yticklabels(counts.index, fontsize=9)
    ax.set_xlabel('Number of plasmids')
    ax.set_xlim(0, counts.max() * 1.18)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    _save(fig, "fig5B_category_prevalence")


# ===========================================================================
# Fig 5C — per-family bubble plot of categories per plasmid
# Each bubble is a count of plasmids in that family with that many viral
# categories. Empty (0-category) bubbles are hollow.
# ===========================================================================
def plot_per_family_bubbles(cls, mob):
    # Build per-plasmid category count
    by_plasmid = (cls.drop_duplicates(['replicon', 'new_category'])
                     .groupby('replicon')['new_category'].nunique())
    family = mob.set_index('sample_id')['gtdb_family']
    phylum = mob.set_index('sample_id')['gtdb_phylum']
    df = (pd.DataFrame({'n_cat': by_plasmid})
            .reindex(mob['sample_id'].unique(), fill_value=0)
            .assign(family=lambda d: family.reindex(d.index),
                    phylum=lambda d: phylum.reindex(d.index)))
    df['family_short'] = df['family'].map(_short).fillna('Unclassified')

    # Family rows: order by phylum then by descending total plasmid count
    fam_summary = (df.groupby(['family', 'phylum']).agg(
        n=('n_cat', 'size'),
        n_with=('n_cat', lambda s: (s > 0).sum()),
        n_no=('n_cat', lambda s: (s == 0).sum())).reset_index())
    # Drop tiny families to keep panel readable
    fam_summary = fam_summary[fam_summary['n'] >= 2].copy()
    PHYLUM_ORDER = ['p__Halobacteriota', 'p__Methanobacteriota',
                    'p__Methanobacteriota_B', 'p__Thermoproteota',
                    'p__Thermoplasmatota']
    fam_summary['phy_rank'] = fam_summary['phylum'].map(
        {p: i for i, p in enumerate(PHYLUM_ORDER)}).fillna(99)
    fam_summary = fam_summary.sort_values(['phy_rank', 'n'],
                                          ascending=[True, False])
    fam_order = fam_summary['family'].tolist()

    max_cat = int(df['n_cat'].max())
    x_positions = np.arange(0, max_cat + 1)

    # Compute bubble matrix: (family, n_cat) → count
    sub = df[df['family'].isin(fam_order)]
    bubble = (sub.groupby(['family', 'n_cat']).size()
                 .unstack(fill_value=0)
                 .reindex(index=fam_order,
                          columns=range(max_cat + 1), fill_value=0))

    fig, ax = plt.subplots(figsize=(8.5, max(4.5, 0.4 * len(fam_order))))
    y_positions = np.arange(len(fam_order))[::-1]
    BUBBLE_BASE = 30
    BUBBLE_SCALE = 80
    for yi, fam in zip(y_positions, fam_order):
        phy = fam_summary.set_index('family').loc[fam, 'phylum']
        col = PHYLUM_COLOR.get(phy, '#666')
        for x in x_positions:
            n = int(bubble.loc[fam, x])
            if n == 0:
                continue
            size = BUBBLE_BASE + BUBBLE_SCALE * np.sqrt(n)
            face = 'none' if x == 0 else col
            edge = col
            ax.scatter(x, yi, s=size, facecolor=face, edgecolor=edge,
                       linewidth=1.0, alpha=0.85)

    # Family labels with n and 'no viral'
    labels = []
    for fam in fam_order:
        row = fam_summary.set_index('family').loc[fam]
        label = f"{_short(fam)} (n = {int(row['n'])}"
        if row['n_no'] > 0:
            label += f"; {int(row['n_no'])} no viral"
        label += ')'
        labels.append(label)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=8)
    for tick, fam in zip(ax.get_yticklabels(), fam_order):
        phy = fam_summary.set_index('family').loc[fam, 'phylum']
        tick.set_color(PHYLUM_COLOR.get(phy, '#333'))

    # Right-side: "collective categories; max co-occurring" labels
    for yi, fam in zip(y_positions, fam_order):
        sub_fam = sub[sub['family'] == fam]
        ids = sub_fam.index.tolist()
        # collective categories: distinct categories across family's plasmids
        coll = (cls[cls['replicon'].isin(ids)]['new_category']
                .nunique())
        # max co-occurring on a single plasmid:
        maxc = int(sub_fam['n_cat'].max()) if len(sub_fam) else 0
        if coll == 0 and maxc == 0:
            continue
        ax.text(max_cat + 0.6, yi,
                f"{coll} collective; {maxc} max co-occurring",
                va='center', fontsize=7, color='#444')

    # Phylum-block separators (dashed horizontal lines between blocks)
    last_phy = None
    for yi, fam in zip(y_positions, fam_order):
        phy = fam_summary.set_index('family').loc[fam, 'phylum']
        if last_phy is not None and phy != last_phy:
            ax.axhline(yi + 0.5, color='gray', linestyle='--', linewidth=0.5,
                       alpha=0.5)
        last_phy = phy

    ax.set_xticks(x_positions)
    ax.set_xlabel('Number of viral protein categories per plasmid')
    ax.set_xlim(-0.5, max_cat + 4.2)
    ax.set_ylim(-0.7, max(y_positions) + 0.7)
    ax.spines[['top', 'right']].set_visible(False)

    # Phylum legend
    handles = [mpatches.Patch(color=PHYLUM_COLOR[p], label=_short(p))
               for p in PHYLUM_ORDER if p in PHYLUM_COLOR]
    handles.append(mpatches.Patch(edgecolor='gray', facecolor='none',
                                  label='No viral proteins'))
    leg = ax.legend(handles=handles, frameon=False, fontsize=7, title='Phylum',
                    loc='lower right', bbox_to_anchor=(1.0, -0.18), ncol=3)
    leg.get_title().set_fontsize(8)
    # Bubble-size legend
    sizes = [1, 5, 20, 50]
    sx = max_cat + 1.4
    sy0 = -0.5
    for i, s in enumerate(sizes):
        ax.scatter(sx, sy0 - i * 0.6, s=BUBBLE_BASE + BUBBLE_SCALE * np.sqrt(s),
                   facecolor='gray', alpha=0.5, edgecolor='gray',
                   linewidth=0.4)
        ax.text(sx + 0.25, sy0 - i * 0.6, str(s), va='center', fontsize=7)
    ax.text(sx, sy0 + 0.6, '# plasmids:', fontsize=7, color='#444')

    plt.tight_layout()
    _save(fig, "fig5C_per_family_bubbles")


# ===========================================================================
# Fig 5D — proportion of viral category annotations per phylum
# ===========================================================================
def plot_phylum_category_proportions(cls):
    cls = cls.copy()
    cls['phy'] = cls['gtdb_phylum'].map(_short)
    counts = cls.groupby(['phy', 'new_category']).size().unstack(fill_value=0)
    counts = counts.reindex(columns=CAT_ORDER, fill_value=0)
    PHYLUM_ORDER = ['Halobacteriota', 'Thermoproteota',
                    'Methanobacteriota', 'Methanobacteriota_B']
    counts = counts.reindex(index=[p for p in PHYLUM_ORDER if p in counts.index],
                            fill_value=0)
    pct = counts.div(counts.sum(axis=1).replace(0, 1), axis=0)

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    y = np.arange(len(counts))[::-1]
    left = np.zeros(len(counts))
    for cat in CAT_ORDER:
        vals = pct[cat].values
        ax.barh(y, vals, left=left, color=CAT_COLOR[cat],
                edgecolor='white', linewidth=0.4, label=cat)
        left += vals
    ax.set_yticks(y); ax.set_yticklabels(counts.index, fontsize=9, style='italic')
    ax.set_xlabel('Proportion of annotations')
    ax.set_xlim(0, 1)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, fontsize=7, loc='center left',
              bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout()
    _save(fig, "fig5D_phylum_category_proportions")


# ===========================================================================
# Fig 5E — per-phylum prevalence stacked bars (with/without viral proteins)
# ===========================================================================
def plot_per_phylum_prevalence(cls, mob):
    viral_set = set(cls['replicon'].unique())
    df = (mob[['sample_id', 'gtdb_phylum']].drop_duplicates()
            .assign(has_viral=lambda d: d['sample_id'].isin(viral_set)))
    df['phy'] = df['gtdb_phylum'].map(_short)
    grp = (df.groupby('phy').agg(n=('sample_id', 'count'),
                                  n_viral=('has_viral', 'sum'))
             .assign(pct=lambda d: 100 * d['n_viral'] / d['n']))
    grp = grp.sort_values('n', ascending=False)

    fig, ax = plt.subplots(figsize=(5.8, 4))
    x = np.arange(len(grp))
    no_viral = grp['n'] - grp['n_viral']
    PHY_BLUE = '#3a72c4'
    ax.bar(x, grp['n_viral'], color=PHY_BLUE, edgecolor='white', linewidth=0.4,
           label='With viral proteins')
    ax.bar(x, no_viral, bottom=grp['n_viral'], color='#cccccc',
           edgecolor='white', linewidth=0.4, label='Without viral proteins')
    for xi, total, pct in zip(x, grp['n'], grp['pct']):
        ax.text(xi, total + 2, f"{pct:.1f}%", ha='center', va='bottom',
                fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(grp.index, rotation=30, ha='right', fontsize=8,
                       style='italic')
    ax.set_ylabel('Number of plasmids')
    ax.set_ylim(0, grp['n'].max() * 1.15)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc='upper right')
    plt.tight_layout()
    _save(fig, "fig5E_per_phylum_prevalence")


# ===========================================================================
# Extended Data A — pairwise co-occurrence heatmap (lower triangle)
# ===========================================================================
def plot_pairwise_cooccurrence(cls):
    pivot = (cls.drop_duplicates(['replicon', 'new_category'])
                .pivot_table(index='replicon', columns='new_category',
                             values='protein', aggfunc='count', fill_value=0))
    pivot = (pivot > 0).astype(int)
    pivot = pivot.reindex(columns=CAT_ORDER, fill_value=0)
    co = pivot.T.dot(pivot)
    # Lower-triangular mask
    mask = np.tril(np.ones_like(co.values, dtype=bool))
    arr = co.values.astype(float)
    arr[~mask] = np.nan

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(arr, cmap='YlOrRd', aspect='auto')
    for i, c in enumerate(CAT_ORDER):
        for j, c2 in enumerate(CAT_ORDER):
            if not mask[i, j]:
                continue
            v = int(co.values[i, j])
            if v == 0:
                continue
            txt = 'white' if v > co.values.max() * 0.6 else 'black'
            ax.text(j, i, str(v), ha='center', va='center', color=txt,
                    fontsize=7)
    ax.set_xticks(np.arange(len(CAT_ORDER)))
    ax.set_xticklabels(CAT_ORDER, rotation=80, ha='right', fontsize=8)
    ax.set_yticks(np.arange(len(CAT_ORDER)))
    ax.set_yticklabels(CAT_ORDER, fontsize=8)
    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('Number of plasmids', fontsize=8)
    plt.tight_layout()
    _save(fig, "extA_pairwise_cooccurrence")


# ===========================================================================
# Extended Data B — proportion of unique proteins per category, stratified by
# number of viral categories per plasmid
# ===========================================================================
def plot_category_count_distribution(cls, mob):
    pivot = (cls.drop_duplicates(['replicon', 'new_category'])
                .pivot_table(index='replicon', columns='new_category',
                             values='protein', aggfunc='count', fill_value=0))
    pivot = (pivot > 0).astype(int)
    pivot = pivot.reindex(columns=CAT_ORDER, fill_value=0)
    n_cat_per_plasmid = pivot.sum(axis=1)
    # All plasmids (including 0)
    all_ids = mob['sample_id'].unique()
    n_cat_full = pd.Series(0, index=all_ids)
    n_cat_full.update(n_cat_per_plasmid)
    counts_by_n = n_cat_full.value_counts().sort_index()
    bin_order = sorted(counts_by_n.index.tolist())

    # For each bin, fraction of plasmids carrying each category
    rows = []
    for n in bin_order:
        plas_ids = n_cat_full[n_cat_full == n].index
        if n == 0:
            rows.append({'n_cat': n, 'n_plasmids': len(plas_ids),
                         **{c: 0 for c in CAT_ORDER}})
            continue
        sub = pivot.loc[pivot.index.isin(plas_ids)]
        # Fraction of plasmids in this bin that carry each category
        frac = sub.sum(axis=0) / len(plas_ids)
        # normalise so categories sum to 1 within each bin (stacked = 1)
        frac = frac / frac.sum() if frac.sum() > 0 else frac
        rows.append({'n_cat': n, 'n_plasmids': len(plas_ids),
                     **{c: frac.get(c, 0) for c in CAT_ORDER}})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    x = np.arange(len(df))
    bottom = np.zeros(len(df))
    for cat in CAT_ORDER:
        vals = df[cat].values
        ax.bar(x, vals, bottom=bottom, color=CAT_COLOR[cat],
               edgecolor='white', linewidth=0.4, label=cat)
        bottom += vals
    for xi, n_p in zip(x, df['n_plasmids']):
        ax.text(xi, 1.02, f"n = {n_p}", ha='center', va='bottom',
                fontsize=7, color='#444')
    ax.set_xticks(x); ax.set_xticklabels(df['n_cat'].astype(int))
    ax.set_xlabel('Number of viral categories per plasmid')
    ax.set_ylabel('Proportion of unique proteins')
    ax.set_ylim(0, 1.10)
    ax.legend(frameon=False, fontsize=7, loc='center left',
              bbox_to_anchor=(1.0, 0.5))
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    _save(fig, "extB_category_count_distribution")


# ===========================================================================
# Extended Data C — presence/absence Phylum × Category matrix
# ===========================================================================
def plot_phylum_category_presence(cls):
    cls = cls.copy()
    cls['phy'] = cls['gtdb_phylum'].map(_short)
    pa = (cls.groupby(['phy', 'new_category']).size()
             .unstack(fill_value=0))
    pa = (pa > 0).astype(int)
    pa = pa.reindex(columns=CAT_ORDER, fill_value=0)
    PHYLUM_ORDER = ['Halobacteriota', 'Thermoproteota',
                    'Methanobacteriota', 'Methanobacteriota_B']
    pa = pa.reindex(index=[p for p in PHYLUM_ORDER if p in pa.index])

    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    ax.imshow(np.where(pa.values == 1, 1.0, 0.7),
              cmap='Blues', aspect='auto', vmin=0, vmax=1.2)
    for i, phy in enumerate(pa.index):
        for j, cat in enumerate(CAT_ORDER):
            ax.scatter(j, i, s=80, color='white' if pa.iloc[i, j] else 'lightgray',
                       edgecolor='gray', linewidth=0.5)
    ax.set_xticks(np.arange(len(CAT_ORDER)))
    ax.set_xticklabels(CAT_ORDER, rotation=80, ha='right', fontsize=8)
    ax.set_yticks(np.arange(len(pa)))
    ax.set_yticklabels(pa.index, fontsize=9, style='italic')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    _save(fig, "extC_phylum_category_presence")


# ===========================================================================
# Extended Data D — per-family stacked bars of viral category proportions
# ===========================================================================
def plot_per_family_category_proportions(cls, mob):
    cls = cls.copy()
    cls['fam_short'] = cls['gtdb_family'].map(_short).fillna('Unclassified')
    counts = cls.groupby(['fam_short', 'new_category']).size().unstack(fill_value=0)
    counts = counts.reindex(columns=CAT_ORDER, fill_value=0)
    counts['total'] = counts.sum(axis=1)
    counts = counts.sort_values('total', ascending=False)
    fam_order = counts.head(12).index.tolist()
    counts = counts.loc[fam_order, CAT_ORDER]
    pct = counts.div(counts.sum(axis=1).replace(0, 1), axis=0)

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    x = np.arange(len(fam_order))
    bottom = np.zeros(len(fam_order))
    for cat in CAT_ORDER:
        vals = pct[cat].values
        ax.bar(x, vals, bottom=bottom, color=CAT_COLOR[cat],
               edgecolor='white', linewidth=0.4, label=cat)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(fam_order, rotation=35, ha='right', fontsize=8,
                       style='italic')
    ax.set_ylabel('Proportion of annotations')
    ax.set_ylim(0, 1.02)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, fontsize=7, loc='center left',
              bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout()
    _save(fig, "extD_per_family_category_proportions")


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("Loading data...")
    cls, mob, clusters, conj, complexity, full_complexity = load_data()
    print(f"  {mob['sample_id'].nunique()} plasmids; "
          f"{cls['replicon'].nunique()} with viral proteins")

    print("\nGenerating Fig 5 panels:")
    plot_viral_prevalence_pie(cls, mob)
    plot_category_prevalence(cls)
    plot_per_family_bubbles(cls, mob)
    plot_phylum_category_proportions(cls)
    plot_per_phylum_prevalence(cls, mob)

    print("\nGenerating Extended Data panels:")
    plot_pairwise_cooccurrence(cls)
    plot_category_count_distribution(cls, mob)
    plot_phylum_category_presence(cls)
    plot_per_family_category_proportions(cls, mob)

    print(f"\nAll outputs in {FIG_DIR}")


if __name__ == "__main__":
    main()
