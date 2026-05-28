#!/usr/bin/env python3
"""Generate every panel for the prevalence-and-abundance figure set.

Outputs go to ``outputs/figures/`` next to this script. Every quantity is
re-derived from the input data and the other scripts in this pipeline; no
numbers are hard-coded. Each panel is written as both PNG (raster) and SVG
(with ``svg.fonttype = 'none'`` so text labels remain editable in vector
software).

Run as ``python 99_plots.py``.
"""
from collections import defaultdict
from pathlib import Path
import sys

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path as MplPath
import numpy as np
import pandas as pd
from scipy import stats

# Reuse the loaders and constants the rest of the pipeline depends on.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_data, load_model_species, PHYLUM_COLORS  # noqa


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
matplotlib.rcParams['font.family']  = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

CARRIER_PHYLA = list(PHYLUM_COLORS.keys())


def _short(name):
    """Strip the GTDB rank prefix (e.g. ``p__``) from a taxon label."""
    if not isinstance(name, str):
        return ''
    return name.split('__', 1)[-1] if '__' in name else name


def _save(fig, stem):
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=200, bbox_inches='tight')
    fig.savefig(OUT_DIR / f"{stem}.svg", bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {stem}.png + .svg")


# ---------------------------------------------------------------------------
# Helpers used by multiple panels
# ---------------------------------------------------------------------------
def _prepare(reps):
    """Add short-name taxonomy columns used throughout the figure set."""
    df = reps.copy()
    for rank in ['gtdb_phylum', 'gtdb_class', 'gtdb_order', 'gtdb_family',
                 'gtdb_genus', 'gtdb_species']:
        df[f"{rank}_short"] = df[rank].map(_short)
    return df


def _phylum_color(phy):
    return PHYLUM_COLORS.get(phy, '#888888')


# ---------------------------------------------------------------------------
# Panel: species- and phylum-level prevalence pies
# ---------------------------------------------------------------------------
def plot_prevalence_pies(reps):
    """Two donut/pie charts: species with/without plasmids; phyla with/without."""
    n_species = len(reps)
    n_carriers = int(reps['is_carrier'].sum())
    n_noncarriers = n_species - n_carriers

    phyla = reps['gtdb_phylum'].dropna().unique()
    n_phyla = len(phyla)
    n_carrier_phyla = int(reps.loc[reps['is_carrier'] == 1, 'gtdb_phylum']
                          .nunique())
    n_noncarrier_phyla = n_phyla - n_carrier_phyla

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.2))

    def _pie(ax, sizes, labels, colours, title, sub):
        wedges, _ = ax.pie(sizes, colors=colours, startangle=90,
                           wedgeprops=dict(edgecolor='white', linewidth=1.5))
        for i, (s, lab) in enumerate(zip(sizes, labels)):
            ax.text(0, 0, sub, ha='center', va='center', fontsize=9,
                    color='#333')
        # Outer labels
        total = sum(sizes)
        for w, s, lab in zip(wedges, sizes, labels):
            ang = (w.theta1 + w.theta2) / 2
            x = 1.15 * np.cos(np.deg2rad(ang))
            y = 1.15 * np.sin(np.deg2rad(ang))
            ax.text(x, y, f"{lab}\n{s}  ({s/total*100:.1f}%)",
                    ha='center', va='center', fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_aspect('equal')

    _pie(axes[0],
         [n_carriers, n_noncarriers],
         ['With plasmids', 'No plasmids'],
         ['#4daf4a', '#c8c8c8'],
         'Species', f"n = {n_species:,}")
    _pie(axes[1],
         [n_carrier_phyla, n_noncarrier_phyla],
         ['With plasmids', 'No plasmids'],
         ['#4daf4a', '#c8c8c8'],
         'Phyla', f"n = {n_phyla}")
    plt.tight_layout()
    _save(fig, "01_prevalence_pies")


# ---------------------------------------------------------------------------
# Panel: per-phylum carrier rate (bars, carrier phyla only)
# ---------------------------------------------------------------------------
def plot_per_phylum_rate(reps):
    df = (reps.groupby('gtdb_phylum')
              .agg(n_species=('accession', 'size'),
                   n_carriers=('is_carrier', 'sum'))
              .reset_index())
    df = df[df['n_carriers'] > 0].copy()
    df['pct'] = 100 * df['n_carriers'] / df['n_species']
    df = df.sort_values('pct', ascending=False)
    fig, ax = plt.subplots(figsize=(5.0, 0.45 * len(df) + 1.0))
    y = np.arange(len(df))[::-1]
    colours = [_phylum_color(p) for p in df['gtdb_phylum']]
    ax.barh(y, df['pct'], color=colours, edgecolor='white', linewidth=0.4)
    for yi, val, lab in zip(y, df['pct'], df['gtdb_phylum']):
        ax.text(val + max(df['pct']) * 0.01, yi, f"{val:.1f}%",
                va='center', fontsize=8, color='#333')
    ax.set_yticks(y)
    ax.set_yticklabels([_short(p) for p in df['gtdb_phylum']], fontsize=9)
    ax.set_xlabel('% species with plasmids')
    ax.set_xlim(0, max(df['pct']) * 1.18)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    _save(fig, "02_per_phylum_carrier_rate")


# ---------------------------------------------------------------------------
# Panel: taxonomy hierarchy of carrier species (Phylum → Class → Order →
# Family). Manual Sankey-style rendering in pure matplotlib so the SVG
# remains editable downstream.
# ---------------------------------------------------------------------------
def plot_taxonomy_hierarchy(reps):
    df = _prepare(reps)
    carriers = df[df['is_carrier'] == 1].copy()
    ranks = ['gtdb_phylum_short', 'gtdb_class_short',
             'gtdb_order_short', 'gtdb_family_short']
    rank_titles = ['Phylum', 'Class', 'Order', 'Family']

    # carrier counts per node (count carrier species, not plasmids)
    node_counts = {r: carriers.groupby(r).size().to_dict() for r in ranks}

    # ordering: by parent phylum total, descending
    phylum_totals = (carriers['gtdb_phylum_short'].value_counts()
                     .sort_values(ascending=False).index.tolist())

    # Build per-rank ordered node lists, grouped by parent
    def _ordered_nodes(rank_idx):
        parent_rank = ranks[rank_idx - 1] if rank_idx > 0 else None
        if parent_rank is None:
            return phylum_totals
        nodes = []
        for parent in _ordered_nodes(rank_idx - 1):
            sub = carriers[carriers[parent_rank] == parent]
            child_counts = sub[ranks[rank_idx]].value_counts().sort_values(
                ascending=False)
            for c in child_counts.index:
                nodes.append(c)
        return nodes

    rank_nodes = [_ordered_nodes(i) for i in range(len(ranks))]

    # Position layout
    fig_h = max(7.0, 0.18 * max(len(n) for n in rank_nodes))
    fig, ax = plt.subplots(figsize=(11, fig_h))

    GAP_BETWEEN_NODES = 0.6       # vertical gap (count units)
    COLUMN_X = [0, 4, 8, 12]
    COLUMN_W = 0.6
    PADDING = 1.0
    col_heights = []
    for nodes in rank_nodes:
        h = sum(node_counts_of(node, ranks, node_counts, rank_nodes.index(nodes))
                for node in nodes) + GAP_BETWEEN_NODES * (len(nodes) - 1)
        col_heights.append(h)
    canvas_h = max(col_heights) + 2 * PADDING

    def _layout_column(nodes, rank_idx):
        positions = {}
        # start at top
        y_cursor = canvas_h - PADDING
        for n in nodes:
            cnt = node_counts[ranks[rank_idx]].get(n, 0)
            top = y_cursor
            bottom = y_cursor - cnt
            positions[n] = (top, bottom, cnt)
            y_cursor = bottom - GAP_BETWEEN_NODES
        return positions

    positions = [
        _layout_column(rank_nodes[i], i) for i in range(len(ranks))
    ]

    # Build links between consecutive ranks (carrier-count contributions)
    # We need: for each (parent, child), how many carriers contribute
    link_counts = []
    for i in range(len(ranks) - 1):
        pairs = (carriers.groupby([ranks[i], ranks[i + 1]])
                 .size().reset_index(name='cnt'))
        link_counts.append(pairs)

    # Draw nodes
    for col_idx, (nodes, pos_map) in enumerate(zip(rank_nodes, positions)):
        x = COLUMN_X[col_idx]
        for n in nodes:
            top, bottom, cnt = pos_map[n]
            # Colour by phylum (root level)
            if col_idx == 0:
                col = _phylum_color('p__' + n)
            else:
                # Find ancestor phylum
                phy = carriers.loc[carriers[ranks[col_idx]] == n,
                                   'gtdb_phylum_short'].iloc[0]
                col = _phylum_color('p__' + phy)
            ax.add_patch(mpatches.Rectangle(
                (x, bottom), COLUMN_W, top - bottom,
                facecolor=col, edgecolor='white', linewidth=0.6, alpha=0.9))
            label = f"{n} ({cnt})"
            # Place text to the right of leaf columns, to the left of root col
            if col_idx == 0:
                ax.text(x - 0.2, (top + bottom) / 2, label,
                        ha='right', va='center', fontsize=8,
                        color=col, fontweight='bold' if col_idx == 0 else 'normal')
            else:
                ax.text(x + COLUMN_W + 0.15, (top + bottom) / 2, label,
                        ha='left', va='center', fontsize=7, color='#222')

    def _ordered_link_iter(rank_idx):
        """Yield (parent, child, count) ordered so that ribbons don't cross."""
        parent_rank = ranks[rank_idx]
        child_rank = ranks[rank_idx + 1]
        pairs = link_counts[rank_idx]
        # order children by the parent ordering then the child ordering
        parent_order = {p: i for i, p in enumerate(rank_nodes[rank_idx])}
        child_order  = {c: i for i, c in enumerate(rank_nodes[rank_idx + 1])}
        pairs = pairs.copy()
        pairs['_p_ord'] = pairs[parent_rank].map(parent_order)
        pairs['_c_ord'] = pairs[child_rank].map(child_order)
        return pairs.sort_values(['_p_ord', '_c_ord']).itertuples(index=False)

    for i in range(len(ranks) - 1):
        # Cursors (top of remaining capacity) for each parent and each child
        parent_cursor = {n: positions[i][n][0] for n in rank_nodes[i]}
        child_cursor  = {n: positions[i + 1][n][0] for n in rank_nodes[i + 1]}
        for row in _ordered_link_iter(i):
            parent = getattr(row, ranks[i])
            child  = getattr(row, ranks[i + 1])
            cnt    = getattr(row, 'cnt')

            p_top = parent_cursor[parent]
            p_bot = p_top - cnt
            parent_cursor[parent] = p_bot

            c_top = child_cursor[child]
            c_bot = c_top - cnt
            child_cursor[child] = c_bot

            x_l = COLUMN_X[i] + COLUMN_W
            x_r = COLUMN_X[i + 1]
            # control points for cubic bezier
            cx_l = x_l + (x_r - x_l) * 0.5
            cx_r = x_r - (x_r - x_l) * 0.5
            verts = [
                (x_l, p_top),
                (cx_l, p_top), (cx_r, c_top), (x_r, c_top),
                (x_r, c_bot),
                (cx_r, c_bot), (cx_l, p_bot), (x_l, p_bot),
                (x_l, p_top),
            ]
            codes = [MplPath.MOVETO,
                     MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
                     MplPath.LINETO,
                     MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
                     MplPath.CLOSEPOLY]
            phy = carriers.loc[carriers[ranks[i + 1]] == child,
                               'gtdb_phylum_short'].iloc[0]
            col = _phylum_color('p__' + phy)
            ax.add_patch(mpatches.PathPatch(
                MplPath(verts, codes), facecolor=col, edgecolor='none',
                alpha=0.35))

    # Column titles
    for x, title in zip(COLUMN_X, rank_titles):
        ax.text(x + COLUMN_W / 2, canvas_h - PADDING * 0.3, title,
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xlim(COLUMN_X[0] - 4, COLUMN_X[-1] + 4)
    ax.set_ylim(-1, canvas_h + 1)
    ax.set_axis_off()
    plt.tight_layout()
    _save(fig, "03_carrier_taxonomy_hierarchy")


def node_counts_of(node, ranks, node_counts, rank_idx):
    return node_counts[ranks[rank_idx]].get(node, 0)


# ---------------------------------------------------------------------------
# Panel: genus-level prevalence (mirror-bar layout: genus size on left,
# within-genus prevalence on right, grouped by phylum)
# ---------------------------------------------------------------------------
def plot_genus_prevalence(reps):
    df = _prepare(reps)
    g = (df.groupby(['gtdb_phylum', 'gtdb_genus_short'])
           .agg(n_species=('accession', 'size'),
                n_carriers=('is_carrier', 'sum'))
           .reset_index())
    g = g[(g['n_carriers'] > 0) & (g['gtdb_genus_short'].notna()) &
          (g['gtdb_genus_short'] != '')].copy()
    g['pct'] = 100 * g['n_carriers'] / g['n_species']
    # group by phylum, then sort genera within phylum by pct ascending so
    # the highest carrier-rate genera land at the bottom of each group.
    phylum_order = (g.groupby('gtdb_phylum')['n_carriers'].sum()
                    .sort_values(ascending=True).index.tolist())
    rows = []
    for phy in phylum_order:
        sub = g[g['gtdb_phylum'] == phy].sort_values('pct', ascending=True)
        for _, r in sub.iterrows():
            rows.append(r.to_dict())
        rows.append(None)  # separator row
    if rows and rows[-1] is None:
        rows.pop()
    n_rows = len(rows)

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(11, max(8, 0.18 * n_rows)),
        gridspec_kw={'width_ratios': [1, 2], 'wspace': 0.05},
        sharey=True,
    )

    y_positions = np.arange(n_rows)
    for axi in (axL, axR):
        axi.set_yticks(y_positions)
    labels = [r['gtdb_genus_short'] if r is not None else '' for r in rows]
    axL.set_yticklabels(labels, fontsize=7)
    for tick, r in zip(axL.get_yticklabels(), rows):
        if r is not None:
            tick.set_color(_phylum_color(r['gtdb_phylum']))

    # Left panel: total genus size; dark bar = carrier subset.
    max_n = max((r['n_species'] for r in rows if r is not None), default=1)
    for y, r in zip(y_positions, rows):
        if r is None:
            continue
        col = _phylum_color(r['gtdb_phylum'])
        axL.barh(y, r['n_species'], color=col, alpha=0.25, edgecolor='none')
        axL.barh(y, r['n_carriers'], color=col, alpha=0.9, edgecolor='none')
    axL.invert_xaxis()
    axL.set_xlabel('Number of species')
    axL.set_title('Genus size', fontsize=10)
    axL.spines[['top', 'left']].set_visible(False)

    # Right panel: within-genus prevalence percentage with light bar
    for y, r in zip(y_positions, rows):
        if r is None:
            continue
        col = _phylum_color(r['gtdb_phylum'])
        axR.barh(y, r['pct'], color=col, alpha=0.85, edgecolor='none')
        axR.text(r['pct'] + 1.5, y, f"{r['pct']:.1f}%",
                 va='center', fontsize=6.5, color='#333')
    axR.set_xlim(0, 110)
    axR.set_xlabel('Within-genus plasmid prevalence (%)')
    axR.set_title('Within-genus prevalence', fontsize=10)
    axR.spines[['top', 'right']].set_visible(False)
    axR.axvline(50, color='gray', linestyle=':', linewidth=0.5)
    axR.axvline(100, color='gray', linestyle=':', linewidth=0.5)

    # Phylum group labels on far-right
    cumulative = 0
    for phy in phylum_order:
        sub = g[g['gtdb_phylum'] == phy].sort_values('pct', ascending=True)
        midpoint = cumulative + (len(sub) - 1) / 2
        axR.text(112, midpoint, f"{_short(phy)} ({len(sub)} genera)",
                 ha='left', va='center', fontsize=9,
                 color=_phylum_color(phy), fontweight='bold')
        cumulative += len(sub) + 1  # +1 separator

    for axi in (axL, axR):
        axi.invert_yaxis()  # smallest at top → but our rows already sorted
        axi.set_ylim(n_rows - 0.5, -0.5)

    plt.tight_layout()
    _save(fig, "04_genus_within_phylum_prevalence")


# ---------------------------------------------------------------------------
# Panel: family-level summary (3 subpanels)
# ---------------------------------------------------------------------------
def plot_family_summary(reps):
    df = _prepare(reps)
    fam = (df.groupby(['gtdb_phylum', 'gtdb_family_short'])
             .agg(n_species=('accession', 'size'),
                  n_carriers=('is_carrier', 'sum'),
                  total_plasmids=('plasmid_abundance', 'sum'))
             .reset_index())
    fam = fam[(fam['n_carriers'] > 0) & (fam['gtdb_family_short'].notna()) &
              (fam['gtdb_family_short'] != '')].copy()
    fam['pct_carriers'] = 100 * fam['n_carriers'] / fam['n_species']
    fam['mean_abundance'] = (fam['total_plasmids'] /
                             fam['n_carriers'].replace(0, np.nan))
    fam = fam.sort_values(['gtdb_phylum', 'pct_carriers'],
                          ascending=[False, True])

    n_fam = len(fam)
    fig, axes = plt.subplots(
        1, 3, figsize=(11, max(4, 0.32 * n_fam)),
        gridspec_kw={'width_ratios': [1.4, 1, 1], 'wspace': 0.05},
        sharey=True,
    )
    y = np.arange(n_fam)
    colours = [_phylum_color(p) for p in fam['gtdb_phylum']]

    axes[0].barh(y, fam['pct_carriers'], color=colours, edgecolor='white',
                 linewidth=0.4)
    for yi, p, c, n in zip(y, fam['pct_carriers'], fam['n_carriers'],
                            fam['n_species']):
        axes[0].text(p + 0.4, yi, f"{int(c)}/{int(n)}",
                     va='center', fontsize=7, color='#333')
    axes[0].set_xlabel('% species with plasmids')
    axes[0].set_xlim(0, max(fam['pct_carriers']) * 1.18)

    axes[1].barh(y, fam['total_plasmids'], color=colours, edgecolor='white',
                 linewidth=0.4)
    axes[1].set_xlabel('Total plasmid abundance')

    axes[2].barh(y, fam['mean_abundance'], color=colours, edgecolor='white',
                 linewidth=0.4)
    axes[2].set_xlabel('Mean abundance\n(plasmid-bearing spp.)')

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(fam['gtdb_family_short'], fontsize=8)
    for tick, p in zip(axes[0].get_yticklabels(), fam['gtdb_phylum']):
        tick.set_color(_phylum_color(p))
    for axi in axes:
        axi.spines[['top', 'right']].set_visible(False)
    axes[0].invert_yaxis()
    plt.tight_layout()
    _save(fig, "05_family_summary")


# ---------------------------------------------------------------------------
# Panel: top plasmid-bearing species, with model organisms highlighted
# ---------------------------------------------------------------------------
def plot_top_species(reps, top_n=15):
    nonhalo_models, halo_models = load_model_species()
    models = set(nonhalo_models) | set(halo_models)
    carriers = reps[reps['is_carrier'] == 1].copy()
    top = (carriers.sort_values('plasmid_abundance', ascending=False)
                   .head(top_n).copy())
    top['short'] = top['gtdb_species'].map(_short).map(
        lambda s: s.replace('Halobacterium ', 'H. ')
                   .replace('Haloarcula ', 'H. ')
                   .replace('Haloferax ', 'H. ')
                   .replace('Saccharolobus ', 'S. ')
                   .replace('Haloarcula ', 'H. ')
                   .replace('Natrinema ', 'N. ')
                   .replace('Halobaculum ', 'H. ')
                   .replace('Haloplanus ', 'H. ')
                   .replace('Halorubrum ', 'H. '))
    top['is_model'] = top['gtdb_species'].isin(models)

    fig, ax = plt.subplots(figsize=(5.5, max(3, 0.3 * len(top))))
    y = np.arange(len(top))[::-1]
    colours = ['#e6b800' if m else '#3a72c4' for m in top['is_model']]
    ax.barh(y, top['plasmid_abundance'], color=colours, edgecolor='white',
            linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(top['short'], fontsize=8, style='italic')
    for tick, m in zip(ax.get_yticklabels(), top['is_model']):
        if m:
            tick.set_color('#a37d00')
    ax.set_xlabel('Number of plasmids')
    handles = [mpatches.Patch(color='#e6b800', label='Model organism'),
               mpatches.Patch(color='#3a72c4', label='Non-model organism')]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc='lower right')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    _save(fig, "06_top_plasmid_bearing_species")


# ---------------------------------------------------------------------------
# Panel: model-organism contribution to total plasmid abundance,
# Halo vs Non-Halo
# ---------------------------------------------------------------------------
def plot_model_contribution(reps):
    nonhalo_models, halo_models = load_model_species()
    carriers = reps[reps['is_carrier'] == 1].copy()
    halo = carriers[carriers['gtdb_phylum'] == 'p__Halobacteriota']
    nonhalo = carriers[carriers['gtdb_phylum'] != 'p__Halobacteriota']

    def _split(arm, model_list):
        in_model = arm[arm['gtdb_species'].isin(model_list)]
        out_model = arm[~arm['gtdb_species'].isin(model_list)]
        return (int(in_model['plasmid_abundance'].sum()),
                int(out_model['plasmid_abundance'].sum()))

    h_mod, h_other = _split(halo, halo_models)
    n_mod, n_other = _split(nonhalo, nonhalo_models)

    arms = ['Non-Halo.\nplasmids', 'Halo.\nplasmids']
    mod_counts   = [n_mod, h_mod]
    other_counts = [n_other, h_other]
    totals       = [n_mod + n_other, h_mod + h_other]

    fig, ax = plt.subplots(figsize=(3.6, 4))
    xpos = np.arange(2)
    ax.bar(xpos, other_counts, color='#fcbfbf', edgecolor='white',
           linewidth=0.4, label='Other species')
    ax.bar(xpos, mod_counts, bottom=other_counts, color='#d04141',
           edgecolor='white', linewidth=0.4, label='Model organisms')
    for x, total, mod, other in zip(xpos, totals, mod_counts, other_counts):
        ax.text(x, other / 2, f"{other}\n({other/total*100:.0f}%)",
                ha='center', va='center', fontsize=8, color='#333')
        ax.text(x, other + mod / 2, f"{mod}\n({mod/total*100:.0f}%)",
                ha='center', va='center', fontsize=8, color='white')
    ax.set_xticks(xpos)
    ax.set_xticklabels(arms, fontsize=9)
    ax.set_ylabel('Total plasmids')
    ax.set_ylim(0, max(totals) * 1.10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc='upper right')
    plt.tight_layout()
    _save(fig, "07_model_organism_contribution")


# ---------------------------------------------------------------------------
# Panel: plasmid-abundance histogram and per-phylum strip plot
# ---------------------------------------------------------------------------
def plot_abundance_histogram(reps):
    carriers = reps[reps['is_carrier'] == 1].copy()
    abundance = carriers['plasmid_abundance'].astype(int).values
    median_val = float(np.median(abundance))
    max_val = int(abundance.max())

    fig, ax = plt.subplots(figsize=(4, 3))
    bins = np.arange(1, max_val + 2) - 0.5
    ax.hist(abundance, bins=bins, color='#3a72c4', edgecolor='white',
            linewidth=0.5)
    ax.axvline(median_val, color='red', linestyle='--', linewidth=1.2)
    ax.text(median_val + 0.4, ax.get_ylim()[1] * 0.92,
            f"Median = {median_val:.0f}",
            color='red', fontsize=8)
    ax.set_xlabel('Plasmid abundance\n(number of plasmids per species)')
    ax.set_ylabel('Number of species')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    _save(fig, "08_abundance_histogram")


def plot_abundance_per_phylum(reps):
    carriers = reps[reps['is_carrier'] == 1].copy()
    df = _prepare(carriers)
    df = df[df['gtdb_phylum'].isin(CARRIER_PHYLA)]
    phylum_order = (df.groupby('gtdb_phylum')['plasmid_abundance']
                      .median().sort_values(ascending=False).index.tolist())

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    rng = np.random.default_rng(0)
    for i, phy in enumerate(phylum_order):
        vals = df.loc[df['gtdb_phylum'] == phy,
                       'plasmid_abundance'].astype(int).values
        x = i + rng.normal(0, 0.08, size=len(vals))
        ax.scatter(x, vals, s=18, color=_phylum_color(phy),
                   alpha=0.7, edgecolor='none')
        # median line
        med = np.median(vals)
        ax.plot([i - 0.25, i + 0.25], [med, med], color='black', lw=1.0)
    ax.set_xticks(np.arange(len(phylum_order)))
    ax.set_xticklabels([_short(p) for p in phylum_order],
                       rotation=20, ha='right', fontsize=8)
    ax.set_ylabel('Plasmid abundance')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    _save(fig, "09_abundance_per_phylum")


# ---------------------------------------------------------------------------
# Panel: single vs multi-plasmid carriers, Halo vs Non-Halo, with Fisher OR
# (Tukey 1.5*IQR fence to be consistent with 02_abundance_distribution.py)
# ---------------------------------------------------------------------------
def plot_single_vs_multi(reps):
    carriers = reps[reps['is_carrier'] == 1].copy()
    q1, q3 = np.percentile(carriers['plasmid_abundance'], [25, 75])
    upper_fence = q3 + 1.5 * (q3 - q1)
    kept = carriers[carriers['plasmid_abundance'] <= upper_fence]
    halo = kept[kept['gtdb_phylum'] == 'p__Halobacteriota']
    non  = kept[kept['gtdb_phylum'] != 'p__Halobacteriota']

    s_h = int((halo['plasmid_abundance'] == 1).sum())
    m_h = int((halo['plasmid_abundance'] > 1).sum())
    s_n = int((non['plasmid_abundance'] == 1).sum())
    m_n = int((non['plasmid_abundance'] > 1).sum())
    OR, p = stats.fisher_exact([[s_h, m_h], [s_n, m_n]])

    fig, ax = plt.subplots(figsize=(3.4, 3.6))
    arms = ['Halo.', 'Non-Halo.']
    totals  = np.array([s_h + m_h, s_n + m_n], dtype=float)
    multi_pct  = np.array([m_h, m_n]) / totals * 100
    single_pct = np.array([s_h, s_n]) / totals * 100
    multi_n    = [m_h, m_n]
    single_n   = [s_h, s_n]
    xpos = np.arange(2)
    bar_colors_multi  = ['#3a72c4', '#a30a0a']
    bar_colors_single = ['#bcd1ed', '#f3b1b1']
    ax.bar(xpos, multi_pct, color=bar_colors_multi, edgecolor='white',
           linewidth=0.5, label='Multi-plasmid')
    ax.bar(xpos, single_pct, bottom=multi_pct, color=bar_colors_single,
           edgecolor='white', linewidth=0.5, label='Single')
    for x, mp, sp, mn, sn in zip(xpos, multi_pct, single_pct, multi_n, single_n):
        ax.text(x, mp / 2, f"{mn}\n({mp:.0f}%)", ha='center', va='center',
                fontsize=8, color='white')
        ax.text(x, mp + sp / 2, f"{sn}\n({sp:.0f}%)", ha='center',
                va='center', fontsize=8, color='#333')
    ax.set_xticks(xpos)
    ax.set_xticklabels(arms)
    ax.set_ylabel('% of carriers')
    ax.set_ylim(0, 108)
    ax.set_title(f"(OR = {OR:.2g}, p = {p:.4f})", fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc='upper right')
    plt.tight_layout()
    _save(fig, "10_single_vs_multi_plasmid")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading data...")
    reps, _ = load_data()
    print(f"  {len(reps):,} representative species; "
          f"{int(reps['is_carrier'].sum())} carriers")

    print("\nGenerating panels:")
    plot_prevalence_pies(reps)
    plot_per_phylum_rate(reps)
    plot_taxonomy_hierarchy(reps)
    plot_genus_prevalence(reps)
    plot_family_summary(reps)
    plot_top_species(reps)
    plot_model_contribution(reps)
    plot_abundance_histogram(reps)
    plot_abundance_per_phylum(reps)
    plot_single_vs_multi(reps)
    print(f"\nAll outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
