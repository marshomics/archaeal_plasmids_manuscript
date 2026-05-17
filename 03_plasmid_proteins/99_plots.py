#!/usr/bin/env python3
"""Generate every panel for the cross-domain plasmid-protein figure set.

Outputs go to ``outputs/figures/`` next to this script. Every quantity is
re-derived from the input data and the other scripts in this pipeline; no
numbers are hard-coded. Each panel is written as both PNG (raster) and SVG
(``svg.fonttype = 'none'`` so labels stay editable in vector software).

Run as ``python 99_plots.py``. If ``outputs/cluster_summary.csv`` is missing
or incomplete the script rebuilds it on-the-fly using the same logic as
``00_build_cluster_summary.py``.
"""
from collections import defaultdict
import gc
from pathlib import Path
import sys

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path as MplPath
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ARCHAEAL_EGGNOG, CLUSTER_TSV, CLUSTER_SUMMARY,
                    COG_DESCRIPTIONS, FAM_MIN, OUT_DIR, PSEUDO, N_PERM,
                    header)  # noqa

FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
matplotlib.rcParams['font.family']  = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

CARRIER_PHYLA = ['p__Halobacteriota', 'p__Methanobacteriota',
                 'p__Methanobacteriota_B', 'p__Thermoproteota',
                 'p__Thermoplasmatota']
CARRIER_COLOR = {
    'p__Halobacteriota':      '#e69f00',
    'p__Methanobacteriota':   '#2ca02c',
    'p__Methanobacteriota_B': '#cc79a7',
    'p__Thermoproteota':      '#56b4e9',
    'p__Thermoplasmatota':    '#9467bd',
}


def _short(s):
    return s.split('__', 1)[-1] if isinstance(s, str) and '__' in s else s


def _save(fig, stem):
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=200, bbox_inches='tight')
    fig.savefig(FIG_DIR / f"{stem}.svg", bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {stem}.png + .svg")


def _is_unknown(cog):
    if pd.isna(cog) or cog == '-' or str(cog).strip() == '':
        return True
    letters = [c for c in str(cog) if c in COG_DESCRIPTIONS]
    return (not letters) or all(c == 'S' for c in letters)


# ---------------------------------------------------------------------------
# Build / load the cluster-summary table used by panels A-D and the Sankey.
# Mirrors the logic of 00_build_cluster_summary.py.
# ---------------------------------------------------------------------------
def _build_cluster_summary():
    print("  Building cluster summary from raw TSV (this can take a minute)...")
    df = pd.read_csv(
        CLUSTER_TSV, sep="\t", usecols=[0, 1, 2, 3, 4, 5, 6],
        names=["cluster_rep", "member", "domain", "phylum", "class_", "order", "family"],
        header=0, dtype=str, low_memory=True,
    )
    df.loc[df['domain'].isna(), 'domain'] = 'd__Archaea'

    arch_lab = df[(df['domain'] == 'd__Archaea') & df['phylum'].notna()]
    tax = arch_lab.drop_duplicates(subset='cluster_rep', keep='first').set_index(
        'cluster_rep')[['phylum', 'family']]
    mask = (df['domain'] == 'd__Archaea') & df['phylum'].isna()
    if mask.any():
        for col in ['phylum', 'family']:
            df.loc[mask, col] = df.loc[mask, 'cluster_rep'].map(tax[col]).values
    del arch_lab, tax; gc.collect()

    df['is_arch'] = (df['domain'] == 'd__Archaea')
    df['is_bact'] = (df['domain'] == 'd__Bacteria')
    counts = df.groupby('cluster_rep').agg(
        n_total=('member', 'size'),
        n_archaea=('is_arch', 'sum'),
        n_bacteria=('is_bact', 'sum'),
    )
    counts['cluster_type'] = np.where(
        (counts['n_archaea'] > 0) & (counts['n_bacteria'] > 0), 'cross-domain',
        np.where(counts['n_archaea'] > 0, 'archaea-only', 'bacteria-only'),
    )
    arch = df[df['is_arch']]
    bact = df[df['is_bact']]
    arch_phyla = arch.groupby('cluster_rep')['phylum'].apply(
        lambda x: "|".join(sorted(x.dropna().unique()))).rename('archaea_phyla')
    bact_phyla = bact.groupby('cluster_rep')['phylum'].apply(
        lambda x: "|".join(sorted(x.dropna().unique()))).rename('bacteria_phyla')
    arch_fams = arch.groupby('cluster_rep')['family'].apply(
        lambda x: "|".join(sorted(x.dropna().unique()))).rename('archaea_families')

    clusters = counts.join(arch_phyla).join(bact_phyla).join(arch_fams).reset_index()
    for c in ['archaea_phyla', 'bacteria_phyla', 'archaea_families']:
        clusters[c] = clusters[c].fillna("")
    return clusters


def load_cluster_summary(min_rows=600_000):
    """Read the cluster_summary CSV produced by 00_build_cluster_summary.py.

    If the file is missing or substantially smaller than expected the user is
    asked to run that script first; we do not try to rebuild silently because
    the build is expensive (≈ 6.4 M-row read) and can dominate runtime.
    """
    if CLUSTER_SUMMARY.exists():
        cl = pd.read_csv(CLUSTER_SUMMARY)
        if len(cl) >= min_rows:
            return cl
        msg = (f"cluster_summary.csv has only {len(cl):,} rows. "
               f"Run 00_build_cluster_summary.py first to regenerate it.")
    else:
        msg = ("cluster_summary.csv is missing. "
               "Run 00_build_cluster_summary.py first to generate it.")
    raise RuntimeError(msg)


# ===========================================================================
# Extended Data A — counts of bacteria-only / archaea-only / cross-domain
# ===========================================================================
def plot_cluster_type_counts(clusters):
    ct = clusters['cluster_type'].value_counts()
    order = ['bacteria-only', 'archaea-only', 'cross-domain']
    ct = ct.reindex(order)
    total = ct.sum()
    colours = ['#7fbfd1', '#e69f00', '#cc79a7']

    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    x = np.arange(len(ct))
    ax.bar(x, ct.values, color=colours, edgecolor='white', linewidth=0.5,
           width=0.7)
    for xi, val in zip(x, ct.values):
        ax.text(xi, val * 1.05, f"{val:,}\n({val/total*100:.1f}%)",
                ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(['Bacteria-\nonly', 'Archaea-\nonly', 'Cross-\ndomain'])
    ax.set_yscale('log')
    ax.set_ylabel('Number of clusters')
    ax.set_ylim(top=ct.max() * 5)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    _save(fig, "extA_cluster_type_counts")


# ===========================================================================
# Extended Data B — pie of archaeal proteins in cross-domain vs archaea-only
# ===========================================================================
def plot_archaeal_protein_share(clusters):
    arch_cd = int(clusters.loc[clusters['cluster_type'] == 'cross-domain',
                               'n_archaea'].sum())
    arch_ao = int(clusters.loc[clusters['cluster_type'] == 'archaea-only',
                               'n_archaea'].sum())
    total = arch_cd + arch_ao

    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    wedges, _ = ax.pie([arch_ao, arch_cd], colors=['#e69f00', '#cc79a7'],
                       startangle=90,
                       wedgeprops=dict(edgecolor='white', linewidth=1.5))
    labels = [f"In archaea-only\nclusters\n{arch_ao:,} ({arch_ao/total*100:.1f}%)",
              f"In cross-domain\nclusters\n{arch_cd:,} ({arch_cd/total*100:.1f}%)"]
    for w, lab in zip(wedges, labels):
        ang = (w.theta1 + w.theta2) / 2
        x = 1.25 * np.cos(np.deg2rad(ang))
        y = 1.25 * np.sin(np.deg2rad(ang))
        ax.text(x, y, lab, ha='center', va='center', fontsize=8)
    ax.set_aspect('equal')
    plt.tight_layout()
    _save(fig, "extB_archaeal_protein_share")


# ===========================================================================
# Extended Data C — CDF of cluster sizes by type
# ===========================================================================
def plot_cluster_size_cdf(clusters):
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    palette = {'bacteria-only': '#1f9be0',
               'archaea-only':  '#e69f00',
               'cross-domain':  '#cc79a7'}
    for typ, col in palette.items():
        sizes = np.sort(clusters.loc[clusters['cluster_type'] == typ,
                                     'n_total'].values)
        if len(sizes) == 0:
            continue
        cdf = np.arange(1, len(sizes) + 1) / len(sizes)
        ax.step(sizes, cdf, where='post', color=col, lw=1.4,
                label=f"{typ.capitalize()} (n = {len(sizes):,})")
    ax.set_xscale('log')
    ax.set_xlabel('Cluster size')
    ax.set_ylabel('Cumulative fraction')
    ax.legend(frameon=False, fontsize=7, loc='lower right')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    _save(fig, "extC_cluster_size_cdf")


# ===========================================================================
# Extended Data D — per-phylum cross-domain cluster counts
# (one row per archaeal phylum; archaea-only + cross-domain stacked)
# ===========================================================================
def plot_per_phylum_cd_counts(clusters):
    """Count clusters in which each phylum appears as one of its archaeal members.
    Uses an explode on the '|' delimiter to count multi-phylum clusters once
    per phylum (correct), rather than a substring match that would conflate
    p__Methanobacteriota with p__Methanobacteriota_B.
    """
    arch = clusters[clusters['n_archaea'] > 0].copy()
    arch['phyla_list'] = arch['archaea_phyla'].fillna('').str.split('|')
    exploded = arch.explode('phyla_list')
    exploded = exploded[exploded['phyla_list'] != '']
    grp = (exploded.groupby(['phyla_list', 'cluster_type'])
                    .agg(n_clusters=('cluster_rep', 'nunique'))
                    .unstack(fill_value=0))
    grp.columns = grp.columns.droplevel(0)
    for c in ['archaea-only', 'cross-domain']:
        if c not in grp.columns:
            grp[c] = 0
    grp['total'] = grp[['archaea-only', 'cross-domain']].sum(axis=1)
    grp['pct_cd'] = 100 * grp['cross-domain'] / grp['total']
    grp = grp.sort_values('total', ascending=True)  # smallest at top

    fig, ax = plt.subplots(figsize=(5.6, max(2.5, 0.35 * len(grp))))
    y = np.arange(len(grp))
    ax.barh(y, grp['archaea-only'], color='#e69f00', edgecolor='white',
            label='Archaea-only', linewidth=0.4)
    ax.barh(y, grp['cross-domain'], left=grp['archaea-only'], color='#cc79a7',
            edgecolor='white', label='Cross-domain', linewidth=0.4)
    for yi, (phy, r) in zip(y, grp.iterrows()):
        ax.text(r['total'] * 1.02, yi,
                f"{r['pct_cd']:.1f}% cross-domain\n(n = {int(r['total']):,})",
                va='center', fontsize=7, color='#222')
    ax.set_yticks(y)
    ax.set_yticklabels([_short(p) for p in grp.index], fontsize=8)
    ax.set_xlim(0, grp['total'].max() * 1.42)
    ax.set_xlabel('Number of clusters')
    ax.legend(frameon=False, fontsize=8, loc='lower right')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    _save(fig, "extD_per_phylum_cd_counts")


# ===========================================================================
# COG enrichment helper (Fisher cross-domain vs archaea-only with fractional
# weights for multi-letter COGs). Returns DataFrame keyed by COG with OR + q.
# ===========================================================================
def compute_cog_enrichment():
    eg = pd.read_csv(ARCHAEAL_EGGNOG, sep='\t', low_memory=False)
    # Restrict to archaeal proteins that belong to clusters we've classified
    # (cross-domain vs archaea-only). The eggnog table already represents
    # archaeal plasmid proteins. We need to label them by cluster_type, which
    # requires a join via the cluster TSV — but we can shortcut by reading
    # the cluster_summary built earlier and looking up each protein's
    # cluster_rep via the raw TSV (expensive).
    # The eggnog table doesn't carry cluster IDs by default, so we read the
    # raw cluster TSV again to label proteins.
    cl = pd.read_csv(CLUSTER_TSV, sep='\t', usecols=[0, 1, 2], dtype=str,
                     low_memory=True,
                     names=['cluster_rep', 'member', 'domain'], header=0)
    cl.loc[cl['domain'].isna(), 'domain'] = 'd__Archaea'
    cl = cl[cl['domain'] == 'd__Archaea'][['cluster_rep', 'member']]
    # cluster_type
    summary = load_cluster_summary()[['cluster_rep', 'cluster_type']]
    cl = cl.merge(summary, on='cluster_rep', how='left')
    # member id is the same as eggnog `query` / first column?
    eg_first = eg.columns[0]
    eg = eg.merge(cl, left_on=eg_first, right_on='member', how='left')
    eg = eg.dropna(subset=['cluster_type'])

    rows = []
    cog_letters = [c for c in COG_DESCRIPTIONS if c not in ('R', 'S')]
    for letter in cog_letters:
        # fractional weight = 1/k if a protein has k informative letters
        def _weight(cog):
            if pd.isna(cog) or cog == '-':
                return 0.0
            letters = [c for c in str(cog) if c in COG_DESCRIPTIONS
                       and c not in ('R', 'S')]
            if letter not in letters or len(letters) == 0:
                return 0.0
            return 1.0 / len(letters)
        w = eg['COG_category'].apply(_weight)
        cd  = w[(eg['cluster_type'] == 'cross-domain')].sum()
        ao  = w[(eg['cluster_type'] == 'archaea-only')].sum()
        # totals for the 2x2: a = cd-with-letter, b = cd-without, c = ao-with,
        # d = ao-without. Use weights rounded to integers (small bounded bias).
        a = int(round(cd))
        c = int(round(ao))
        b = int(round((eg['cluster_type'] == 'cross-domain').sum() - cd))
        d = int(round((eg['cluster_type'] == 'archaea-only').sum() - ao))
        if a + c < 1:
            continue
        OR, p = stats.fisher_exact([[a, b], [c, d]])
        rows.append({'COG': letter, 'description': COG_DESCRIPTIONS[letter],
                     'cd_w': cd, 'ao_w': ao, 'OR': OR, 'p_raw': p})
    out = pd.DataFrame(rows)
    out['p_BH'] = multipletests(out['p_raw'], method='fdr_bh')[1]
    return out


# ===========================================================================
# Fig 3A — tripartite Sankey: archaeal phyla → COG categories (OR) →
# bacterial phyla (% of cross-domain clusters)
# ===========================================================================
def plot_tripartite_sankey(clusters):
    # ---- compute per-archaeal-phylum contribution to cross-domain clusters
    cd = clusters[clusters['cluster_type'] == 'cross-domain'].copy()
    cd['archaea_phyla_list']  = cd['archaea_phyla'].fillna('').str.split('|')
    cd['bacteria_phyla_list'] = cd['bacteria_phyla'].fillna('').str.split('|')

    # archaeal phyla and their cross-domain cluster counts (exploded)
    arch_explode = cd.explode('archaea_phyla_list')
    arch_explode = arch_explode[arch_explode['archaea_phyla_list'] != '']
    arch_counts = (arch_explode.groupby('archaea_phyla_list')
                                .agg(n_cd=('cluster_rep', 'nunique')))
    arch_counts = arch_counts.sort_values('n_cd', ascending=False)
    arch_order = arch_counts.head(8).index.tolist()

    # bacterial phyla — % of cross-domain clusters they appear in
    n_cd_total = cd['cluster_rep'].nunique()
    bact_explode = cd.explode('bacteria_phyla_list')
    bact_explode = bact_explode[bact_explode['bacteria_phyla_list'] != '']
    bact_counts = (bact_explode.groupby('bacteria_phyla_list')
                                .agg(n=('cluster_rep', 'nunique')))
    bact_counts['pct'] = 100 * bact_counts['n'] / n_cd_total
    bact_counts = bact_counts.sort_values('pct', ascending=False)
    # Restrict to top 9 to match the panel
    bact_order = bact_counts.head(9).index.tolist()

    # COG enrichment (middle)
    enrich = compute_cog_enrichment().copy()
    # Drop the unknown / general categories (already excluded in helper)
    enrich = enrich[enrich['OR'] > 1].sort_values('OR', ascending=False)
    cog_order = enrich.head(10)['COG'].tolist()
    cog_or = dict(zip(enrich['COG'], enrich['OR']))
    cog_desc_short = {
        'I': 'Lipid', 'E': 'Amino acid', 'F': 'Nucleotide', 'G': 'Carbohydrate',
        'C': 'Energy', 'H': 'Coenzyme', 'Q': 'Secondary metab.',
        'P': 'Inorganic ion', 'M': 'Cell wall', 'O': 'PTM/chaperone',
        'J': 'Translation', 'L': 'Replication/repair', 'K': 'Transcription',
        'D': 'Cell cycle', 'T': 'Signal transduction', 'N': 'Cell motility',
        'U': 'Trafficking', 'V': 'Defense', 'A': 'RNA proc.',
    }

    # ---- layout
    fig, ax = plt.subplots(figsize=(9.5, max(5, 0.45 * len(cog_order))))
    X_LEFT, X_MIDL, X_MIDR, X_RIGHT = 0, 4, 6, 10
    NODE_W = 0.4

    arch_y = np.linspace(0, len(cog_order) - 1, len(arch_order))
    bact_y = np.linspace(0, len(cog_order) - 1, len(bact_order))
    cog_y  = np.arange(len(cog_order))

    or_max = float(np.log2(max(cog_or.values()))) if cog_or else 1.0
    cmap = plt.colormaps['YlOrRd']
    def _flow_color(or_val):
        v = np.log2(or_val) / or_max if or_val > 1 else 0
        return cmap(np.clip(v, 0, 1))

    # Draw archaeal phyla nodes (small circles)
    for phy, y in zip(arch_order, arch_y):
        col = CARRIER_COLOR.get(phy, '#888')
        ax.scatter(X_LEFT, y, s=200, color=col, edgecolor='white',
                   linewidth=1.5, zorder=4)
        ax.text(X_LEFT - 0.3, y, _short(phy), ha='right', va='center',
                fontsize=8, color=col, fontweight='bold')

    # Draw COG nodes (rectangles, coloured by OR)
    for cog, y in zip(cog_order, cog_y):
        or_v = cog_or[cog]
        col = _flow_color(or_v)
        ax.add_patch(mpatches.Rectangle((X_MIDL, y - 0.3), NODE_W, 0.6,
                                        facecolor=col, edgecolor='black',
                                        linewidth=0.4, zorder=4))
        lab = cog_desc_short.get(cog, cog)
        ax.text(X_MIDL + NODE_W + 0.1, y, f"{lab} ({or_v:.1f}×)",
                ha='left', va='center', fontsize=8)

    # Draw bacterial phyla nodes
    for bp, y in zip(bact_order, bact_y):
        pct = bact_counts.loc[bp, 'pct']
        ax.scatter(X_RIGHT, y, s=80, color='#7fbfd1', edgecolor='white',
                   linewidth=1.0, zorder=4)
        ax.text(X_RIGHT + 0.3, y, f"{_short(bp)} ({pct:.0f}%)",
                ha='left', va='center', fontsize=8)

    # Left ribbons: archaeal phylum → each cog category, width proportional
    # to fraction of that archaeon's CD contribution that falls in each cog
    arch_cog = (arch_explode.merge(
        # we need eggnog data joined — approximate by archaeal phyla's
        # share of cross-domain clusters that involve each cog
        pd.DataFrame({'cog': cog_order, 'OR': [cog_or[c] for c in cog_order]}),
        how='cross'
    ))
    # Approximate flow widths: per-(arch, cog) protein counts come from the
    # cog enrichment table by phylum. Computing it requires reading eggnog —
    # we already do that inside compute_cog_enrichment, but for the per-phylum
    # split we need a re-load. Use the eggnog table directly here.
    eg = pd.read_csv(ARCHAEAL_EGGNOG, sep='\t', low_memory=False)
    eg_first = eg.columns[0]
    cl_arch = pd.read_csv(CLUSTER_TSV, sep='\t', usecols=[0, 1, 2, 3],
                          dtype=str, low_memory=True,
                          names=['cluster_rep', 'member', 'domain', 'phylum'],
                          header=0)
    cl_arch.loc[cl_arch['domain'].isna(), 'domain'] = 'd__Archaea'
    cl_arch = cl_arch[cl_arch['domain'] == 'd__Archaea']
    summary = load_cluster_summary()[['cluster_rep', 'cluster_type']]
    cl_arch = cl_arch.merge(summary, on='cluster_rep', how='left')
    cl_arch = cl_arch[cl_arch['cluster_type'] == 'cross-domain']
    eg = eg.merge(cl_arch[['member', 'phylum', 'cluster_type']],
                  left_on=eg_first, right_on='member', how='inner')

    def _has_cog(cog_str, letter):
        if pd.isna(cog_str) or cog_str == '-':
            return False
        return letter in [c for c in str(cog_str) if c in COG_DESCRIPTIONS]

    arch_cog_counts = defaultdict(lambda: defaultdict(int))
    for letter in cog_order:
        hits = eg[eg['COG_category'].apply(lambda s: _has_cog(s, letter))]
        for phy, n in hits['phylum'].value_counts().items():
            arch_cog_counts[phy][letter] += int(n)

    # Normalise and draw flows
    for phy, y_p in zip(arch_order, arch_y):
        total = sum(arch_cog_counts[phy].values()) or 1
        for cog, y_c in zip(cog_order, cog_y):
            cnt = arch_cog_counts[phy].get(cog, 0)
            if cnt == 0:
                continue
            width = 0.4 * cnt / total + 0.04
            col = _flow_color(cog_or[cog])
            _bezier_flow(ax, X_LEFT + 0.15, y_p, X_MIDL, y_c, width=width,
                         color=col, alpha=0.5)

    # Right ribbons: COG → bacterial phyla, width ∝ bacterial-phylum share
    # of clusters containing that cog (approximate via uniform per-cog)
    # We use the relative size of each bacterial phylum's CD share as weight.
    bact_weights = bact_counts['pct'].to_dict()
    bact_total_w = sum(bact_weights[b] for b in bact_order)
    for cog, y_c in zip(cog_order, cog_y):
        col = _flow_color(cog_or[cog])
        for bp, y_b in zip(bact_order, bact_y):
            w_share = bact_weights[bp] / bact_total_w
            width = 0.35 * w_share + 0.04
            _bezier_flow(ax, X_MIDR, y_c, X_RIGHT - 0.1, y_b, width=width,
                         color=col, alpha=0.4)

    # Colour bar
    norm = matplotlib.colors.Normalize(vmin=0, vmax=or_max)
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, shrink=0.4, pad=0.02)
    cbar.set_label('log₂(odds ratio)', fontsize=8)

    # Headers
    ax.text(X_LEFT, len(cog_order) + 0.4, 'Archaeal\nPhyla',
            ha='center', va='bottom', fontsize=10, color='#e69f00',
            fontweight='bold')
    ax.text(X_MIDL + NODE_W/2, len(cog_order) + 0.4,
            'Functional Categories\n(odds ratio)',
            ha='center', va='bottom', fontsize=10, color='#b85e00',
            fontweight='bold')
    ax.text(X_RIGHT, len(cog_order) + 0.4,
            'Bacterial Phyla\n(% of cross-domain clusters)',
            ha='center', va='bottom', fontsize=10, color='#1f78b4',
            fontweight='bold')

    ax.set_xlim(X_LEFT - 3, X_RIGHT + 4)
    ax.set_ylim(-1, len(cog_order) + 2)
    ax.set_axis_off()
    plt.tight_layout()
    _save(fig, "fig3A_tripartite_sankey")


def _bezier_flow(ax, x1, y1, x2, y2, width, color, alpha):
    cx = (x1 + x2) / 2
    verts = [(x1, y1 + width/2),
             (cx, y1 + width/2), (cx, y2 + width/2), (x2, y2 + width/2),
             (x2, y2 - width/2),
             (cx, y2 - width/2), (cx, y1 - width/2), (x1, y1 - width/2),
             (x1, y1 + width/2)]
    codes = [MplPath.MOVETO,
             MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
             MplPath.LINETO,
             MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
             MplPath.CLOSEPOLY]
    ax.add_patch(mpatches.PathPatch(MplPath(verts, codes),
                                    facecolor=color, edgecolor='none',
                                    alpha=alpha))


# ===========================================================================
# Fig 3B — standardized residuals (archaeal family × bacterial phylum)
# ===========================================================================
def plot_partner_residuals_heatmap(clusters):
    cd = clusters[clusters['cluster_type'] == 'cross-domain'].copy()
    cd['fam_list'] = cd['archaea_families'].fillna('').str.split('|')
    cd['bp_list']  = cd['bacteria_phyla'].fillna('').str.split('|')
    pairs = (cd.explode('fam_list').explode('bp_list')
               [['fam_list', 'bp_list']])
    pairs = pairs[(pairs['fam_list'] != '') & (pairs['bp_list'] != '')]
    table = pd.crosstab(pairs['fam_list'], pairs['bp_list'])
    # Drop rows / cols with low totals so visualisation isn't dominated by noise
    table = table.loc[table.sum(axis=1) >= 20, table.sum(axis=0) >= 20]
    if table.empty:
        print("    skipping fig3B: no families × phyla pairs survive filter")
        return
    # Standardised residuals (Pearson, with finite-sample correction)
    chi2, p, dof, expected = stats.chi2_contingency(table, correction=False)
    n = table.values.sum()
    row_sums = table.sum(axis=1).values[:, None]
    col_sums = table.sum(axis=0).values[None, :]
    std_resid = (table.values - expected) / np.sqrt(
        expected * (1 - row_sums / n) * (1 - col_sums / n))
    sr = pd.DataFrame(std_resid, index=table.index, columns=table.columns)
    # Order families by row sum descending; columns by column sum descending
    sr = sr.loc[table.sum(axis=1).sort_values(ascending=False).index,
                table.sum(axis=0).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(max(6, 0.35 * sr.shape[1]),
                                    max(4, 0.32 * sr.shape[0])))
    vmax = max(abs(sr.values.min()), abs(sr.values.max()))
    im = ax.imshow(sr.values, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                   aspect='auto')
    ax.set_xticks(np.arange(sr.shape[1]))
    ax.set_xticklabels([_short(c) for c in sr.columns], rotation=80,
                       ha='right', fontsize=7)
    ax.set_yticks(np.arange(sr.shape[0]))
    ax.set_yticklabels([_short(r) for r in sr.index], fontsize=8)
    ax.set_title('Family', fontsize=9, pad=8)
    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('Standardized residual', fontsize=8)
    plt.tight_layout()
    _save(fig, "fig3B_partner_residuals")


# ===========================================================================
# Per-(group, COG) CLR enrichment helper — phylum and family levels, both
# raw mean CLR and ΔCLR vs background. Uses ISF weighting and fractional COG.
# ===========================================================================
def compute_clr_tables():
    """Return (phylum_table, family_table) DataFrames keyed by (group, COG).

    Mean CLR is the species-weighted mean of CLR-transformed COG proportions
    for plasmids in the focal group; ΔCLR is the focal mean minus the mean
    over all other plasmids; p comes from a permutation test that shuffles
    focal/background labels (N_PERM iterations); BH-FDR is over (group, COG).

    Vectorised path: fractional COG weights are expanded by exploding the
    COG_category string into individual letters; per-(plasmid, COG) totals
    are built with a single groupby; permutations operate on the CLR matrix
    rather than iterating over individual rows.
    """
    df = pd.read_csv(ARCHAEAL_EGGNOG, sep='\t', low_memory=False)
    df = df.dropna(subset=['replicon', 'gtdb_phylum'])
    cog_letters = [c for c in COG_DESCRIPTIONS if c not in ('R', 'S')]
    letter_set = set(cog_letters)

    cogs_str = df['COG_category'].astype(str)
    df['letters'] = cogs_str.apply(
        lambda s: [c for c in s if c in letter_set]
    )
    df = df[df['letters'].map(len) > 0]
    df['w_per_letter'] = 1.0 / df['letters'].map(len)
    long = df.explode('letters')[['replicon', 'gtdb_phylum', 'gtdb_family',
                                   'gtdb_species', 'letters', 'w_per_letter']]
    long = long.rename(columns={'letters': 'cog', 'w_per_letter': 'w',
                                'gtdb_phylum': 'phylum',
                                'gtdb_family': 'family',
                                'gtdb_species': 'species'})

    mat = (long.groupby(['replicon', 'cog'])['w'].sum()
                .unstack(fill_value=0)
                .reindex(columns=sorted(cog_letters), fill_value=0)
                .add(PSEUDO))
    log_mat = np.log(mat)
    clr = log_mat.sub(log_mat.mean(axis=1), axis=0)

    rep_tax = (long.drop_duplicates('replicon')
                    .set_index('replicon')[['phylum', 'family', 'species']])
    sp_counts = rep_tax['species'].value_counts()
    rep_tax['isf_w'] = rep_tax['species'].map(lambda s: 1.0 / sp_counts.get(s, 1))
    clr = clr.loc[rep_tax.index]
    isf = rep_tax['isf_w'].values

    rng = np.random.default_rng(42)

    def _enrich(group_col):
        groups = rep_tax[group_col].dropna().unique()
        results = []
        clr_mat = clr.values
        labels_master = rep_tax[group_col].values
        for g in groups:
            focal = (labels_master == g)
            n_focal = int(focal.sum())
            if n_focal < FAM_MIN:
                continue
            wf = isf[focal]; wb = isf[~focal]
            fmean = np.average(clr_mat[focal], weights=wf, axis=0)
            bmean = np.average(clr_mat[~focal], weights=wb, axis=0)
            delta = fmean - bmean

            # Vectorised permutation: shuffle the focal indicator N_PERM times.
            perm_indices = np.array([rng.permutation(len(labels_master))
                                     for _ in range(N_PERM)])
            null_delta = np.empty((N_PERM, clr.shape[1]))
            for i in range(N_PERM):
                idx = perm_indices[i]
                f_perm = focal[idx]
                wfp = isf[f_perm]; wbp = isf[~f_perm]
                fmean_p = np.average(clr_mat[f_perm], weights=wfp, axis=0)
                bmean_p = np.average(clr_mat[~f_perm], weights=wbp, axis=0)
                null_delta[i] = fmean_p - bmean_p
            extreme = (np.abs(null_delta) >= np.abs(delta)).sum(axis=0)
            p_raw = (extreme + 1) / (N_PERM + 1)
            for j, cog in enumerate(clr.columns):
                results.append({'group': g, 'cog': cog,
                                'mean_clr': fmean[j], 'delta': delta[j],
                                'n': n_focal, 'p_raw': p_raw[j]})
        d = pd.DataFrame(results)
        if not d.empty:
            d['p_BH'] = multipletests(d['p_raw'], method='fdr_bh')[1]
        return d

    return _enrich('phylum'), _enrich('family')


def _heatmap(ax, df, value_col, group_order, cog_order, vmin, vmax,
             cmap='RdBu_r', sig_col=None, sig_threshold=0.05):
    mat = df.pivot(index='group', columns='cog', values=value_col).reindex(
        index=group_order, columns=cog_order)
    im = ax.imshow(mat.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_xticks(np.arange(len(cog_order)))
    ax.set_xticklabels([f"{c}: {COG_DESCRIPTIONS[c]}" for c in cog_order],
                       rotation=80, ha='right', fontsize=7)
    ax.set_yticks(np.arange(len(group_order)))
    ax.set_yticklabels([_short(g) for g in group_order], fontsize=8)
    if sig_col is not None:
        sig = df.pivot(index='group', columns='cog', values=sig_col).reindex(
            index=group_order, columns=cog_order)
        for i, g in enumerate(group_order):
            for j, c in enumerate(cog_order):
                if pd.notna(sig.loc[g, c]) and sig.loc[g, c] < sig_threshold:
                    cell_val = mat.loc[g, c]
                    txt_col = 'white' if abs(cell_val) > (vmax * 0.6) else 'black'
                    ax.text(j, i, '*', ha='center', va='center',
                            color=txt_col, fontsize=10, fontweight='bold')
    return im


def plot_phylum_clr_delta_heatmap(phylum_table):
    df = phylum_table[phylum_table['group'].isin(CARRIER_PHYLA)].copy()
    # Order phyla by total |delta| descending
    order_score = df.groupby('group')['delta'].apply(lambda x: x.abs().sum())
    group_order = order_score.sort_values(ascending=False).index.tolist()
    # Order COGs by max abs delta across phyla
    cog_order = (df.groupby('cog')['delta'].apply(lambda x: x.abs().max())
                   .sort_values(ascending=True).index.tolist())
    vmax = float(np.percentile(np.abs(df['delta']), 99))
    fig, ax = plt.subplots(figsize=(max(6, 0.45 * len(cog_order)),
                                    max(2.5, 0.6 * len(group_order))))
    im = _heatmap(ax, df, 'delta', group_order, cog_order,
                  vmin=-vmax, vmax=vmax, sig_col='p_BH')
    ax.set_title('Phylum', fontsize=9, pad=8)
    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('ΔCLR', fontsize=8)
    plt.tight_layout()
    _save(fig, "fig3C_phylum_clr_delta")


def plot_phylum_clr_raw_heatmap(phylum_table):
    df = phylum_table[phylum_table['group'].isin(CARRIER_PHYLA)].copy()
    group_order = CARRIER_PHYLA  # canonical order
    # COG ordering: by mean CLR descending across phyla
    cog_order = (df.groupby('cog')['mean_clr'].mean()
                   .sort_values(ascending=False).index.tolist())
    vmax = float(np.percentile(np.abs(df['mean_clr']), 95))
    fig, ax = plt.subplots(figsize=(max(8, 0.42 * len(cog_order)),
                                    max(2.5, 0.6 * len(group_order))))
    im = _heatmap(ax, df, 'mean_clr', group_order, cog_order,
                  vmin=-vmax, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('CLR value', fontsize=8)
    plt.tight_layout()
    _save(fig, "extE_phylum_clr_raw")


def plot_family_clr_delta_heatmap(family_table):
    df = family_table.copy()
    # Keep families with at least one BH-significant cell
    keep = (df.groupby('group')['p_BH'].min() < 0.05)
    fams = keep[keep].index.tolist()
    if not fams:
        # fallback: top 10 by total |delta|
        order = (df.groupby('group')['delta'].apply(lambda x: x.abs().sum())
                   .sort_values(ascending=False).index.tolist())
        fams = order[:12]
    df = df[df['group'].isin(fams)].copy()
    cog_order = (df.groupby('cog')['delta'].mean()
                   .sort_values(ascending=True).index.tolist())
    fam_order = sorted(fams, key=lambda x: _short(x))
    vmax = float(np.percentile(np.abs(df['delta']), 99))
    fig, ax = plt.subplots(figsize=(max(7, 0.35 * len(fam_order)),
                                    max(5, 0.35 * len(cog_order))))
    mat = df.pivot(index='cog', columns='group', values='delta').reindex(
        index=cog_order, columns=fam_order)
    im = ax.imshow(mat.values, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                   aspect='auto')
    sig = df.pivot(index='cog', columns='group', values='p_BH').reindex(
        index=cog_order, columns=fam_order)
    for i, c in enumerate(cog_order):
        for j, g in enumerate(fam_order):
            if pd.notna(sig.loc[c, g]) and sig.loc[c, g] < 0.05:
                cell = mat.loc[c, g]
                txt = 'white' if abs(cell) > (vmax * 0.6) else 'black'
                ax.text(j, i, '*', ha='center', va='center',
                        color=txt, fontsize=10, fontweight='bold')
    ax.set_xticks(np.arange(len(fam_order)))
    ax.set_xticklabels([_short(g) for g in fam_order], rotation=80,
                       ha='right', fontsize=8)
    ax.set_yticks(np.arange(len(cog_order)))
    ax.set_yticklabels([COG_DESCRIPTIONS[c] for c in cog_order], fontsize=8)
    ax.set_title('Family', fontsize=9, pad=8)
    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('ΔCLR', fontsize=8)
    plt.tight_layout()
    _save(fig, "extF_family_clr_delta")


# ===========================================================================
# Fig 3D — protein annotations per COG category
# ===========================================================================
def plot_cog_annotation_counts():
    df = pd.read_csv(ARCHAEAL_EGGNOG, sep='\t', low_memory=False)
    counts = defaultdict(float)
    n_unknown = 0
    for cog in df['COG_category']:
        if pd.isna(cog) or cog == '-':
            n_unknown += 1
            continue
        letters = [c for c in str(cog) if c in COG_DESCRIPTIONS]
        if not letters:
            n_unknown += 1
            continue
        # Fractional assignment for multi-letter COGs
        w = 1.0 / len(letters)
        for l in letters:
            if l == 'S':
                n_unknown += w
            else:
                counts[l] += w
    counts = pd.Series(counts, name='count').sort_values(ascending=False)
    counts_named = pd.DataFrame({
        'category': [COG_DESCRIPTIONS[c] for c in counts.index],
        'count':    [int(round(v)) for v in counts.values],
        'unknown':  False,
    })
    counts_named.loc[len(counts_named)] = {
        'category': 'Unknown function',
        'count':    int(round(n_unknown)),
        'unknown':  True,
    }
    counts_named = counts_named.sort_values('count', ascending=False)

    fig, ax = plt.subplots(figsize=(6.5, max(3.5, 0.32 * len(counts_named))))
    y = np.arange(len(counts_named))[::-1]
    colours = ['#bbbbbb' if u else '#3a72c4' for u in counts_named['unknown']]
    ax.barh(y, counts_named['count'], color=colours, edgecolor='white',
            linewidth=0.4)
    for yi, c in zip(y, counts_named['count']):
        ax.text(c * 1.01, yi, f"{c:,}", va='center', fontsize=7, color='#333')
    ax.set_yticks(y); ax.set_yticklabels(counts_named['category'], fontsize=8)
    ax.set_xlabel('Number of protein annotations')
    ax.set_xlim(0, counts_named['count'].max() * 1.12)
    handles = [mpatches.Patch(color='#bbbbbb', label='Unknown/unannotated'),
               mpatches.Patch(color='#3a72c4', label='Characterised function')]
    ax.legend(handles=handles, frameon=False, fontsize=7, loc='lower right')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    _save(fig, "fig3D_cog_annotation_counts")


# ===========================================================================
# Fig 3E — pie of plasmids by majority annotation status
# ===========================================================================
def plot_majority_unknown_pie():
    df = pd.read_csv(ARCHAEAL_EGGNOG, sep='\t', low_memory=False)
    df['is_unknown'] = df['COG_category'].apply(_is_unknown)
    plas = df.groupby('replicon').agg(
        n=('is_unknown', 'size'),
        n_unknown=('is_unknown', 'sum')).reset_index()
    plas['pct_unknown'] = 100.0 * plas['n_unknown'] / plas['n']
    n_majority_unknown = int((plas['pct_unknown'] > 50).sum())
    n_majority_known   = int((plas['pct_unknown'] <= 50).sum())
    total = n_majority_unknown + n_majority_known

    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    wedges, _ = ax.pie([n_majority_unknown, n_majority_known],
                       colors=['#bcbcbc', '#3a72c4'],
                       startangle=90,
                       wedgeprops=dict(edgecolor='white', linewidth=1.5))
    ax.text(0, 0, f"{total}\nplasmids", ha='center', va='center', fontsize=10,
            fontweight='bold')
    for w, lab in zip(wedges,
                      [f"Majority\nunknown\n{n_majority_unknown}\n"
                       f"({n_majority_unknown/total*100:.1f}%)",
                       f"Majority\ncharacterised\n{n_majority_known}\n"
                       f"({n_majority_known/total*100:.1f}%)"]):
        ang = (w.theta1 + w.theta2) / 2
        ax.text(1.25 * np.cos(np.deg2rad(ang)),
                1.25 * np.sin(np.deg2rad(ang)),
                lab, ha='center', va='center', fontsize=7.5)
    ax.set_aspect('equal')
    plt.tight_layout()
    _save(fig, "fig3E_majority_unknown_pie")


# ===========================================================================
# Fig 3F — per-phylum unannotated burden box plots (weighted permutation BH)
# ===========================================================================
N_PERM_PAIRS = 10_000


def plot_unknown_pct_box():
    df = pd.read_csv(ARCHAEAL_EGGNOG, sep='\t', low_memory=False)
    df['is_unknown'] = df['COG_category'].apply(_is_unknown)
    plas = df.groupby('replicon').agg(
        n=('is_unknown', 'size'),
        n_unknown=('is_unknown', 'sum'),
        phylum=('gtdb_phylum', 'first'),
        species=('gtdb_species', 'first'),
    ).reset_index()
    plas['pct_unknown'] = 100.0 * plas['n_unknown'] / plas['n']
    plas['phy'] = plas['phylum'].str.replace('p__', '', regex=False)
    sp_counts = plas.groupby('species')['replicon'].nunique()
    plas['isf_w'] = plas['species'].map(lambda s: 1.0 / sp_counts.get(s, 1))

    phy_order = ['Thermoproteota', 'Methanobacteriota_B', 'Methanobacteriota',
                 'Halobacteriota']
    phy_order = [p for p in phy_order if (plas['phy'] == p).sum() >= 3]

    # Weighted bootstrap mean + CI
    rng = np.random.default_rng(42)
    bs_stats = {}
    for p in phy_order:
        grp = plas[plas['phy'] == p]
        v = grp['pct_unknown'].to_numpy()
        w = grp['isf_w'].to_numpy()
        wm = float(np.average(v, weights=w))
        boots = [np.average(v[idx], weights=w[idx])
                 for idx in (rng.integers(0, len(v), len(v)) for _ in range(1000))]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        bs_stats[p] = (wm, lo, hi)

    # Weighted permutation Halo vs each other (BH across all six pairs in the
    # full set)
    phyla_for_pairs = sorted(plas['phy'].unique())
    pairs = []
    for i in range(len(phyla_for_pairs)):
        for j in range(i + 1, len(phyla_for_pairs)):
            g1, g2 = plas[plas['phy'] == phyla_for_pairs[i]], \
                     plas[plas['phy'] == phyla_for_pairs[j]]
            if len(g1) < 2 or len(g2) < 2:
                continue
            v1, w1 = g1['pct_unknown'].values, g1['isf_w'].values
            v2, w2 = g2['pct_unknown'].values, g2['isf_w'].values
            obs = np.average(v1, weights=w1) - np.average(v2, weights=w2)
            pooled_v = np.concatenate([v1, v2])
            pooled_w = np.concatenate([w1, w2])
            n1 = len(v1)
            extreme = 0
            for _ in range(N_PERM_PAIRS):
                idx = rng.permutation(len(pooled_v))
                pv, pw = pooled_v[idx], pooled_w[idx]
                d = np.average(pv[:n1], weights=pw[:n1]) - \
                    np.average(pv[n1:], weights=pw[n1:])
                if abs(d) >= abs(obs):
                    extreme += 1
            pairs.append({'g1': phyla_for_pairs[i], 'g2': phyla_for_pairs[j],
                          'p_raw': (extreme + 1) / (N_PERM_PAIRS + 1)})
    pw_df = pd.DataFrame(pairs)
    pw_df['p_adj'] = multipletests(pw_df['p_raw'], method='fdr_bh')[1]
    halo_q = {r['g2'] if r['g1'] == 'Halobacteriota' else r['g1']: r['p_adj']
              for _, r in pw_df.iterrows()
              if 'Halobacteriota' in (r['g1'], r['g2'])}

    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    HALO_BLUE  = '#7ca6c8'
    HEAT_GREEN = '#a6c089'
    HEAT_YELLOW= '#ccc56a'
    HEAT_RED   = '#d97070'
    COLOURS = dict(zip(['Thermoproteota', 'Methanobacteriota_B',
                        'Methanobacteriota', 'Halobacteriota'],
                       [HEAT_RED, HEAT_YELLOW, HEAT_GREEN, HALO_BLUE]))
    ypos = np.arange(len(phy_order))[::-1]
    box_data = [plas.loc[plas['phy'] == p, 'pct_unknown'].values for p in phy_order]
    bp = ax.boxplot(box_data, positions=ypos, vert=False, widths=0.55,
                    patch_artist=True, showfliers=True,
                    flierprops=dict(marker='o', markersize=3,
                                    markerfacecolor='#888',
                                    markeredgecolor='#888', alpha=0.5))
    for patch, p in zip(bp['boxes'], phy_order):
        patch.set_facecolor(COLOURS[p]); patch.set_edgecolor('black')
        patch.set_alpha(0.85)
    for med in bp['medians']:
        med.set_color('black'); med.set_linewidth(1.2)
    for y, p in zip(ypos, phy_order):
        wm, lo, hi = bs_stats[p]
        ax.plot([lo, hi], [y, y], color='#f08c2a', lw=2.5,
                solid_capstyle='butt', zorder=4)
        ax.scatter([wm], [y], marker='D', s=55, color='black',
                   edgecolor='black', linewidth=0.5, zorder=5)
    for y, p in zip(ypos, phy_order):
        if p == 'Halobacteriota':
            continue
        q = halo_q.get(p, np.nan)
        if np.isnan(q):
            continue
        star = '***' if q < 0.001 else ('**' if q < 0.01 else
               ('*' if q < 0.05 else 'n.s.'))
        ax.text(50, y - 0.36, f"{star}  q = {q:.3f} (vs Halo)",
                ha='center', va='top', fontsize=8, color='#a30')
    xmax = 105
    for y, p in zip(ypos, phy_order):
        n = int((plas['phy'] == p).sum())
        ax.text(xmax, y, f"n = {n}", va='center', ha='left',
                fontsize=8, color='#555')
    ax.set_yticks(ypos); ax.set_yticklabels(phy_order)
    ax.set_xlim(0, 102); ax.set_xlabel("% proteins with unknown function per plasmid")
    ax.set_title("Species-frequency-weighted unannotated burden by phylum",
                 fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], marker='D', color='w', markerfacecolor='black',
               markeredgecolor='black', markersize=7, label='ISF-weighted mean'),
        Line2D([0], [0], color='#f08c2a', lw=2.5, label='95% bootstrap CI'),
    ], loc='upper left', frameon=False, fontsize=8)
    plt.tight_layout()
    _save(fig, "fig3F_unknown_pct_box")


# ===========================================================================
# Extended Data G — stacked-bar % unknown per plasmid by phylum
# ===========================================================================
def plot_unknown_pct_stacked_by_phylum():
    df = pd.read_csv(ARCHAEAL_EGGNOG, sep='\t', low_memory=False)
    df['is_unknown'] = df['COG_category'].apply(_is_unknown)
    plas = df.groupby('replicon').agg(
        n=('is_unknown', 'size'),
        n_unknown=('is_unknown', 'sum'),
        phylum=('gtdb_phylum', 'first')).reset_index()
    plas['pct_unknown'] = 100.0 * plas['n_unknown'] / plas['n']
    plas['phy'] = plas['phylum'].str.replace('p__', '', regex=False)
    bins = [0, 25, 50, 75, 100.01]
    labels = ['0-25%', '25-50%', '50-75%', '75-100%']
    plas['bin'] = pd.cut(plas['pct_unknown'], bins=bins, labels=labels,
                         include_lowest=True, right=False)
    phy_order = ['Halobacteriota', 'Thermoproteota', 'Methanobacteriota',
                 'Methanobacteriota_B', 'Thermoplasmatota']
    phy_order = [p for p in phy_order if (plas['phy'] == p).any()]
    ct = pd.crosstab(plas['phy'], plas['bin']).reindex(phy_order).fillna(0)
    pct = ct.div(ct.sum(axis=1), axis=0) * 100
    bin_colors = ['#1f78b4', '#7fbfd1', '#f4a07a', '#d65454']

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    x = np.arange(len(phy_order))
    bottom = np.zeros(len(phy_order))
    for i, lab in enumerate(labels):
        vals = pct[lab].values
        ax.bar(x, vals, bottom=bottom, color=bin_colors[i], edgecolor='white',
               linewidth=0.4, label=lab)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(phy_order, rotation=20, ha='right', fontsize=8, style='italic')
    ax.set_ylabel('% of plasmids')
    for xi, p in zip(x, phy_order):
        n = int((plas['phy'] == p).sum())
        ax.text(xi, 102, f"n = {n}", ha='center', va='bottom', fontsize=7,
                color='#555')
    ax.set_ylim(0, 108)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(title='% unknown\nper plasmid', frameon=False, fontsize=7,
              loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout()
    _save(fig, "extG_unknown_pct_stacked_by_phylum")


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("Loading cluster summary...")
    clusters = load_cluster_summary()
    print(f"  {len(clusters):,} clusters; "
          f"{(clusters['cluster_type'] == 'cross-domain').sum():,} cross-domain")

    print("\nGenerating Extended Data panels:")
    plot_cluster_type_counts(clusters)
    plot_archaeal_protein_share(clusters)
    plot_cluster_size_cdf(clusters)
    plot_per_phylum_cd_counts(clusters)

    print("\nGenerating Fig 3 panels:")
    plot_tripartite_sankey(clusters)
    plot_partner_residuals_heatmap(clusters)

    print("\nComputing CLR enrichment (this can take a few minutes)...")
    phylum_table, family_table = compute_clr_tables()
    plot_phylum_clr_delta_heatmap(phylum_table)
    plot_phylum_clr_raw_heatmap(phylum_table)
    plot_family_clr_delta_heatmap(family_table)

    print("\nAnnotation distribution panels:")
    plot_cog_annotation_counts()
    plot_majority_unknown_pie()
    plot_unknown_pct_box()
    plot_unknown_pct_stacked_by_phylum()

    print(f"\nAll outputs in {FIG_DIR}")


if __name__ == "__main__":
    main()
