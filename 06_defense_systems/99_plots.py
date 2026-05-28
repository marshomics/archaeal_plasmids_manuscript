#!/usr/bin/env python3
"""Generate every panel for the defence-system and CRISPR figure set.

Outputs go to ``outputs/figures/`` next to this script. Every quantity is
re-derived from the input data and the other scripts in this pipeline; no
numbers are hard-coded. Each panel is written as both PNG (raster) and SVG
(``svg.fonttype = 'none'`` so labels stay editable in vector software).

Run as ``python 99_plots.py``. Several of the slower analyses (cooccurrence
permutation, weighted defence-vs-VirB4 GLM, per-array CRISPR permutation,
per-array Wilcoxon enrichment) are cached to ``outputs/`` by the numbered
analysis scripts; ``99_plots.py`` falls back to those caches when present
and recomputes them otherwise.
"""
from collections import defaultdict
from pathlib import Path
import sys

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.interpolate import UnivariateSpline
from statsmodels.stats.multitest import multipletests
import networkx as nx
from sklearn.metrics import roc_auc_score  # noqa: only here so the import
                                            # graph mirrors the rest of the
                                            # pipeline; not used directly.

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (BLAST_TSV, CONJ_TXT, MOB_FILE, OUT_DIR, SEQ_SPACE,
                    SPACER_FASTA, SUB_FILE, header, load_defense_tables,
                    N_PERM, SEED)  # noqa


FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
matplotlib.rcParams['font.family']  = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']


PHYLUM_ORDER = ['Halobacteriota', 'Methanobacteriota', 'Methanobacteriota_B',
                'Thermoplasmatota', 'Thermoproteota']
PHYLUM_COLOR = {
    'Halobacteriota':      '#7ca6c8',
    'Methanobacteriota':   '#e0a065',
    'Methanobacteriota_B': '#a6c089',
    'Thermoplasmatota':    '#9467bd',
    'Thermoproteota':      '#ccc56a',
}

DEFENSE_CATEGORY = {
    'RM':       'Restriction-Modification',
    'DISARM':   'Restriction-Modification',
    'BREX':     'Restriction-Modification',
    'Cas':      'CRISPR-Cas',
    'pAgo':     'Nucleic acid degradation',
    'CBASS':    'Signaling',
    'Tiamat':   'Signaling',
    'AbiE':     'Abortive infection',
    'SoFIC':    'Abortive infection',
    'HEC-05':   'Other',
    'Ceres':    'Other',
    'Shango':   'Other',
    'Hachiman': 'Other',
    'Druantia': 'Other',
    'DS-11':    'Other',
    'Prometheus':'Other',
}
CATEGORY_COLOR = {
    'Restriction-Modification': '#e63946',
    'CRISPR-Cas':               '#1f77b4',
    'Nucleic acid degradation': '#2ca02c',
    'Signaling':                '#ff7f0e',
    'Abortive infection':       '#9467bd',
    'Other':                    '#999999',
}


def _save(fig, stem):
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=200, bbox_inches='tight')
    fig.savefig(FIG_DIR / f"{stem}.svg", bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {stem}.png + .svg")


def _short(s):
    return s.split('__', 1)[-1] if isinstance(s, str) and '__' in s else s


# ===========================================================================
# Fig 6A — distribution of total defence instances per plasmid
# ===========================================================================
def plot_total_defense_hist(type_df):
    vals = type_df['n_instances'].astype(int).values
    mean, median = vals.mean(), float(np.median(vals))
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    bins = np.arange(vals.max() + 2) - 0.5
    ax.hist(vals, bins=bins, color='#3a72c4', edgecolor='white', linewidth=0.5)
    ax.axvline(median, color='#2c7a3e', linestyle=':', linewidth=1.5,
               label=f"Median = {median:.0f}")
    ax.axvline(mean, color='#c0223b', linestyle='--', linewidth=1.5,
               label=f"Mean = {mean:.1f}")
    ax.set_xlabel('Total defense system\ninstances per plasmid')
    ax.set_ylabel('Number of plasmids')
    ax.legend(frameon=False, fontsize=8, loc='upper right')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    _save(fig, "fig6A_total_defense_hist")


# ===========================================================================
# Fig 6B — distribution of distinct defence types per plasmid
# ===========================================================================
def plot_type_richness_hist(type_df):
    vals = type_df['n_types'].astype(int).values
    mean, median = vals.mean(), float(np.median(vals))
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    bins = np.arange(vals.max() + 2) - 0.5
    ax.hist(vals, bins=bins, color='#a6c089', edgecolor='white', linewidth=0.5)
    ax.axvline(median, color='#3a72c4', linestyle=':', linewidth=1.5,
               label=f"Median = {median:.0f}")
    ax.axvline(mean, color='#c0223b', linestyle='--', linewidth=1.5,
               label=f"Mean = {mean:.1f}")
    ax.set_xlabel('Number of distinct defense\nsystem types per plasmid')
    ax.set_ylabel('Number of plasmids')
    ax.legend(frameon=False, fontsize=8, loc='upper right')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    _save(fig, "fig6B_type_richness_hist")


# ===========================================================================
# Fig 6C — per-phylum defence-count box plot
# ===========================================================================
def plot_per_phylum_box(type_df):
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    data = []
    labels = []
    colours = []
    for phy in PHYLUM_ORDER:
        sub = type_df[type_df['phylum'] == phy]
        if len(sub) == 0:
            continue
        data.append(sub['n_instances'].values)
        labels.append(phy)
        colours.append(PHYLUM_COLOR[phy])
    bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=True,
                    medianprops=dict(color='#f08c2a', linewidth=1.4),
                    flierprops=dict(marker='o', markersize=4,
                                    markerfacecolor='white',
                                    markeredgecolor='#666', alpha=0.8))
    for patch, col in zip(bp['boxes'], colours):
        patch.set_facecolor(col); patch.set_edgecolor('black')
        patch.set_alpha(0.85)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=8,
                       style='italic')
    ax.set_ylabel('Total defense systems\nper plasmid')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    _save(fig, "fig6C_per_phylum_box")


# ===========================================================================
# Fig 6D — top-10 defence-type prevalence (TYPE level, not subtype)
# ===========================================================================
def plot_type_prevalence(type_df, type_cols):
    n_plas = len(type_df)
    counts = (type_df[type_cols] > 0).sum().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    y = np.arange(len(counts))[::-1]
    ax.barh(y, 100 * counts.values / n_plas, color='#7ca6c8',
            edgecolor='white', linewidth=0.4)
    ax.set_yticks(y); ax.set_yticklabels(counts.index, fontsize=9)
    ax.set_xlabel('Prevalence (%)')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    _save(fig, "fig6D_type_prevalence")


# ===========================================================================
# Fig 6E — defence-system co-occurrence network
# ===========================================================================
def _cooccurrence_table(binary_type, type_cols):
    """Recompute the weighted permutation table (matches the analysis script)."""
    active = [c for c in type_cols if (binary_type[c] > 0).sum() >= 3]
    weights = binary_type['weight'].values
    rng = np.random.default_rng(SEED)
    perm_idx = np.array([rng.permutation(len(binary_type))
                         for _ in range(N_PERM)])
    rows = []
    n_pairs = len(active) * (len(active) - 1) // 2
    for i in range(len(active)):
        a = binary_type[active[i]].values
        for j in range(i + 1, len(active)):
            b = binary_type[active[j]].values
            w_sum = weights.sum()
            pa = (a * weights).sum() / w_sum
            pb = (b * weights).sum() / w_sum
            obs = (a * b * weights).sum() / w_sum - pa * pb
            null = []
            for k in range(N_PERM):
                a_perm = a[perm_idx[k]]
                null.append((a_perm * b * weights).sum() / w_sum -
                            (a_perm * weights).sum() / w_sum * pb)
            null = np.array(null)
            p = (np.abs(null) >= abs(obs)).sum() / N_PERM
            rows.append({
                'system_a': active[i], 'system_b': active[j],
                'obs_excess': obs, 'direction': 'enriched' if obs > 0 else 'depleted',
                'perm_p_value': p,
                'n_cooccur': int(((a > 0) & (b > 0)).sum()),
            })
    df = pd.DataFrame(rows)
    df['p_adjusted'] = multipletests(df['perm_p_value'], method='fdr_bh')[1]
    df['significant'] = df['p_adjusted'] < 0.05
    return df


def plot_cooccurrence_network(binary_type, type_cols):
    cache = OUT_DIR / "cooccurrence_weighted_perm.csv"
    coocc = pd.read_csv(cache) if cache.exists() else _cooccurrence_table(
        binary_type, type_cols)

    sig_edges = coocc[coocc['significant']]
    nominal = coocc[(coocc['perm_p_value'] < 0.05) & (~coocc['significant'])]
    nodes = sorted(set(coocc['system_a']) | set(coocc['system_b']))

    G = nx.Graph()
    for n in nodes:
        G.add_node(n)
    for _, r in nominal.iterrows():
        G.add_edge(r['system_a'], r['system_b'], kind='nominal',
                   obs=r['obs_excess'], p=r['p_adjusted'])
    for _, r in sig_edges.iterrows():
        G.add_edge(r['system_a'], r['system_b'], kind='sig',
                   obs=r['obs_excess'], p=r['p_adjusted'])

    # Layout: kamada-kawai over largest connected component, ring for isolates
    np.random.seed(SEED)
    components = list(nx.connected_components(G))
    biggest = max(components, key=len)
    sub = G.subgraph(biggest)
    inner = nx.kamada_kawai_layout(sub)
    xs = np.array([p[0] for p in inner.values()])
    ys = np.array([p[1] for p in inner.values()])
    if np.ptp(xs) > 0: xs = 2 * (xs - xs.min()) / np.ptp(xs) - 1
    if np.ptp(ys) > 0: ys = 2 * (ys - ys.min()) / np.ptp(ys) - 1
    pos = {n: np.array([x * 0.65, y * 0.65])
           for (n, _), x, y in zip(inner.items(), xs, ys)}
    isolates = [n for n in nodes if n not in pos]
    for i, n in enumerate(isolates):
        ang = 2 * np.pi * i / max(len(isolates), 1) + np.pi / 6
        pos[n] = np.array([1.2 * np.cos(ang), 1.2 * np.sin(ang)])

    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    sig_pairs = [(u, v) for u, v, d in G.edges(data=True) if d['kind'] == 'sig']
    nom_pairs = [(u, v) for u, v, d in G.edges(data=True) if d['kind'] == 'nominal']
    nx.draw_networkx_edges(G, pos, edgelist=nom_pairs, edge_color='#bcbcbc',
                           style='dashed', width=1.0, alpha=0.75, ax=ax)
    sig_widths = [max(3.0, 120 * d['obs'])
                  for _, _, d in G.edges(data=True) if d['kind'] == 'sig']
    nx.draw_networkx_edges(G, pos, edgelist=sig_pairs, edge_color='#e63946',
                           width=sig_widths, alpha=0.95, ax=ax)
    node_colors = [CATEGORY_COLOR[DEFENSE_CATEGORY.get(n, 'Other')] for n in nodes]
    sig_nodes = set(u for u, v in sig_pairs) | set(v for u, v in sig_pairs)
    sizes = [700 if n in sig_nodes else 450 for n in nodes]
    nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=node_colors,
                           node_size=sizes, edgecolors='white', linewidths=1.5,
                           ax=ax)
    for n in nodes:
        x, y = pos[n]
        ax.text(x, y + 0.075, n, ha='center', va='bottom', fontsize=8,
                fontweight='bold' if n in sig_nodes else 'normal')
    for _, r in sig_edges.iterrows():
        a, b = r['system_a'], r['system_b']
        mx, my = (pos[a][0] + pos[b][0]) / 2, (pos[a][1] + pos[b][1]) / 2
        ax.annotate(f"weighted excess = {r['obs_excess']:.3f}\n"
                    f"BH q = {r['p_adjusted']:.4f}",
                    xy=(mx, my), xytext=(mx + 0.35, my - 0.25),
                    fontsize=7.5, color='#a00',
                    arrowprops=dict(arrowstyle='-', color='#a00', lw=0.6),
                    bbox=dict(facecolor='white', edgecolor='#e63946',
                              boxstyle='round,pad=0.3', alpha=0.95))
    from matplotlib.lines import Line2D
    cat_handles = [mpatches.Patch(facecolor=c, edgecolor='white', label=cat)
                   for cat, c in CATEGORY_COLOR.items()]
    edge_handles = [
        Line2D([0], [0], color='#e63946', lw=3, label='BH q < 0.05 (FDR)'),
        Line2D([0], [0], color='#bcbcbc', lw=1.2, linestyle='--',
               label='p < 0.05 (nominal)'),
    ]
    ax.legend(handles=cat_handles + edge_handles, fontsize=7, frameon=False,
              loc='center left', bbox_to_anchor=(1.0, 0.5),
              title='Defense system category')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.4, 1.4)
    ax.set_axis_off()
    plt.tight_layout()
    _save(fig, "fig6E_cooccurrence_network")


# ===========================================================================
# Fig 6F — defence count vs plasmid size scatter with cubic spline
# ===========================================================================
def plot_defense_vs_size(type_df):
    mob = pd.read_csv(MOB_FILE, sep='\t')
    sizes = mob[['sample_id', 'size']].rename(
        columns={'sample_id': 'replicon', 'size': 'size_bp'})
    df = type_df.merge(sizes, on='replicon', how='inner')
    df = df[df['size_bp'] > 0]
    df['size_kb'] = df['size_bp'] / 1000

    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    for phy, col in PHYLUM_COLOR.items():
        sub = df[df['phylum'] == phy]
        ax.scatter(sub['size_kb'], sub['n_instances'], s=10, color=col,
                   alpha=0.55, edgecolor='white', linewidth=0.2,
                   label=phy)
    # Cubic spline through medians per log-binned size
    bins = np.logspace(np.log10(df['size_kb'].min()),
                       np.log10(df['size_kb'].max()), 20)
    df['size_bin'] = pd.cut(df['size_kb'], bins=bins)
    binned = (df.groupby('size_bin', observed=True)
                 .agg(median_n=('n_instances', 'median'),
                      mean_size=('size_kb', 'mean'))
                 .dropna())
    if len(binned) >= 4:
        spline = UnivariateSpline(np.log10(binned['mean_size']),
                                  binned['median_n'], k=3, s=2)
        xs = np.logspace(np.log10(df['size_kb'].min()),
                         np.log10(df['size_kb'].max()), 200)
        ax.plot(xs, spline(np.log10(xs)), color='black', linewidth=1.2,
                label='Cubic spline')
    ax.set_xscale('log')
    ax.set_xlabel('Plasmid size (kb)')
    ax.set_ylabel('Defense system count')
    ax.legend(frameon=False, fontsize=7, loc='upper left')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    _save(fig, "fig6F_defense_vs_size")


# ===========================================================================
# Fig 6G / 6H — defence burden / type richness vs VirB4-T4CP (Halo only),
# size-adjusted IRR from the cached NB-GLM if available
# ===========================================================================
def _glm_results():
    """Read the species-weighted NB-GLM table from disk or recompute it."""
    cache = OUT_DIR / "virb4_defense_glm.csv"
    if cache.exists():
        return pd.read_csv(cache).set_index('response')

    sub_df = pd.read_csv(SUB_FILE, sep='\t')
    TAX_COLS = ['replicon', 'gtdb_phylum', 'gtdb_class', 'gtdb_order',
                'gtdb_family', 'gtdb_genus', 'gtdb_species']
    defense_cols = [c for c in sub_df.columns if c not in TAX_COLS]
    for c in defense_cols:
        sub_df[c] = pd.to_numeric(sub_df[c], errors='coerce').fillna(0).astype(int)
    mob = pd.read_csv(MOB_FILE, sep='\t')
    sizes = mob[['sample_id', 'size']].rename(
        columns={'sample_id': 'replicon', 'size': 'size_bp'})
    with open(CONJ_TXT) as f:
        conj = {ln.strip() for ln in f if ln.strip()}
    df = sub_df.merge(sizes, on='replicon', how='inner')
    df['is_conjugative'] = df['replicon'].isin(conj).astype(int)
    df['defense_burden'] = df[defense_cols].sum(axis=1)
    df['type_richness']  = (df[defense_cols] > 0).sum(axis=1)
    df['log10_size']     = np.log10(df['size_bp'])
    df['species']        = df['gtdb_species'].str.replace('s__', '', regex=False)
    df = df[df['gtdb_phylum'] == 'p__Halobacteriota'].copy()
    df = df[df['size_bp'] > 0]
    sc = df['species'].value_counts()
    df['w'] = 1.0 / df['species'].map(sc); df['w'] = df['w'] / df['w'].sum() * len(df)

    X = sm.add_constant(df[['log10_size', 'is_conjugative']].values)
    results = {}
    for resp in ['defense_burden', 'type_richness']:
        y = df[resp].values
        nb_unw = sm.NegativeBinomial(y, X).fit(disp=False, maxiter=500)
        alpha = float(np.exp(nb_unw.params[-1]))
        glm = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=alpha),
                     freq_weights=df['w'].values).fit()
        beta = glm.params[2]; se = glm.bse[2]
        results[resp] = {
            'conj_IRR':   float(np.exp(beta)),
            'conj_CI_lo': float(np.exp(beta - 1.96 * se)),
            'conj_CI_hi': float(np.exp(beta + 1.96 * se)),
            'conj_P':     float(glm.pvalues[2]),
        }
    return pd.DataFrame(results).T.rename_axis('response')


def _violin_plot(df_halo, value_col, irr, ci_lo, ci_hi, p, stem, title):
    pos = df_halo.loc[df_halo['is_conjugative'] == 1, value_col].values
    neg = df_halo.loc[df_halo['is_conjugative'] == 0, value_col].values

    fig, ax = plt.subplots(figsize=(3.4, 3.6))
    COL_POS, COL_NEG = '#e69138', '#6fa8dc'
    parts = ax.violinplot([pos, neg], positions=[1, 2], showmeans=False,
                          showmedians=False, showextrema=False, widths=0.7)
    for body, c in zip(parts['bodies'], (COL_POS, COL_NEG)):
        body.set_facecolor(c); body.set_edgecolor('none'); body.set_alpha(0.45)
    bp = ax.boxplot([pos, neg], positions=[1, 2], widths=0.25,
                    patch_artist=True, showfliers=False)
    for patch, c in zip(bp['boxes'], (COL_POS, COL_NEG)):
        patch.set_facecolor(c); patch.set_edgecolor('black'); patch.set_alpha(0.9)
    for med in bp['medians']:
        med.set_color('black'); med.set_linewidth(1.4)
    rng = np.random.default_rng(SEED)
    for vals, xc, col in [(pos, 1, COL_POS), (neg, 2, COL_NEG)]:
        x = xc + rng.normal(0, 0.04, size=len(vals))
        ax.scatter(x, vals, s=6, color=col, alpha=0.25, edgecolor='none')
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['VirB4-T4CP+', 'VirB4-T4CP−'])
    ax.set_ylabel('Size-adjusted count')
    ax.set_title(f"IRR = {irr:.2f} [{ci_lo:.2f}, {ci_hi:.2f}]\n"
                 f"p = {p:.2e}", fontsize=10)
    y_top = max(np.max(pos), np.max(neg)) * 1.05
    ax.plot([1, 1, 2, 2], [y_top * 0.97, y_top, y_top, y_top * 0.97],
            color='black', lw=0.8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylim(bottom=-0.3)
    plt.tight_layout()
    _save(fig, stem)


def plot_virb4_burden_and_richness():
    glm = _glm_results()
    sub_df = pd.read_csv(SUB_FILE, sep='\t')
    TAX_COLS = ['replicon', 'gtdb_phylum', 'gtdb_class', 'gtdb_order',
                'gtdb_family', 'gtdb_genus', 'gtdb_species']
    defense_cols = [c for c in sub_df.columns if c not in TAX_COLS]
    for c in defense_cols:
        sub_df[c] = pd.to_numeric(sub_df[c], errors='coerce').fillna(0).astype(int)
    mob = pd.read_csv(MOB_FILE, sep='\t')
    sizes = mob[['sample_id', 'size']].rename(
        columns={'sample_id': 'replicon', 'size': 'size_bp'})
    with open(CONJ_TXT) as f:
        conj = {ln.strip() for ln in f if ln.strip()}
    df = sub_df.merge(sizes, on='replicon', how='inner')
    df['is_conjugative'] = df['replicon'].isin(conj).astype(int)
    df['defense_burden'] = df[defense_cols].sum(axis=1)
    df['type_richness']  = (df[defense_cols] > 0).sum(axis=1)
    df = df[(df['gtdb_phylum'] == 'p__Halobacteriota') & (df['size_bp'] > 0)]

    r = glm.loc['defense_burden']
    _violin_plot(df, 'defense_burden', r['conj_IRR'], r['conj_CI_lo'],
                 r['conj_CI_hi'], r['conj_P'],
                 "fig6G_burden_virb4_violin", 'Defense burden')
    r = glm.loc['type_richness']
    _violin_plot(df, 'type_richness', r['conj_IRR'], r['conj_CI_lo'],
                 r['conj_CI_hi'], r['conj_P'],
                 "fig6H_richness_virb4_violin", 'Type richness')


# ===========================================================================
# Fig 6I — CRISPR-targeting fraction (Virus vs Plasmid, expected vs observed)
# ===========================================================================
def plot_crispr_target_fraction():
    seq = pd.read_csv(SEQ_SPACE, sep='\t')
    plasmid_kb = float(seq.loc[seq['category'] == 'plasmid', 'total_kb'].values[0])
    virus_kb   = float(seq.loc[seq['category'] == 'virus',   'total_kb'].values[0])
    total = plasmid_kb + virus_kb
    exp_plasmid = plasmid_kb / total * 100
    exp_virus   = virus_kb   / total * 100

    hits = pd.read_csv(BLAST_TSV, sep='\t')
    plas_hits  = (hits['target_category'] == 'plasmid').sum()
    virus_hits = (hits['target_category'] == 'virus').sum()
    obs_total  = plas_hits + virus_hits
    obs_plasmid = plas_hits / obs_total * 100
    obs_virus   = virus_hits / obs_total * 100

    fig, ax = plt.subplots(figsize=(4.0, 3.8))
    x_groups = ['Virus', 'Plasmid']
    w = 0.35
    x = np.arange(len(x_groups))
    ax.bar(x - w/2, [exp_virus, exp_plasmid], width=w, color='#cccccc',
           edgecolor='white', linewidth=0.5, label='Expected (from db size)')
    ax.bar(x + w/2, [obs_virus, obs_plasmid], width=w, color='#3a72c4',
           edgecolor='white', linewidth=0.5, label='Observed')
    for xi, exp, obs in zip(x, [exp_virus, exp_plasmid],
                            [obs_virus, obs_plasmid]):
        ax.text(xi - w/2, exp + 1.5, f"{exp:.1f}%", ha='center',
                va='bottom', fontsize=8)
        ax.text(xi + w/2, obs + 1.5, f"{obs:.1f}%", ha='center',
                va='bottom', fontsize=8)
    # Chi-squared on the pooled 2x2 against the proportional null
    chi2, p_chi = stats.chisquare(
        f_obs=[virus_hits, plas_hits],
        f_exp=[obs_total * virus_kb / total,
               obs_total * plasmid_kb / total])[:2]
    star = '***' if p_chi < 0.001 else ('**' if p_chi < 0.01 else
           ('*' if p_chi < 0.05 else 'n.s.'))
    ax.text(0.5, max(obs_virus, obs_plasmid) * 1.10, star, ha='center',
            fontsize=14, color='black')
    ax.plot([x[0], x[0], x[1], x[1]],
            [max(obs_virus, obs_plasmid) * 1.04,
             max(obs_virus, obs_plasmid) * 1.07,
             max(obs_virus, obs_plasmid) * 1.07,
             max(obs_virus, obs_plasmid) * 1.04], color='black', lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels(x_groups)
    ax.set_ylabel('Fraction (%)')
    ax.set_ylim(0, 115)
    ax.legend(frameon=False, fontsize=8, loc='upper center')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    _save(fig, "fig6I_crispr_target_fraction")


# ===========================================================================
# Fig 6J — per-kb hits enrichment (Virus vs Plasmid)
# ===========================================================================
def plot_crispr_per_kb():
    seq = pd.read_csv(SEQ_SPACE, sep='\t')
    plasmid_kb = float(seq.loc[seq['category'] == 'plasmid', 'total_kb'].values[0])
    virus_kb   = float(seq.loc[seq['category'] == 'virus',   'total_kb'].values[0])
    hits = pd.read_csv(BLAST_TSV, sep='\t')
    plas  = (hits['target_category'] == 'plasmid').sum()
    virus = (hits['target_category'] == 'virus').sum()
    plas_per_kb  = plas  / plasmid_kb * 1e6  # per Mb
    virus_per_kb = virus / virus_kb   * 1e6
    ratio = plas_per_kb / virus_per_kb if virus_per_kb > 0 else float('inf')

    fig, ax = plt.subplots(figsize=(3.5, 3.6))
    ax.bar([0], [virus_per_kb], color='#cccccc', edgecolor='white',
           linewidth=0.5, width=0.6, label='Virus')
    ax.bar([1], [plas_per_kb], color='#3a72c4', edgecolor='white',
           linewidth=0.5, width=0.6, label='Plasmid')
    ax.text(0, virus_per_kb + plas_per_kb * 0.02, f"{virus_per_kb:.1f}",
            ha='center', va='bottom', fontsize=9)
    ax.text(1, plas_per_kb + plas_per_kb * 0.02, f"{plas_per_kb:.0f}",
            ha='center', va='bottom', fontsize=9)
    # bracket
    y_top = plas_per_kb * 1.10
    ax.plot([0, 0, 1, 1], [y_top * 0.97, y_top, y_top, y_top * 0.97],
            color='black', lw=0.7)
    ax.text(0.5, y_top * 1.01, f"{ratio:.0f}× plasmid\nenrichment",
            ha='center', va='bottom', fontsize=9)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Virus', 'Plasmid'])
    ax.set_ylabel('Hits per Mb\n(normalised)')
    ax.set_ylim(0, plas_per_kb * 1.30)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    _save(fig, "fig6J_crispr_per_kb")


# ===========================================================================
# Extended Data A — subtype prevalence + RM and CRISPR-Cas subtype pies
# ===========================================================================
def plot_subtype_panels(type_df):
    cache = OUT_DIR / "subtype_prevalence.csv"
    if cache.exists():
        sub = pd.read_csv(cache)
    else:
        sub_df = pd.read_csv(SUB_FILE, sep='\t')
        TAX = ['replicon', 'gtdb_phylum', 'gtdb_class', 'gtdb_order',
               'gtdb_family', 'gtdb_genus', 'gtdb_species']
        cols = [c for c in sub_df.columns if c not in TAX]
        for c in cols:
            sub_df[c] = pd.to_numeric(sub_df[c], errors='coerce').fillna(0).astype(int)
        sub = (sub_df[cols].gt(0).sum()
                .reset_index().rename(columns={'index': 'subtype', 0: 'count'}))
        sub['instances'] = sub_df[cols].sum().values
    n_plas = len(type_df)
    sub['pct'] = 100 * sub['count'] / n_plas
    top10 = sub.sort_values('count', ascending=False).head(10).reset_index(drop=True)

    fig = plt.figure(figsize=(9.5, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0], hspace=0.45,
                          wspace=0.3)

    # Top: bar of top-10 subtype prevalence
    ax1 = fig.add_subplot(gs[0, :])
    y = np.arange(len(top10))[::-1]
    ax1.barh(y, top10['pct'].values, color='#a6c089', edgecolor='white',
             linewidth=0.4)
    ax1.set_yticks(y); ax1.set_yticklabels(
        [s.replace('_', ' ').replace('CAS Class1-Subtype-', 'CAS Class1 Subtype ')
         for s in top10['subtype']], fontsize=9)
    ax1.set_xlabel('Prevalence (%)')
    ax1.spines[['top', 'right']].set_visible(False)

    # Bottom-left: RM type pie
    ax2 = fig.add_subplot(gs[1, 0])
    rm = sub[sub['subtype'].str.startswith('RM_')].copy()
    rm['short'] = rm['subtype'].str.replace('RM_', '', regex=False).str.replace('_', ' ')
    rm = rm.sort_values('instances', ascending=False)
    total_rm = int(rm['instances'].sum())
    wedges, _ = ax2.pie(rm['instances'], colors=plt.cm.Blues(
        np.linspace(0.9, 0.4, len(rm))), startangle=90,
        wedgeprops=dict(edgecolor='white', linewidth=1))
    for w, lab, n in zip(wedges, rm['short'], rm['instances']):
        ang = (w.theta1 + w.theta2) / 2
        ax2.text(1.15 * np.cos(np.deg2rad(ang)),
                 1.15 * np.sin(np.deg2rad(ang)),
                 f"{n/total_rm*100:.1f}%\n{lab}",
                 ha='center', va='center', fontsize=7)
    ax2.set_title(f"RM System Subtypes\n(n = {total_rm})", fontsize=9)
    ax2.set_aspect('equal')

    # Bottom-right: CRISPR-Cas pie
    ax3 = fig.add_subplot(gs[1, 1])
    cas = sub[sub['subtype'].str.startswith('CAS')].copy()
    cas['short'] = (cas['subtype'].str.replace('CAS_Class1-Subtype-', 'I: ', regex=False)
                                   .str.replace('CAS_Class2-Subtype-', 'II: ', regex=False)
                                   .str.replace('CAS_', '', regex=False))
    cas = cas.sort_values('instances', ascending=False)
    total_cas = int(cas['instances'].sum())
    wedges, _ = ax3.pie(cas['instances'], colors=plt.cm.Oranges(
        np.linspace(0.9, 0.3, len(cas))), startangle=90,
        wedgeprops=dict(edgecolor='white', linewidth=1))
    for w, lab, n in zip(wedges, cas['short'], cas['instances']):
        ang = (w.theta1 + w.theta2) / 2
        ax3.text(1.15 * np.cos(np.deg2rad(ang)),
                 1.15 * np.sin(np.deg2rad(ang)),
                 f"{n/total_cas*100:.1f}%\n{lab}",
                 ha='center', va='center', fontsize=7)
    ax3.set_title(f"CRISPR-Cas Subtypes\n(n = {total_cas})", fontsize=9)
    ax3.set_aspect('equal')

    _save(fig, "extA_subtype_prevalence_and_pies")


# ===========================================================================
# Extended Data B — % plasmids by # defense types, split by VirB4-T4CP
# ===========================================================================
def plot_types_by_virb4(type_df):
    with open(CONJ_TXT) as f:
        conj = {ln.strip() for ln in f if ln.strip()}
    df = type_df.copy()
    df['is_conj'] = df['replicon'].isin(conj).astype(int)
    df['types_cat'] = df['n_types'].apply(
        lambda x: '0' if x == 0 else '1' if x == 1 else '2' if x == 2 else '3+')
    order = ['0', '1', '2', '3+']
    bg = (df[df['is_conj'] == 1]['types_cat'].value_counts(normalize=True)
          .reindex(order, fill_value=0) * 100)
    fg = (df[df['is_conj'] == 0]['types_cat'].value_counts(normalize=True)
          .reindex(order, fill_value=0) * 100)
    pct = pd.DataFrame({'VirB4-T4CP+': bg, 'VirB4-T4CP−': fg})
    color_map = {'0': '#cccccc', '1': '#f4d35e', '2': '#e8807a', '3+': '#a32e2e'}

    fig, ax = plt.subplots(figsize=(3.6, 4.0))
    x = np.arange(2)
    bottom = np.zeros(2)
    for cat in order:
        vals = pct.loc[cat].values
        ax.bar(x, vals, bottom=bottom, color=color_map[cat], edgecolor='white',
               linewidth=0.4, label=cat)
        for xi, b, v in zip(x, bottom, vals):
            if v >= 4:
                ax.text(xi, b + v / 2, f"{int(round(v))}%",
                        ha='center', va='center', fontsize=8, color='white',
                        fontweight='bold')
        bottom += vals
    ax.set_xticks(x); ax.set_xticklabels(['VirB4-T4CP+', 'VirB4-T4CP−'])
    ax.set_ylabel('Percentage of plasmids')
    ax.set_ylim(0, 100)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(title='# Defense types', frameon=False, fontsize=8,
              loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout()
    _save(fig, "extB_types_by_virb4")


# ===========================================================================
# Extended Data C — spacers per CRISPR-bearing plasmid
# ===========================================================================
def plot_spacers_per_plasmid():
    # Parse the spacer FASTA — each header is >plasmid_arrayN__spacerM
    # Keep each array as a separate entry (do NOT collapse to parent plasmid)
    counts = defaultdict(int)
    with open(SPACER_FASTA) as f:
        for ln in f:
            if not ln.startswith('>'):
                continue
            array_id = ln[1:].strip().split('__spacer')[0]
            counts[array_id] += 1
    vals = np.array(list(counts.values()))
    n_arrays = len(counts)
    total_spacers = int(vals.sum())
    median_v = float(np.median(vals))

    # number of arrays with >= 1 BLAST hit
    hits = pd.read_csv(BLAST_TSV, sep='\t')
    arrays_with_hit = hits['source_plasmid'].nunique()

    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    ax.hist(vals, bins=np.arange(0, vals.max() + 8, 5), color='#3a72c4',
            edgecolor='white', linewidth=0.4)
    ax.axvline(median_v, color='gray', linestyle='--', linewidth=1.5)
    ax.text(median_v + vals.max() * 0.02, ax.get_ylim()[1] * 0.92,
            f"median = {median_v:.0f}", color='gray', fontsize=8)
    ax.text(0.98, 0.96,
            f"n = {n_arrays} plasmids\n{total_spacers:,} total spacers\n"
            f"{arrays_with_hit} ({arrays_with_hit/n_arrays*100:.0f}%) "
            f"with ≥ 1 hit",
            transform=ax.transAxes, ha='right', va='top', fontsize=8,
            color='#444')
    ax.set_xlabel('Spacers per plasmid')
    ax.set_ylabel('Number of plasmids')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    _save(fig, "extC_spacers_per_plasmid")


# ===========================================================================
# Extended Data D — observed vs expected within-family CRISPR targeting
# ===========================================================================
def plot_within_family_targeting():
    # CIs aren't cached by the analysis script, so we always recompute.
    (obs, null, p, n_arrays, n_only_same,
     obs_lo, obs_hi, null_lo, null_hi) = _within_family_recompute()
    obs *= 100; null *= 100

    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    bar_colors = ['#bba240', '#9e9e9e']
    positions = [0, 1]
    heights = [obs, null]
    labels = ['Observed', 'Expected\n(per-array null)']
    obs_err = [[(obs - obs_lo)], [(obs_hi - obs)]]
    null_err = [[(null - null_lo)], [(null_hi - null)]]
    ax.bar(positions[0], heights[0], color=bar_colors[0], edgecolor='black',
           linewidth=0.6, width=0.6, yerr=obs_err, capsize=5,
           error_kw=dict(ecolor='black', lw=1.0))
    ax.bar(positions[1], heights[1], color=bar_colors[1], edgecolor='black',
           linewidth=0.6, width=0.6, yerr=null_err, capsize=5,
           error_kw=dict(ecolor='black', lw=1.0))
    for x, h, txt in zip(positions, heights, [f"{obs:.1f}%", f"{null:.1f}%"]):
        ax.text(x, h / 2, txt, ha='center', va='center', color='white',
                fontsize=11, fontweight='bold')
    y_top = max(heights[0] + obs_err[1][0], heights[1] + null_err[1][0]) + 5
    ax.plot([0, 0, 1, 1], [y_top * 0.97, y_top, y_top, y_top * 0.97],
            color='black', lw=0.9)
    star = '***' if p < 0.001 else ('**' if p < 0.01 else
           ('*' if p < 0.05 else 'n.s.'))
    ax.text(0.5, y_top + 1.5, f"{star}  p = {p:.4f} (permutation)",
            ha='center', va='bottom', fontsize=9)
    ax.set_xticks(positions); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean per-array within-family\ntargeting fraction (%)")
    ax.set_ylim(0, 115)
    ax.spines[['top', 'right']].set_visible(False)
    ax.text(0.98, 0.50, f"n = {n_arrays} arrays\n{n_only_same}/{n_arrays} arrays "
            f"target\nonly the same family",
            transform=ax.transAxes, ha='right', va='center', fontsize=7.5,
            color='#444',
            bbox=dict(facecolor='white', edgecolor='lightgray',
                      boxstyle='round,pad=0.4', alpha=0.9))
    plt.tight_layout()
    _save(fig, "extD_within_family_targeting")


def _within_family_recompute():
    """Reproduce the per-array stats from 08_crispr_within_family.py."""
    hits = pd.read_csv(BLAST_TSV, sep='\t')
    ph = hits[hits['target_category'] == 'plasmid'].copy()
    ph['same_family'] = (ph['source_family'] == ph['target_family'])
    per_array = (ph.groupby('source_plasmid')
                   .agg(n_hits=('same_family', 'size'),
                        same_family=('same_family', 'sum'),
                        source_family=('source_family', 'first'))
                   .reset_index())
    per_array['frac_same'] = per_array['same_family'] / per_array['n_hits']
    n_arrays = len(per_array)
    obs_mean = per_array['frac_same'].mean()
    n_only_same = int((per_array['frac_same'] == 1).sum())

    rng = np.random.default_rng(SEED)
    target_families = ph['target_family'].values
    null_means = np.empty(5000)
    for i in range(5000):
        fracs = []
        for _, r in per_array.iterrows():
            draws = rng.choice(target_families, size=int(r['n_hits']),
                               replace=True)
            fracs.append(np.mean(draws == r['source_family']))
        null_means[i] = float(np.mean(fracs))
    p = (np.sum(null_means >= obs_mean) + 1) / (5000 + 1)
    null_mean = null_means.mean()
    null_lo, null_hi = np.percentile(null_means, [2.5, 97.5])
    # Bootstrap CI on observed mean
    rng2 = np.random.default_rng(2024)
    boots = [per_array['frac_same'].sample(
                n_arrays, replace=True,
                random_state=int(rng2.integers(0, 10**9))).mean()
             for _ in range(2000)]
    obs_lo, obs_hi = np.percentile(boots, [2.5, 97.5])
    return (obs_mean, null_mean, p, n_arrays, n_only_same,
            obs_lo * 100, obs_hi * 100, null_lo * 100, null_hi * 100)


# ===========================================================================
# Extended Data E — VirB4-T4CP+/- percentages in Database vs Targeted
# ===========================================================================
def plot_virb4_database_vs_targeted():
    hits = pd.read_csv(BLAST_TSV, sep='\t')
    mob  = pd.read_csv(MOB_FILE, sep='\t')
    with open(CONJ_TXT) as f:
        conj = {ln.strip() for ln in f if ln.strip()}

    all_ids = set(mob['sample_id'])
    targets = set(hits.loc[hits['target_category'] == 'plasmid',
                           'target_plasmid'].dropna())
    n_pos_all = len(all_ids & conj)
    n_neg_all = len(all_ids - conj)
    n_pos_tar = len(targets & conj)
    n_neg_tar = len(targets - conj)

    fig, ax = plt.subplots(figsize=(4.0, 3.4))
    groups = ['Database', 'Targeted']
    x = np.arange(len(groups))
    w = 0.35
    pos_pct = [n_pos_all / len(all_ids) * 100,
               n_pos_tar / len(targets) * 100 if targets else 0]
    neg_pct = [n_neg_all / len(all_ids) * 100,
               n_neg_tar / len(targets) * 100 if targets else 0]
    ax.bar(x - w/2, pos_pct, width=w, color='#e0457b', edgecolor='white',
           linewidth=0.5, label='VirB4-T4CP+')
    ax.bar(x + w/2, neg_pct, width=w, color='#56b4e9', edgecolor='white',
           linewidth=0.5, label='VirB4-T4CP−')
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel('Percentage (%)')
    ax.set_ylim(0, max(pos_pct + neg_pct) * 1.15)
    ax.legend(frameon=False, fontsize=8, loc='upper right')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    _save(fig, "extE_virb4_database_vs_targeted")


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("Loading defence tables...")
    type_df, binary_type, sub_df, type_cols, sub_cols = load_defense_tables()
    print(f"  {len(type_df)} plasmids; "
          f"{int((type_df['n_instances'] > 0).sum())} with ≥ 1 defence")

    print("\nGenerating Fig 6 panels:")
    plot_total_defense_hist(type_df)
    plot_type_richness_hist(type_df)
    plot_per_phylum_box(type_df)
    plot_type_prevalence(type_df, type_cols)
    plot_cooccurrence_network(binary_type, type_cols)
    plot_defense_vs_size(type_df)
    plot_virb4_burden_and_richness()
    plot_crispr_target_fraction()
    plot_crispr_per_kb()

    print("\nGenerating Extended Data panels:")
    plot_subtype_panels(type_df)
    plot_types_by_virb4(type_df)
    plot_spacers_per_plasmid()
    plot_within_family_targeting()
    plot_virb4_database_vs_targeted()

    print(f"\nAll outputs in {FIG_DIR}")


if __name__ == "__main__":
    main()
