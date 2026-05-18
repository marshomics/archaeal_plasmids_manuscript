#!/usr/bin/env python3
"""Generate every panel for the sampling-bias figure set.

Outputs go to ``outputs/figures/`` next to this script. Every quantity is
re-derived from the loaded input data and the other scripts in this
pipeline; no numbers are hard-coded. Each panel is written as both PNG
(raster) and SVG (with ``svg.fonttype = 'none'`` so labels stay editable in
vector software).

Run as ``python 99_plots.py``.
"""
from pathlib import Path
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_data, DEPTH_BINS  # noqa


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
matplotlib.rcParams['font.family']  = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

# A pared-back colour scheme that survives editing in Illustrator.
HALO_COLOR   = '#1f77b4'
OTHER_COLOR  = '#d62728'
CARRIER_BLUE = '#2c7a3e'
NONCARRIER_RED = '#d04141'
PHY_COLOURS = {
    'p__Halobacteriota':      '#2ca02c',
    'p__Methanobacteriota':   '#ff9f1c',
    'p__Methanobacteriota_B': '#56B4E9',
    'p__Thermoproteota':      '#cc79a7',
    'p__Thermoplasmatota':    '#1abc9c',
}
CARRIER_PHYLA = list(PHY_COLOURS.keys())


def _short(name):
    return name.split('__', 1)[-1] if isinstance(name, str) and '__' in name else name


def _depth_bin(n):
    for lo, hi, lab in DEPTH_BINS:
        if lo <= n <= hi:
            return lab
    return None


def _save(fig, stem):
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=200, bbox_inches='tight')
    fig.savefig(OUT_DIR / f"{stem}.svg", bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {stem}.png + .svg")


# ===========================================================================
# Fig 2A — carrier rate by NCBI assembly level
# ===========================================================================
def plot_assembly_level_carrier_rate(reps_meta):
    df = (reps_meta.groupby('ncbi_assembly_level')
                   .agg(n=('accession', 'count'),
                        n_carriers=('is_carrier', 'sum'))
                   .reset_index())
    df['rate'] = 100 * df['n_carriers'] / df['n']
    # Display order: Complete Genome (+ Chromosome merged conceptually), Scaffold, Contig
    order = ['Complete Genome', 'Scaffold', 'Contig']
    df = df[df['ncbi_assembly_level'].isin(order)].copy()
    df['ncbi_assembly_level'] = pd.Categorical(df['ncbi_assembly_level'],
                                               categories=order, ordered=True)
    df = df.sort_values('ncbi_assembly_level')
    labels = {'Complete Genome': 'Complete\ngenome',
              'Scaffold': 'Scaffold',
              'Contig': 'Contig'}

    fig, ax = plt.subplots(figsize=(3.4, 3.6))
    x = np.arange(len(df))
    ax.bar(x, df['rate'], color=HALO_COLOR, edgecolor='white', linewidth=0.5,
           width=0.7)
    for xi, r, c, n in zip(x, df['rate'], df['n_carriers'], df['n']):
        ax.text(xi, r + 1, f"{int(c)}/{int(n)}", ha='center', va='bottom',
                fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[l] for l in df['ncbi_assembly_level']])
    ax.set_ylabel("Carrier rate (%)")
    ax.set_ylim(0, df['rate'].max() * 1.18)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    _save(fig, "fig2A_carrier_rate_by_assembly_level")


# ===========================================================================
# Fig 2B — carrier rate by per-species genome count (depth bins)
# ===========================================================================
def plot_carrier_rate_by_depth(reps):
    df = reps.copy()
    df['depth_bin'] = df['n_genomes'].apply(_depth_bin)
    bin_order = [b[2] for b in DEPTH_BINS]
    g = (df.groupby('depth_bin')
           .agg(n=('accession', 'count'),
                n_carriers=('is_carrier', 'sum'))
           .reindex(bin_order)
           .reset_index())
    g['rate'] = 100 * g['n_carriers'] / g['n']

    fig, ax = plt.subplots(figsize=(3.8, 3.6))
    x = np.arange(len(bin_order))
    ax.bar(x, g['rate'], color=HALO_COLOR, edgecolor='white', linewidth=0.5,
           width=0.7)
    for xi, r, c, n in zip(x, g['rate'], g['n_carriers'], g['n']):
        ax.text(xi, r + 0.3, f"{int(c)}/{int(n)}\n({r:.1f}%)",
                ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_order)
    ax.set_xlabel('Genomes per species')
    ax.set_ylabel('Species with plasmid (%)')
    ax.set_ylim(0, g['rate'].max() * 1.30)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    _save(fig, "fig2B_carrier_rate_by_depth")


# ===========================================================================
# Fig 2C — per-phylum carrier rate by depth, with gap ratios
# ===========================================================================
def plot_per_phylum_depth(reps):
    df = reps.copy()
    df['depth_bin'] = df['n_genomes'].apply(_depth_bin)
    df['is_halo']   = (df['gtdb_phylum'] == 'p__Halobacteriota').astype(int)
    bin_order = [b[2] for b in DEPTH_BINS]

    rows = []
    for lab in bin_order:
        for grp_name, mask in [('Halobacteriota', df['is_halo'] == 1),
                               ('Other phyla',    df['is_halo'] == 0)]:
            sub = df[mask & (df['depth_bin'] == lab)]
            n = len(sub); k = int(sub['is_carrier'].sum())
            rows.append({'group': grp_name, 'depth_bin': lab,
                         'carriers': k, 'n': n,
                         'rate': 100.0 * k / n if n else 0.0})
    table = pd.DataFrame(rows)
    pivot = table.pivot(index='depth_bin', columns='group',
                        values='rate').reindex(bin_order)
    gaps = (pivot['Halobacteriota'] / pivot['Other phyla'].replace(0, np.nan))

    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    xpos = np.arange(len(bin_order))
    for grp, col in [('Halobacteriota', HALO_COLOR),
                     ('Other phyla',    OTHER_COLOR)]:
        sub = table[table['group'] == grp].set_index('depth_bin').reindex(bin_order)
        ax.plot(xpos, sub['rate'].values, marker='o', color=col,
                label=grp, linewidth=1.6, markersize=5)
        for x, r, c, n in zip(xpos, sub['rate'], sub['carriers'], sub['n']):
            ax.annotate(f"{int(c)}/{int(n)}", (x, r),
                        textcoords='offset points', xytext=(5, 5),
                        fontsize=7, color=col)

    halo_y  = pivot['Halobacteriota'].values
    other_y = pivot['Other phyla'].values
    for x, hy, oy, g in zip(xpos, halo_y, other_y, gaps.values):
        if np.isfinite(g):
            mid = (hy + oy) / 2
            ax.text(x + 0.06, mid, f"{g:.0f}×", ha='left', va='center',
                    fontsize=8, color='dimgray',
                    bbox=dict(facecolor='white', edgecolor='none',
                              pad=1.0, alpha=0.85))

    ax.set_xticks(xpos)
    ax.set_xticklabels(bin_order)
    ax.set_xlabel('Genomes per species')
    ax.set_ylabel('Species with plasmid (%)')
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylim(0, max(table['rate']) * 1.18)
    plt.tight_layout()
    _save(fig, "fig2C_per_phylum_carrier_rate")


# ===========================================================================
# Fig 2D — logistic regression: complete-genomes subset, predicted curves
# ===========================================================================
def plot_logistic_curves(reps, meta, reps_meta):
    cg = reps_meta[reps_meta['ncbi_assembly_level'] == 'Complete Genome'].copy()
    cg['is_halo'] = (cg['gtdb_phylum'] == 'p__Halobacteriota').astype(int)
    cg['log_depth'] = np.log(cg['n_genomes'].astype(float))
    cov_cols = ['gc_percentage', 'checkm_completeness', 'checkm_contamination',
                'coding_density', 'genome_size', 'contig_count']
    cg = cg.merge(meta[['accession'] + cov_cols], on='accession', how='left')
    cg['log_genome_size'] = np.log(cg['genome_size'].astype(float))
    cg['log_contigs']     = np.log(cg['contig_count'].astype(float).replace(0, 1))
    X_cols = ['is_halo', 'log_depth', 'log_genome_size', 'gc_percentage',
              'checkm_completeness', 'checkm_contamination', 'coding_density',
              'log_contigs']
    cg = cg.dropna(subset=X_cols)
    y = cg['is_carrier'].astype(int).values
    X = sm.add_constant(cg[X_cols].values)
    logit = sm.Logit(y, X).fit(disp=False, maxiter=500)

    X_red = sm.add_constant(cg[[c for c in X_cols if c != 'is_halo']].values)
    logit_red = sm.Logit(y, X_red).fit(disp=False, maxiter=500)
    lrt = 2 * (logit.llf - logit_red.llf)
    p_lrt = 1 - stats.chi2.cdf(lrt, df=1)
    or_halo = np.exp(logit.params[X_cols.index('is_halo') + 1])
    p_wald_halo = logit.pvalues[X_cols.index('is_halo') + 1]
    auc = roc_auc_score(y, logit.predict(X))

    medians = cg[X_cols].median()
    depth_grid = np.linspace(1, cg['n_genomes'].max(), 200)
    log_grid = np.log(depth_grid)

    def _predict(is_halo_val):
        Xp = np.tile(medians.values, (len(depth_grid), 1))
        Xp[:, X_cols.index('is_halo')]   = is_halo_val
        Xp[:, X_cols.index('log_depth')] = log_grid
        Xp_const = np.column_stack([np.ones(len(depth_grid)), Xp])
        eta = Xp_const @ logit.params
        cov = logit.cov_params()
        var_eta = np.einsum('ij,jk,ik->i', Xp_const, cov, Xp_const)
        se_eta = np.sqrt(var_eta)
        return (1/(1+np.exp(-eta)),
                1/(1+np.exp(-(eta - 1.96*se_eta))),
                1/(1+np.exp(-(eta + 1.96*se_eta))))

    halo_curve, halo_lo, halo_hi = _predict(1)
    other_curve, other_lo, other_hi = _predict(0)

    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.fill_between(depth_grid, halo_lo*100, halo_hi*100, color=HALO_COLOR,
                    alpha=0.18, linewidth=0)
    ax.plot(depth_grid, halo_curve*100, color=HALO_COLOR,
            label='Halobacteriota', linewidth=1.8)
    ax.fill_between(depth_grid, other_lo*100, other_hi*100, color=OTHER_COLOR,
                    alpha=0.18, linewidth=0)
    ax.plot(depth_grid, other_curve*100, color=OTHER_COLOR,
            label='Other phyla', linewidth=1.8)

    # Overlay empirical points (bins with ≥3 species)
    for hval, col in [(1, HALO_COLOR), (0, OTHER_COLOR)]:
        sub = cg[cg['is_halo'] == hval]
        for n_val in sorted(sub['n_genomes'].unique()):
            grp = sub[sub['n_genomes'] == n_val]
            if len(grp) >= 3:
                ax.scatter(n_val, grp['is_carrier'].mean()*100,
                           s=np.sqrt(len(grp))*8, color=col, edgecolor='white',
                           linewidth=0.4, alpha=0.65, zorder=3)

    txt = (f"n = {len(y)} complete genomes\n"
           f"Phylum OR = {or_halo:.2f}, p = {p_wald_halo:.3f}\n"
           f"LRT p (phylum | covars) = {p_lrt:.3f}\n"
           f"McFadden R² = {logit.prsquared:.2f},  AUC = {auc:.2f}")
    ax.text(0.97, 0.03, txt, transform=ax.transAxes, ha='right', va='bottom',
            fontsize=7,
            bbox=dict(facecolor='white', edgecolor='lightgray',
                      boxstyle='round,pad=0.4'))
    ax.set_xlabel('Genomes per species')
    ax.set_ylabel('Predicted plasmid detection (%)')
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlim(0, cg['n_genomes'].max() + 1)
    plt.tight_layout()
    _save(fig, "fig2D_logistic_regression")


# ===========================================================================
# Fig 2E — observed vs depth-expected plasmid-bearing species per phylum
# (depth-stratified label permutation; two-sided p with BH-FDR)
# ===========================================================================
def plot_observed_vs_expected(reps, n_perm=10_000):
    df = reps.copy()
    df['depth_bin'] = df['n_genomes'].apply(_depth_bin)
    observed = df.groupby('gtdb_phylum')['is_carrier'].sum()

    bin_rates = df.groupby('depth_bin')['is_carrier'].mean().to_dict()
    expected = {}
    for phy in df['gtdb_phylum'].dropna().unique():
        sub = df[df['gtdb_phylum'] == phy]
        expected[phy] = sum(int((sub['depth_bin'] == db).sum()) * rate
                            for db, rate in bin_rates.items())

    rng = np.random.default_rng(42)
    perm_counts = {phy: np.zeros(n_perm) for phy in expected}
    for i in range(n_perm):
        shuffled = df['is_carrier'].to_numpy().copy()
        for db in df['depth_bin'].unique():
            mask = (df['depth_bin'] == db).to_numpy()
            shuffled[mask] = rng.permutation(shuffled[mask])
        tmp = df.assign(perm=shuffled).groupby('gtdb_phylum')['perm'].sum()
        for phy in perm_counts:
            perm_counts[phy][i] = tmp.get(phy, 0)

    rows = []
    for phy in expected:
        obs = float(observed.get(phy, 0))
        exp = expected[phy]
        oe = obs / exp if exp > 0 else float('nan')
        dev = abs(obs - exp)
        perm_dev = np.abs(perm_counts[phy] - exp)
        p_two = ((perm_dev >= dev).sum() + 1) / (n_perm + 1)
        rows.append({'phylum': phy, 'n_species': int((df['gtdb_phylum'] == phy).sum()),
                     'observed': obs, 'expected': exp, 'OE': oe,
                     'p_two': p_two})
    res = pd.DataFrame(rows)
    res = res[res['n_species'] >= 20].copy()
    carriers = res['observed'] > 0
    if carriers.any():
        res.loc[carriers, 'p_BH'] = multipletests(
            res.loc[carriers, 'p_two'], method='fdr_bh')[1]
    res.loc[~carriers, 'p_BH'] = np.nan
    # Order: carrier-bearing first (by O/E descending), then zero-carrier (by n descending)
    res_carrier = res[carriers].sort_values('OE', ascending=False)
    res_zero    = res[~carriers].sort_values('n_species', ascending=False)
    full = pd.concat([res_carrier, res_zero], ignore_index=True)

    # Figure: left panel = observed (filled) vs expected (open) lollipop;
    # right panel = O/E ratio bar on log scale.
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(8.5, max(4.0, 0.32 * len(full))),
        gridspec_kw={'width_ratios': [1.4, 1.0], 'wspace': 0.05},
        sharey=True,
    )
    y = np.arange(len(full))[::-1]

    for yi, (_, r) in zip(y, full.iterrows()):
        col = PHY_COLOURS.get(r['phylum'], '#999999')
        ax_left = axL
        ax_left.plot([min(r['observed'], r['expected']),
                      max(r['observed'], r['expected'])],
                     [yi, yi], color=col, alpha=0.6, linewidth=2)
        ax_left.scatter(r['observed'], yi, color=col, s=40, zorder=4,
                        edgecolor='black', linewidth=0.4)
        ax_left.scatter(r['expected'], yi, facecolor='white', s=40, zorder=3,
                        edgecolor=col, linewidth=1.4)

    axL.set_yticks(y)
    axL.set_yticklabels([f"{_short(r['phylum'])} (n = {int(r['n_species']):,})"
                         for _, r in full.iterrows()], fontsize=8)
    for tick, phy in zip(axL.get_yticklabels(), full['phylum']):
        if phy in CARRIER_PHYLA:
            tick.set_color(PHY_COLOURS[phy]); tick.set_fontweight('bold')
        else:
            tick.set_color('#777')
    axL.set_xlabel('Number of plasmid-\nbearing species')
    axL.set_xlim(0, max(full[['observed', 'expected']].values.max(), 1) * 1.10)
    axL.spines[['top', 'right']].set_visible(False)

    from matplotlib.lines import Line2D
    axL.legend(handles=[
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markeredgecolor='black', markersize=7, label='Observed'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
               markeredgecolor='gray', markersize=7,
               label='Expected (depth-only null)'),
    ], frameon=False, fontsize=7, loc='lower right')

    # Right panel: O/E on log scale
    for yi, (_, r) in zip(y, full.iterrows()):
        col = PHY_COLOURS.get(r['phylum'], '#bbb')
        oe = r['OE']
        if not np.isfinite(oe) or oe == 0:
            axR.text(1.05, yi, '0', va='center', ha='center', fontsize=7,
                     color='#888')
            continue
        # Use horizontal log-scale bar from 1 to OE
        axR.barh(yi, oe, color=col, alpha=0.85, edgecolor='none')
        # Label
        q = r['p_BH']
        sig = "*" if (np.isfinite(q) and q < 0.05) else ""
        lab = f"{oe:.1g}×{sig}"
        axR.text(oe * 1.1, yi, lab, va='center', fontsize=8, color='#222')
    axR.set_xscale('log')
    axR.set_xlim(1e-2, 1e1)
    axR.axvline(1.0, color='gray', linestyle='--', linewidth=0.7)
    axR.set_xlabel('Observed/\nExpected ratio')
    axR.spines[['top', 'right', 'left']].set_visible(False)
    axR.tick_params(axis='y', which='both', left=False)

    plt.tight_layout()
    _save(fig, "fig2E_observed_vs_expected")


# ===========================================================================
# Fig 2F — assembly composition + complete genome availability (combined,
# shared y-axis ordering)
# ===========================================================================
def plot_assembly_and_completeness(reps, meta):
    m = meta[['gtdb_phylum', 'ncbi_assembly_level']].dropna(
        subset=['gtdb_phylum']).copy()
    def _level3(x):
        return 'Complete' if x in ('Complete Genome', 'Chromosome') else x
    m['level3'] = m['ncbi_assembly_level'].apply(_level3)
    m = m[m['level3'].isin(['Complete', 'Scaffold', 'Contig'])]
    comp = m.groupby(['gtdb_phylum', 'level3']).size().unstack(fill_value=0)
    for lv in ['Complete', 'Scaffold', 'Contig']:
        if lv not in comp.columns:
            comp[lv] = 0
    comp = comp[['Complete', 'Scaffold', 'Contig']]
    comp['total'] = comp.sum(axis=1)
    comp = comp.sort_values('total', ascending=False)
    pct = comp[['Complete', 'Scaffold', 'Contig']].div(comp['total'], axis=0) * 100
    phy_order = list(comp.index)

    cg = meta[meta['ncbi_assembly_level'] == 'Complete Genome']
    species_with_cg = cg.groupby('gtdb_phylum')['gtdb_species'].unique().to_dict()
    reps_carrier_species = set(
        reps.loc[reps['is_carrier'] == 1, 'gtdb_species'].dropna().unique()
    )
    right_rows = []
    for phy in phy_order:
        sp_set = set(species_with_cg.get(phy, []))
        n_total = len(sp_set)
        n_carr = len(sp_set & reps_carrier_species)
        right_rows.append({'phylum': phy, 'n_total': n_total, 'n_carr': n_carr,
                           'pct': round(100 * n_carr / n_total, 0) if n_total else 0})
    right = pd.DataFrame(right_rows).set_index('phylum').loc[phy_order]

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(9.0, max(3.0, 0.30 * len(phy_order))),
        gridspec_kw={'width_ratios': [1.4, 1.0], 'wspace': 0.05},
        sharey=True,
    )
    y = np.arange(len(phy_order))[::-1]
    colours = {'Complete': '#2c7a3e', 'Scaffold': '#f4d35e', 'Contig': '#e8807a'}
    left = np.zeros(len(phy_order))
    for lv in ['Complete', 'Scaffold', 'Contig']:
        vals = np.array([pct.loc[p, lv] for p in phy_order])
        axL.barh(y, vals, left=left, color=colours[lv], edgecolor='white',
                 linewidth=0.5, label=lv)
        left += vals
    labels = [f"{_short(p)} ({comp.loc[p, 'total']:,})" for p in phy_order]
    axL.set_yticks(y); axL.set_yticklabels(labels, fontsize=8)
    for tick, p in zip(axL.get_yticklabels(), phy_order):
        if p in CARRIER_PHYLA:
            tick.set_color(PHY_COLOURS[p]); tick.set_fontweight('bold')
    axL.set_xlabel('Genome composition (%)'); axL.set_xlim(0, 100)
    axL.set_title('Assembly level by phylum', fontsize=10)
    axL.legend(loc='lower right', frameon=False, fontsize=8,
               bbox_to_anchor=(1.0, 1.02), ncol=3)
    axL.spines[['top', 'right']].set_visible(False)

    right_color = '#7fbfd1'
    n_max = max(right['n_total'].max(), 1)
    for yi, p in zip(y, phy_order):
        n_total = int(right.loc[p, 'n_total'])
        n_carr  = int(right.loc[p, 'n_carr'])
        pct_c   = int(right.loc[p, 'pct'])
        if n_total > 0:
            axR.barh(yi, n_total, color=right_color, edgecolor='white',
                     linewidth=0.4)
            axR.text(n_total + n_max * 0.015, yi,
                     f"{n_carr}/{n_total} carriers ({pct_c}%)",
                     va='center', fontsize=7, color='#333')
        else:
            axR.text(n_max * 0.015, yi, '0', va='center', fontsize=7,
                     color='#aaa')
    axR.set_xlabel('Species with complete genomes')
    axR.set_title('Complete genome availability', fontsize=10)
    axR.set_xlim(0, n_max * 1.55)
    axR.spines[['top', 'right', 'left']].set_visible(False)
    axR.tick_params(axis='y', which='both', left=False)

    plt.tight_layout()
    _save(fig, "fig2F_assembly_and_completeness")


# ===========================================================================
# Extended Data A — proportion of group by assembly level (carriers vs non)
# ===========================================================================
def plot_assembly_level_by_carrier(reps_meta):
    ct = pd.crosstab(reps_meta['ncbi_assembly_level'], reps_meta['is_carrier'])
    # Restrict to the three categories shown in the panel; merge Chromosome
    # into Complete to match the typical figure aesthetic if present.
    if 'Chromosome' in ct.index and 'Complete Genome' in ct.index:
        ct.loc['Complete Genome'] = ct.loc['Complete Genome'] + ct.loc['Chromosome']
        ct = ct.drop('Chromosome')
    ct = ct.reindex(['Complete Genome', 'Scaffold', 'Contig']).fillna(0).astype(int)
    pct = ct.div(ct.sum(axis=0), axis=1) * 100

    n_carrier = int(ct.sum(axis=0)[1])
    n_non     = int(ct.sum(axis=0)[0])

    fig, ax = plt.subplots(figsize=(4.0, 3.4))
    x = np.arange(len(ct))
    w = 0.38
    ax.bar(x - w/2, pct[1].values, width=w, color=HALO_COLOR,
           edgecolor='white', linewidth=0.5, label=f"Carriers (n = {n_carrier:,})")
    ax.bar(x + w/2, pct[0].values, width=w, color=OTHER_COLOR,
           edgecolor='white', linewidth=0.5,
           label=f"Non-carriers (n = {n_non:,})")
    ax.set_xticks(x)
    ax.set_xticklabels(['Complete', 'Scaffold', 'Contig'])
    ax.set_ylabel('Proportion of group (%)')
    ax.set_ylim(0, max(pct.values.max() * 1.10, 5))
    ax.legend(frameon=False, fontsize=8, loc='upper right')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    _save(fig, "ext_dataA_assembly_level_by_carrier")


# ===========================================================================
# Extended Data B — proportion of group by NCBI genome category
# ===========================================================================
def plot_genome_category_by_carrier(reps_meta):
    cat_map = {
        'none':                              'Isolate',
        'derived from metagenome':           'MAG',
        'derived from environmental sample': 'Env.\nsample',
        'derived from single cell':          'Single cell',
    }
    df = reps_meta.copy()
    df['cat_short'] = df['ncbi_genome_category'].fillna('Unknown').map(
        lambda x: cat_map.get(x, x))
    order = ['Isolate', 'MAG', 'Env.\nsample', 'Single cell']
    ct = pd.crosstab(df['cat_short'], df['is_carrier']).reindex(order).fillna(0).astype(int)
    pct = ct.div(ct.sum(axis=0), axis=1) * 100

    fig, ax = plt.subplots(figsize=(4.5, 3.4))
    x = np.arange(len(ct))
    w = 0.38
    ax.bar(x - w/2, pct[1].values, width=w, color=HALO_COLOR,
           edgecolor='white', linewidth=0.5, label='Carriers')
    ax.bar(x + w/2, pct[0].values, width=w, color=OTHER_COLOR,
           edgecolor='white', linewidth=0.5, label='Non-carriers')
    ax.set_xticks(x); ax.set_xticklabels(order, fontsize=8)
    ax.set_ylabel('Proportion of group (%)')
    ax.set_ylim(0, max(pct.values.max() * 1.10, 5))
    ax.legend(frameon=False, fontsize=8, loc='upper right')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    _save(fig, "ext_dataB_genome_category_by_carrier")


# ===========================================================================
# Extended Data C — per-phylum carrier rate, three subsets (all / iso / cg)
# ===========================================================================
def _subset_rates(reps_meta, reps):
    subsets = {
        'All species': reps_meta,
        'Isolates only': reps_meta[reps_meta['ncbi_genome_category'] == 'none'],
        'Complete genomes': reps_meta[reps_meta['ncbi_assembly_level'] == 'Complete Genome'],
    }
    carrier_phyla = sorted(
        reps.loc[reps['is_carrier'] == 1, 'gtdb_phylum'].dropna().unique())
    rows = []
    for name, df in subsets.items():
        for phy in carrier_phyla:
            sub = df[df['gtdb_phylum'] == phy]
            n = len(sub); k = int(sub['is_carrier'].sum())
            rows.append({'subset': name, 'phylum': phy, 'n': n, 'carriers': k,
                         'rate': 100*k/n if n else 0.0})
    return pd.DataFrame(rows), carrier_phyla, subsets


def plot_per_phylum_three_subsets(reps_meta, reps):
    rates, phylum_order, subsets = _subset_rates(reps_meta, reps)
    # Order phyla by descending carrier rate in the "All species" subset
    order = (rates[rates['subset'] == 'All species']
             .sort_values('rate', ascending=False)['phylum'].tolist())

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    subset_names = ['All species', 'Isolates only', 'Complete genomes']
    # Use lighter → darker shade of the same colour to encode subset
    shade_alpha = {'All species': 0.35, 'Isolates only': 0.65,
                   'Complete genomes': 1.0}
    n_phy = len(order)
    w = 0.25
    x = np.arange(n_phy)
    for i, sub_name in enumerate(subset_names):
        sub = rates[rates['subset'] == sub_name].set_index('phylum').reindex(order)
        cols = [PHY_COLOURS.get(p, '#666') for p in order]
        ax.bar(x + (i - 1) * w, sub['rate'].values, width=w,
               color=cols, edgecolor='white', linewidth=0.4,
               alpha=shade_alpha[sub_name], label=sub_name)
    ax.set_xticks(x)
    ax.set_xticklabels([_short(p) for p in order], rotation=20, ha='right',
                       fontsize=8)
    ax.set_ylabel('Carrier rate (%)')
    ax.set_ylim(0, rates['rate'].max() * 1.10)
    ax.legend(frameon=False, fontsize=8, loc='upper right')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    _save(fig, "ext_dataC_per_phylum_three_subsets")


# ===========================================================================
# Extended Data D, E — Halo-vs-target Fisher tests within each subset,
# with BH-FDR across all twelve (subset × target) comparisons.
# ===========================================================================
def _fisher_table_with_bh(reps_meta, reps):
    """Reproduce the joint BH-FDR table from 06_quality_subset_robustness.py."""
    subsets = {
        'All species':       reps_meta,
        'Isolates only':     reps_meta[reps_meta['ncbi_genome_category'] == 'none'],
        'Complete genomes':  reps_meta[reps_meta['ncbi_assembly_level'] == 'Complete Genome'],
    }
    carrier_phyla = sorted(
        reps.loc[reps['is_carrier'] == 1, 'gtdb_phylum'].dropna().unique())
    halo = 'p__Halobacteriota'
    rows = []
    for sub_name, df in subsets.items():
        for tgt in (p for p in carrier_phyla if p != halo):
            h_sub = df[df['gtdb_phylum'] == halo]
            t_sub = df[df['gtdb_phylum'] == tgt]
            h_c, h_n = int(h_sub['is_carrier'].sum()), len(h_sub)
            t_c, t_n = int(t_sub['is_carrier'].sum()), len(t_sub)
            if h_n == 0 or t_n == 0:
                continue
            OR, p = stats.fisher_exact([[h_c, h_n - h_c], [t_c, t_n - t_c]])
            rows.append({'subset': sub_name, 'target': tgt,
                         'h_c': h_c, 'h_n': h_n, 't_c': t_c, 't_n': t_n,
                         'OR': OR, 'p_raw': p})
    table = pd.DataFrame(rows)
    table['p_BH'] = multipletests(table['p_raw'], method='fdr_bh')[1]
    return table


def _plot_one_subset(table, subset_name, save_stem):
    sub = table[table['subset'] == subset_name].copy().set_index('target')
    halo_label = 'p__Halobacteriota'
    halo_row = sub.iloc[0]  # any row carries the Halo (h_c, h_n)
    # Fixed phylum order across both panels — matches the other Fig 2 panels.
    fixed_order = ['p__Halobacteriota', 'p__Methanobacteriota',
                   'p__Methanobacteriota_B', 'p__Thermoproteota',
                   'p__Thermoplasmatota']
    bars = [(halo_label, int(halo_row['h_c']), int(halo_row['h_n']))]
    for phy in fixed_order:
        if phy == halo_label or phy not in sub.index:
            continue
        r = sub.loc[phy]
        bars.append((phy, int(r['t_c']), int(r['t_n'])))

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    x = np.arange(len(bars))
    rates = [100 * c / n for _, c, n in bars]
    colours = [PHY_COLOURS.get(p, '#666') for p, _, _ in bars]
    ax.bar(x, rates, color=colours, edgecolor='white', linewidth=0.5,
           width=0.7)
    for xi, (p, c, n) in zip(x, bars):
        ax.text(xi, 100 * c / n + 0.5, f"{c}/{n}", ha='center', va='bottom',
                fontsize=8)
    # significance brackets above bars: Halo vs each target, in the order
    # the bars appear.
    y_top = max(rates) * 1.08
    step = max(rates) * 0.10
    for j, (phy, _, _) in enumerate(bars[1:], start=1):
        q = float(sub.loc[phy, 'p_BH'])
        col = '#a30' if q < 0.05 else '#777'
        x_left, x_right = 0, j
        y = y_top + step * (j - 1)
        ax.plot([x_left, x_left, x_right, x_right],
                [y - step * 0.2, y, y, y - step * 0.2],
                color='lightgray', lw=0.7)
        # Choose format based on size of q (compact for tiny values)
        q_str = f"{q:.1e}" if q < 1e-3 else f"{q:.3f}"
        ax.text((x_left + x_right) / 2, y + 0.1, f"q = {q_str}",
                ha='center', va='bottom', fontsize=8, color=col)
    ax.set_xticks(x)
    ax.set_xticklabels([_short(p) for p, _, _ in bars], rotation=25,
                       ha='right', fontsize=8)
    ax.set_ylabel('Carrier rate (%)')
    ax.set_ylim(0, y_top + step * (len(bars) - 1) + 3)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    _save(fig, save_stem)


def plot_quality_subset_panels(reps_meta, reps):
    table = _fisher_table_with_bh(reps_meta, reps)
    _plot_one_subset(table, 'Isolates only',     'ext_dataD_isolates_only_fisher')
    _plot_one_subset(table, 'Complete genomes',  'ext_dataE_complete_genomes_fisher')


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("Loading data...")
    reps, meta, reps_meta = load_data()
    print(f"  reps:      {len(reps):,} representative species "
          f"({int(reps['is_carrier'].sum())} carriers)")
    print(f"  reps_meta: {len(reps_meta):,} species with NCBI metadata "
          f"({int(reps_meta['is_carrier'].sum())} carriers)")

    print("\nGenerating Fig 2 panels:")
    plot_assembly_level_carrier_rate(reps_meta)
    plot_carrier_rate_by_depth(reps)
    plot_per_phylum_depth(reps)
    plot_logistic_curves(reps, meta, reps_meta)
    plot_observed_vs_expected(reps)
    plot_assembly_and_completeness(reps, meta)

    print("\nGenerating Extended Data panels:")
    plot_assembly_level_by_carrier(reps_meta)
    plot_genome_category_by_carrier(reps_meta)
    plot_per_phylum_three_subsets(reps_meta, reps)
    plot_quality_subset_panels(reps_meta, reps)

    print(f"\nAll outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
