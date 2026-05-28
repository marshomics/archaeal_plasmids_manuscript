#!/usr/bin/env python3
"""Per-COG Fisher enrichment, cross-domain vs archaea-only — single-COG protein attribution.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "streamlined_cross_domain"))

import csv, re
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, shapiro
from statsmodels.stats.multitest import multipletests

from common import (CLUSTER_TSV, CLUSTER_SUMMARY, ARCHAEAL_EGGNOG,
                    COG_DESCRIPTIONS, header)

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)
METABOLIC_COGS = ['E', 'G', 'I']
N_BOOT = 500


def _normalize_id(pid):
    m = re.match(r'^(.+)_(\d+)$', str(pid))
    if m:
        return f'{m.group(1)}_{int(m.group(2)):05d}'
    return pid


def _per_protein_cog_weights(annot):
    """Return a DataFrame with one row per (protein, COG-letter) carrying
    a weight = 1/k, where k is the number of valid COG letters on the
    protein. Proteins with no valid letters get a single (protein, 'S')
    row with weight 1.0."""
    rows = []
    for prot_id, ctype, cog_str in annot[['pid', 'cluster_type',
                                          'COG_category']].itertuples(index=False):
        letters = [c for c in str(cog_str) if c in COG_DESCRIPTIONS]
        if not letters:
            letters = ['S']
        w = 1.0 / len(letters)
        for c in letters:
            rows.append((prot_id, ctype, c, w))
    return pd.DataFrame(rows, columns=['pid', 'cluster_type', 'COG', 'w'])


def main():
    header("CROSS-DOMAIN COG ENRICHMENT (updated: single-COG attribution)")

    cluster_type = dict(pd.read_csv(CLUSTER_SUMMARY,
        usecols=['cluster_rep', 'cluster_type']).itertuples(index=False, name=None))

    print("  Building member → cluster_rep map...")
    member_cluster = {}
    with open(CLUSTER_TSV) as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            member_cluster[row[1]] = row[0]
    print(f"  Member→cluster mappings: {len(member_cluster):,}")

    print("  Loading COG annotations from eggNOG file...")
    annot = pd.read_csv(ARCHAEAL_EGGNOG, sep='\t',
                        usecols=['proteins', 'COG_category'],
                        dtype=str, low_memory=False)
    annot['pid'] = annot['proteins'].apply(_normalize_id)
    annot['cluster_rep'] = annot['pid'].map(member_cluster)
    annot = annot.dropna(subset=['cluster_rep'])
    annot['cluster_type'] = annot['cluster_rep'].map(cluster_type)
    annot = annot[annot['cluster_type'].isin(['cross-domain', 'archaea-only'])]
    annot['COG_category'] = annot['COG_category'].fillna('S')
    annot.loc[annot['COG_category'] == '-', 'COG_category'] = 'S'
    annot.loc[annot['COG_category'].str.strip() == '', 'COG_category'] = 'S'
    print(f"  Mapped proteins: {len(annot):,}")

    weighted = _per_protein_cog_weights(annot)
    print(f"  Weighted (protein, COG) rows: {len(weighted):,}  "
          f"(total weight = {weighted['w'].sum():,.0f} ≈ n_proteins)")

    # Per-COG contingency: weighted protein equivalents.
    pivot = (weighted
             .groupby(['cluster_type', 'COG'])['w']
             .sum()
             .unstack(fill_value=0))
    pivot = pivot.reindex(columns=list(COG_DESCRIPTIONS.keys()), fill_value=0)
    total_cross = float(weighted.loc[weighted['cluster_type'] == 'cross-domain',
                                     'w'].sum())
    total_only  = float(weighted.loc[weighted['cluster_type'] == 'archaea-only',
                                     'w'].sum())

    rows = []
    for cog in COG_DESCRIPTIONS:
        a = float(pivot.at['cross-domain', cog]) if 'cross-domain' in pivot.index else 0
        c = float(pivot.at['archaea-only', cog]) if 'archaea-only' in pivot.index else 0
        if a + c < 10:
            continue
        # Round to integer protein-equivalents for the Fisher test.
        a_i, c_i = int(round(a)), int(round(c))
        b_i = int(round(total_cross - a))
        d_i = int(round(total_only - c))
        OR, p = fisher_exact([[a_i, b_i], [c_i, d_i]])
        rows.append({'COG': cog, 'description': COG_DESCRIPTIONS[cog],
                     'n_cross_weighted': round(a, 1),
                     'n_only_weighted':  round(c, 1),
                     'odds_ratio': OR, 'p_raw': p})
    out = pd.DataFrame(rows)
    out['p_adj'] = multipletests(out['p_raw'], method='fdr_bh')[1]
    out = out.sort_values('odds_ratio', ascending=False)
    out.to_csv(OUT_DIR / "cog_enrichment_crossdomain.csv", index=False)
    pd.set_option('display.float_format', '{:.3g}'.format)
    print(out[['COG', 'description', 'odds_ratio', 'p_adj']].to_string(index=False))

    # Bootstrap on proteins (not on (protein, letter) rows).
    print(f"\nMetabolic-OR bootstrap (COGs {METABOLIC_COGS}, "
          f"{N_BOOT} replicates, resampling proteins):")
    # Per-protein metabolic indicator: sum of weights on E/G/I letters.
    prot_summary = (weighted
        .groupby(['pid', 'cluster_type'])
        .apply(lambda g: pd.Series({
            'w_metab': float(g.loc[g['COG'].isin(METABOLIC_COGS), 'w'].sum()),
            'w_total': float(g['w'].sum()),
        }))
        .reset_index())
    rng = np.random.default_rng(0)
    arr = prot_summary[['cluster_type', 'w_metab', 'w_total']].to_numpy()
    n = len(arr)
    boot_ors = np.empty(N_BOOT)
    for i in range(N_BOOT):
        sample = arr[rng.integers(0, n, n)]
        is_cross = (sample[:, 0] == 'cross-domain')
        wm = sample[:, 1].astype(float)
        wt = sample[:, 2].astype(float)
        a_b = wm[is_cross].sum()
        b_b = wt[is_cross].sum() - a_b
        c_b = wm[~is_cross].sum()
        d_b = wt[~is_cross].sum() - c_b
        if a_b > 0 and b_b > 0 and c_b > 0 and d_b > 0:
            try:
                boot_ors[i], _ = fisher_exact(
                    [[int(round(a_b)), int(round(b_b))],
                     [int(round(c_b)), int(round(d_b))]])
            except Exception:
                boot_ors[i] = np.nan
        else:
            boot_ors[i] = np.nan
    boot_ors = boot_ors[~np.isnan(boot_ors)]
    ci_lo, ci_hi = np.percentile(boot_ors, [2.5, 97.5])
    print(f"  Bootstrap mean OR: {boot_ors.mean():.3f}")
    print(f"  Bootstrap 95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]")

    pd.DataFrame({'bootstrap_OR': boot_ors}).to_csv(
        OUT_DIR / 'metabolic_or_bootstrap.csv', index=False)


if __name__ == "__main__":
    main()
