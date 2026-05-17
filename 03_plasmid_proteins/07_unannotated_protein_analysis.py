#!/usr/bin/env python3
"""Unannotated-protein burden: overall, per-plasmid, and per-phylum with bootstrap CIs.

Pairwise phylum comparisons use a species-weighted permutation test on the
difference of inverse-species-frequency-weighted means (matches the weighting
used in the descriptive estimates), with BH-FDR across pairs.
"""
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from common import ARCHAEAL_EGGNOG, OUT_DIR, COG_DESCRIPTIONS, header

N_BOOT = 1000
N_PERM = 10_000


def _is_unknown(cog):
    """True if no informative COG letter (only S or empty/-)."""
    if pd.isna(cog) or cog == '-' or str(cog).strip() == '':
        return True
    letters = [c for c in str(cog) if c in COG_DESCRIPTIONS]
    return (not letters) or all(c == 'S' for c in letters)


def _weighted_perm_p(vals1, w1, vals2, w2, n_perm=N_PERM, seed=42):
    """Two-sided permutation p on the difference of weighted means."""
    obs = np.average(vals1, weights=w1) - np.average(vals2, weights=w2)
    pooled_v = np.concatenate([vals1, vals2])
    pooled_w = np.concatenate([w1, w2])
    n1 = len(vals1)
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(n_perm):
        idx = rng.permutation(len(pooled_v))
        p_v = pooled_v[idx]
        p_w = pooled_w[idx]
        diff = np.average(p_v[:n1], weights=p_w[:n1]) - \
               np.average(p_v[n1:], weights=p_w[n1:])
        if abs(diff) >= abs(obs):
            extreme += 1
    return obs, (extreme + 1) / (n_perm + 1)


def main():
    header("UNANNOTATED-PROTEIN BURDEN (species-weighted)")
    df = pd.read_csv(ARCHAEAL_EGGNOG, sep='\t', low_memory=False)
    df['is_unknown'] = df['COG_category'].apply(_is_unknown)

    overall = df['is_unknown'].mean() * 100
    print(f"Overall unannotated proteins: {overall:.1f}%")

    plas = df.groupby('replicon').agg(
        n_proteins=('is_unknown', 'size'),
        n_unknown=('is_unknown', 'sum'),
        phylum=('gtdb_phylum', 'first'),
        species=('gtdb_species', 'first'),
    ).reset_index()
    plas['pct_unknown']      = 100.0 * plas['n_unknown'] / plas['n_proteins']
    plas['majority_unknown'] = plas['pct_unknown'] > 50
    plas['phylum_short']     = plas['phylum'].str.replace('p__', '', regex=False)

    n_maj = int(plas['majority_unknown'].sum())
    print(f"Plasmids with majority unannotated: "
          f"{n_maj}/{len(plas)} ({n_maj/len(plas)*100:.1f}%)")

    # inverse-species-frequency weighting; bootstrap the weighted mean
    sp_counts = plas.groupby('species')['replicon'].nunique()
    plas['isf_w'] = plas['species'].map(lambda s: 1.0 / sp_counts.get(s, 1))

    phyla = sorted(plas['phylum_short'].dropna().unique())

    rng = np.random.default_rng(42)
    rows = []
    for phy in phyla:
        grp = plas[plas['phylum_short'] == phy]
        if len(grp) == 0:
            continue
        vals = grp['pct_unknown'].to_numpy()
        w = grp['isf_w'].to_numpy()
        wm = np.average(vals, weights=w)
        boots = [np.average(vals[idx], weights=w[idx])
                 for idx in (rng.integers(0, len(vals), len(vals)) for _ in range(N_BOOT))]
        rows.append({'phylum': phy, 'n_plasmids': len(grp),
                     'weighted_mean_pct_unknown': round(wm, 1),
                     'CI_lower': round(np.percentile(boots, 2.5), 1),
                     'CI_upper': round(np.percentile(boots, 97.5), 1)})
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "unknown_proportion_summary.csv", index=False)
    print("\nPer-phylum weighted unannotated rate:")
    print(summary.to_string(index=False))

    # pairwise weighted permutation test with BH-FDR
    pairs = []
    for i in range(len(phyla)):
        for j in range(i + 1, len(phyla)):
            grp1 = plas[plas['phylum_short'] == phyla[i]]
            grp2 = plas[plas['phylum_short'] == phyla[j]]
            if len(grp1) < 2 or len(grp2) < 2:
                continue
            v1 = grp1['pct_unknown'].to_numpy()
            w1 = grp1['isf_w'].to_numpy()
            v2 = grp2['pct_unknown'].to_numpy()
            w2 = grp2['isf_w'].to_numpy()
            diff, p_raw = _weighted_perm_p(v1, w1, v2, w2,
                                           seed=42 + i * 100 + j)
            pairs.append({'group1': phyla[i], 'group2': phyla[j],
                          'n1': len(grp1), 'n2': len(grp2),
                          'weighted_mean_diff': round(diff, 2),
                          'p_raw': p_raw})
    pw = pd.DataFrame(pairs)
    pw['p_adj'] = multipletests(pw['p_raw'], method='fdr_bh')[1]
    pw.to_csv(OUT_DIR / "pairwise_phylum_unknown_weighted_perm.csv", index=False)
    print("\nPairwise weighted permutation test (BH-FDR):")
    print(pw.to_string(index=False))


if __name__ == "__main__":
    main()
