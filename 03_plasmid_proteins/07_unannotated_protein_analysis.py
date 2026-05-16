#!/usr/bin/env python3
"""Unannotated-protein burden: overall, per-plasmid, and per-phylum with bootstrap CIs."""
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from common import ARCHAEAL_EGGNOG, OUT_DIR, COG_DESCRIPTIONS, header

N_BOOT = 1000


def _is_unknown(cog):
    """True if no informative COG letter (only S or empty/-)."""
    if pd.isna(cog) or cog == '-' or str(cog).strip() == '':
        return True
    letters = [c for c in str(cog) if c in COG_DESCRIPTIONS]
    return (not letters) or all(c == 'S' for c in letters)


def main():
    header("UNANNOTATED-PROTEIN BURDEN")
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

    # pairwise Mann-Whitney with BH-FDR
    groups = [plas.loc[plas['phylum_short'] == p, 'pct_unknown'].to_numpy()
              for p in phyla]
    pairs = []
    for i in range(len(phyla)):
        for j in range(i + 1, len(phyla)):
            U, p_raw = stats.mannwhitneyu(groups[i], groups[j], alternative='two-sided')
            pairs.append({'group1': phyla[i], 'group2': phyla[j],
                          'U': U, 'p_raw': p_raw,
                          'n1': len(groups[i]), 'n2': len(groups[j])})
    pw = pd.DataFrame(pairs)
    pw['p_adj'] = multipletests(pw['p_raw'], method='fdr_bh')[1]
    pw.to_csv(OUT_DIR / "pairwise_phylum_unknown_MWU.csv", index=False)
    print("\nPairwise Mann-Whitney U (BH-FDR):")
    print(pw.to_string(index=False))


if __name__ == "__main__":
    main()
