#!/usr/bin/env python3
"""Phylum-level defence carriage; Halo vs non-Halo, species-weighted.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import fisher_exact

from common import load_defense_tables, OUT_DIR, header

N_PERM = 10_000
SEED = 42


def _weighted_perm_diff(y, group, w, n_perm=N_PERM, seed=SEED):
    """Two-sided permutation p on weighted difference in proportions.
       group: boolean array (True = focal)."""
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    g = np.asarray(group, dtype=bool)

    def _wdiff(mask):
        wA = w[mask]
        wB = w[~mask]
        pA = np.sum(y[mask] * wA) / np.sum(wA) if wA.sum() > 0 else 0.0
        pB = np.sum(y[~mask] * wB) / np.sum(wB) if wB.sum() > 0 else 0.0
        return pA - pB

    obs = _wdiff(g)
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(n_perm):
        mask = rng.permutation(g)
        if abs(_wdiff(mask)) >= abs(obs):
            extreme += 1
    p = (extreme + 1) / (n_perm + 1)
    return obs, p


def main():
    header("DEFENCE CARRIAGE BY PHYLUM (species-weighted)")
    type_df, *_ = load_defense_tables()
    type_df = type_df.copy()
    type_df['has_def'] = (type_df['n_instances'] > 0).astype(int)

    # Per-phylum unweighted and weighted prevalence
    rows = []
    for phy in type_df['phylum'].unique():
        sub = type_df[type_df['phylum'] == phy]
        n = len(sub)
        with_def = int(sub['has_def'].sum())
        wmean = float(np.average(sub['has_def'], weights=sub['weight'])) if n else 0.0
        rows.append({
            'phylum': phy,
            'n_plasmids': n,
            'n_with_defense': with_def,
            'pct_with_defense_unweighted': round(100 * with_def / n, 1),
            'pct_with_defense_species_weighted': round(100 * wmean, 1),
        })
    df = pd.DataFrame(rows).sort_values('pct_with_defense_species_weighted',
                                        ascending=False)
    df.to_csv(OUT_DIR / 'defense_pct_by_phylum.csv', index=False)
    print(df.to_string(index=False))

    # Halo vs non-Halo: weighted permutation primary, plain Fisher reference
    halo = type_df['phylum'] == 'Halobacteriota'
    h_n = int(halo.sum())
    n_n = int((~halo).sum())
    h_yes = int((type_df.loc[halo, 'n_instances'] > 0).sum())
    n_yes = int((type_df.loc[~halo, 'n_instances'] > 0).sum())

    h_w = float(np.average(type_df.loc[halo, 'has_def'],
                           weights=type_df.loc[halo, 'weight']))
    n_w = float(np.average(type_df.loc[~halo, 'has_def'],
                           weights=type_df.loc[~halo, 'weight']))
    diff_w, p_perm = _weighted_perm_diff(type_df['has_def'].values,
                                         halo.values,
                                         type_df['weight'].values)

    OR_f, p_f = fisher_exact([[h_yes, h_n - h_yes],
                              [n_yes, n_n - n_yes]])

    print(f"\nHalo plasmids:        {h_yes}/{h_n} unweighted = {h_yes/h_n*100:.1f}%, "
          f"species-weighted = {h_w*100:.1f}%")
    print(f"Non-Halo plasmids:    {n_yes}/{n_n} unweighted = {n_yes/n_n*100:.1f}%, "
          f"species-weighted = {n_w*100:.1f}%")
    print(f"\nWeighted-permutation Halo vs non-Halo "
          f"(two-sided, {N_PERM} iters):")
    print(f"  Δ(weighted) = {diff_w*100:.1f} pp,  p = {p_perm:.2e}")
    print(f"\nFisher (unweighted, two-sided, reference):")
    print(f"  OR = {OR_f:.2f}, p = {p_f:.2e}")

    pd.DataFrame([{
        'h_yes': h_yes, 'h_n': h_n,
        'n_yes': n_yes, 'n_n': n_n,
        'halo_unweighted_pct': round(h_yes / h_n * 100, 1),
        'non_halo_unweighted_pct': round(n_yes / n_n * 100, 1),
        'halo_weighted_pct': round(h_w * 100, 1),
        'non_halo_weighted_pct': round(n_w * 100, 1),
        'weighted_diff_pp': round(diff_w * 100, 1),
        'p_perm_weighted_two_sided': p_perm,
        'fisher_OR_unweighted': OR_f,
        'p_fisher_two_sided': p_f,
    }]).to_csv(OUT_DIR / 'defense_halo_vs_nonhalo.csv', index=False)


if __name__ == "__main__":
    main()
