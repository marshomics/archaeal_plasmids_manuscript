#!/usr/bin/env python3
"""Weighted permutation co-occurrence on every pair of defence types.

Tests excess of observed weighted co-occurrence over the product of weighted
marginals. Restricted to types appearing on ≥ 3 plasmids; BH-FDR across pairs.
"""
from itertools import combinations
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from common import load_defense_tables, OUT_DIR, SEED, N_PERM, header


def _weighted_perm_cooc(df, a_col, b_col, n_perm, seed):
    a = df[a_col].values.astype(float)
    b = df[b_col].values.astype(float)
    w = df['weight'].values
    w_sum = w.sum()
    expected = (np.sum(a * w) / w_sum) * (np.sum(b * w) / w_sum)
    obs = np.sum(a * b * w) / w_sum - expected
    rng = np.random.RandomState(seed)
    null = np.zeros(n_perm)
    for i in range(n_perm):
        null[i] = np.sum(rng.permutation(a) * b * w) / w_sum - expected
    p = (np.sum(np.abs(null) >= np.abs(obs)) + 1) / (n_perm + 1)
    return obs, p


def main():
    header("WEIGHTED-PERMUTATION CO-OCCURRENCE")
    _, binary_type, _, type_cols, _ = load_defense_tables()
    active = [c for c in type_cols if (binary_type[c] > 0).sum() >= 3]
    pairs = list(combinations(active, 2))
    print(f"Active types (≥ 3 plasmids): {len(active)}")
    print(f"Pairs tested:                {len(pairs)}")

    rows = []
    for i, (ta, tb) in enumerate(pairs):
        excess, p = _weighted_perm_cooc(binary_type, ta, tb, N_PERM, SEED + i)
        rows.append({'system_a': ta, 'system_b': tb,
                     'obs_excess': excess,
                     'direction': 'enriched' if excess > 0 else 'depleted',
                     'perm_p_value': p,
                     'n_cooccur': int(((binary_type[ta] > 0) &
                                        (binary_type[tb] > 0)).sum())})
    out = pd.DataFrame(rows)
    out['p_adjusted'] = multipletests(out['perm_p_value'], method='fdr_bh')[1]
    out['significant'] = out['p_adjusted'] < 0.05
    out = out.sort_values('perm_p_value')
    out.to_csv(OUT_DIR / 'cooccurrence_weighted_perm.csv', index=False)
    print(f"\nFDR-significant pairs: {out['significant'].sum()}")
    print("\nAll FDR-significant enriched pairs:")
    print(out[out['significant'] & (out['direction'] == 'enriched')].to_string(index=False))


if __name__ == "__main__":
    main()
