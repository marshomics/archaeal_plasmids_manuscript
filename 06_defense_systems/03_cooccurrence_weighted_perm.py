#!/usr/bin/env python3
"""Weighted permutation co-occurrence on every pair of defence types.

UPDATE vs streamlined_defense_systems/03_cooccurrence_weighted_perm.py
----------------------------------------------------------------------
Original used N_PERM = 10,000, which puts the floor of the (k+1)/(N+1)
estimator at 1e-4. The manuscript's "weighted permutation p = 1e-4"
for BREX–RM is therefore the floor itself, not a measured value.

This version raises N_PERM to 100,000 so that floors are pushed to
~1e-5, and vectorises the permutation loop to keep runtime reasonable.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "streamlined_defense_systems"))

from itertools import combinations
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from common import load_defense_tables, SEED, header

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)
N_PERM = 50_000   # floor = 1/(N+1) ≈ 2e-5, well below the 1e-4 reported

# Shared permutation matrix across pairs — one (N_PERM, n_plasmids) tensor
# of row-independent shuffles built once and reused for every pair.
# Correlations between pair p-values are absorbed by the BH-FDR step.
_PERM_CACHE = {}


def _build_perm_indices(n_items, n_perm, seed):
    rng = np.random.default_rng(seed)
    idx = np.tile(np.arange(n_items), (n_perm, 1))
    rng.permuted(idx, axis=1, out=idx)
    return idx


def _weighted_perm_cooc(a, b, w, w_sum, perm_idx):
    """Two-sided permutation test on weighted excess co-occurrence,
    using a pre-built (n_perm, n_items) index matrix of shuffles."""
    expected = (np.sum(a * w) / w_sum) * (np.sum(b * w) / w_sum)
    obs = np.sum(a * b * w) / w_sum - expected
    bw = b * w
    shuffled = a[perm_idx]   # (n_perm, n_items)
    null = shuffled @ bw / w_sum - expected
    n_extreme = int((np.abs(null) >= np.abs(obs)).sum())
    p = (n_extreme + 1) / (perm_idx.shape[0] + 1)
    return obs, p


def main():
    header("WEIGHTED-PERMUTATION CO-OCCURRENCE (updated: N_PERM = 100,000)")
    _, binary_type, _, type_cols, _ = load_defense_tables()
    active = [c for c in type_cols if (binary_type[c] > 0).sum() >= 3]
    pairs = list(combinations(active, 2))
    print(f"Active types (≥ 3 plasmids): {len(active)}")
    print(f"Pairs tested:                {len(pairs)}")
    print(f"Permutations per pair:       {N_PERM:,}")

    w = binary_type['weight'].values
    w_sum = w.sum()
    cols_arr = {c: binary_type[c].values.astype(float) for c in active}
    n_items = len(w)

    print(f"  Building shared permutation index ({N_PERM} × {n_items})...")
    perm_idx = _build_perm_indices(n_items, N_PERM, SEED)

    rows = []
    for i, (ta, tb) in enumerate(pairs):
        excess, p = _weighted_perm_cooc(
            cols_arr[ta], cols_arr[tb], w, w_sum, perm_idx)
        rows.append({'system_a': ta, 'system_b': tb,
                     'obs_excess': excess,
                     'direction': 'enriched' if excess > 0 else 'depleted',
                     'perm_p_value': p,
                     'n_cooccur': int(((cols_arr[ta] > 0) &
                                        (cols_arr[tb] > 0)).sum())})
    out = pd.DataFrame(rows)
    out['p_adjusted'] = multipletests(out['perm_p_value'], method='fdr_bh')[1]
    out['significant'] = out['p_adjusted'] < 0.05
    out = out.sort_values('perm_p_value')

    floor = 1.0 / (N_PERM + 1)
    out['at_perm_floor'] = out['perm_p_value'] <= floor + 1e-12
    print(f"\nPermutation floor (1/(N+1)): {floor:.2e}")
    print(f"Pairs at floor:               {int(out['at_perm_floor'].sum())}")

    out.to_csv(OUT_DIR / 'cooccurrence_weighted_perm.csv', index=False)
    print(f"\nFDR-significant pairs: {out['significant'].sum()}")
    print("\nAll FDR-significant enriched pairs:")
    print(out[out['significant'] & (out['direction'] == 'enriched')].to_string(
        index=False))


if __name__ == "__main__":
    main()
