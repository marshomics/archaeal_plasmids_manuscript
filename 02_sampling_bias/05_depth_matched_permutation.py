#!/usr/bin/env python3
"""Permutation O/E with carrier status shuffled within depth bins.

Two-sided p-values (mass at least as extreme as |obs - exp|) and
BH-FDR adjustment across the tested carrier phyla. Also writes a CSV.
"""
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from common import load_data, header


def _depth_bin(n):
    if n == 1:    return '1'
    if n <= 3:    return '2-3'
    if n <= 10:   return '4-10'
    return '≥11'


def main():
    reps, _, _ = load_data()
    header("DEPTH-MATCHED O/E PERMUTATION (TWO-SIDED, BH-FDR)")

    reps = reps.copy()
    reps['depth_bin'] = reps['n_genomes'].apply(_depth_bin)

    observed = reps.groupby('gtdb_phylum')['is_carrier'].sum()

    # expected = sum over depth bins of (phylum bin size × bin-level base rate)
    expected = {}
    for phy in reps['gtdb_phylum'].unique():
        exp = 0.0
        for db in reps['depth_bin'].unique():
            bin_df = reps[reps['depth_bin'] == db]
            bin_rate = bin_df['is_carrier'].mean()
            n_phy_bin = len(reps[(reps['gtdb_phylum'] == phy) &
                                 (reps['depth_bin'] == db)])
            exp += n_phy_bin * bin_rate
        expected[phy] = exp

    n_perm = 10_000
    rng = np.random.default_rng(42)
    perm_counts = {phy: np.zeros(n_perm) for phy in reps['gtdb_phylum'].unique()}
    for i in range(n_perm):
        shuffled = reps['is_carrier'].to_numpy().copy()
        for db in reps['depth_bin'].unique():
            mask = (reps['depth_bin'] == db).to_numpy()
            shuffled[mask] = rng.permutation(shuffled[mask])
        tmp = reps.assign(perm=shuffled).groupby('gtdb_phylum')['perm'].sum()
        for phy in perm_counts:
            perm_counts[phy][i] = tmp.get(phy, 0)

    carrier_phyla = sorted(p for p in observed.index if observed.get(p, 0) > 0)
    print(f"Permutations: {n_perm}\n")

    rows = []
    for phy in carrier_phyla:
        obs = observed.get(phy, 0)
        exp = expected[phy]
        oe = obs / exp if exp > 0 else float('inf')
        deviation = abs(obs - exp)
        # two-sided p = fraction of permuted phyla counts as far or further from exp
        perm_dev = np.abs(perm_counts[phy] - exp)
        p_two = ((perm_dev >= deviation).sum() + 1) / (n_perm + 1)
        rows.append({'phylum': phy, 'observed': obs, 'expected': round(exp, 1),
                     'O_over_E': round(oe, 2), 'p_two_sided': p_two})

    df = pd.DataFrame(rows)
    df['p_BH'] = multipletests(df['p_two_sided'], method='fdr_bh')[1]
    df = df.sort_values('p_BH')

    # Write CSV next to other outputs
    from pathlib import Path
    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / "depth_matched_permutation.csv", index=False)

    print(f"{'Phylum':<28} {'Obs':>5} {'Exp':>8} {'O/E':>7} {'p_two':>10} {'p_BH':>10}")
    print("-" * 74)
    for _, r in df.iterrows():
        p_str = f"{r['p_two_sided']:.4f}" if r['p_two_sided'] > 1/(n_perm+1) else f"<{1/(n_perm+1):.4f}"
        q_str = f"{r['p_BH']:.4f}"
        print(f"{r['phylum']:<28} {int(r['observed']):>5} {r['expected']:>8.1f} "
              f"{r['O_over_E']:>7.2f} {p_str:>10} {q_str:>10}")


if __name__ == "__main__":
    main()
