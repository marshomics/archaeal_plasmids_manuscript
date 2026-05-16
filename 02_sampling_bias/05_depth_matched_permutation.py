#!/usr/bin/env python3
"""Permutation O/E with carrier status shuffled within depth bins."""
import numpy as np

from common import load_data, header


def _depth_bin(n):
    if n == 1:    return '1'
    if n <= 3:    return '2-3'
    if n <= 10:   return '4-10'
    return '≥11'


def main():
    reps, _, _ = load_data()
    header("DEPTH-MATCHED O/E PERMUTATION")

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
    print(f"{'Phylum':<28} {'Obs':>5} {'Exp':>8} {'O/E':>7} {'p':>10}")
    print("-" * 64)
    for phy in carrier_phyla:
        obs = observed.get(phy, 0)
        exp = expected[phy]
        oe = obs / exp if exp > 0 else float('inf')
        if obs >= exp:
            p = (perm_counts[phy] >= obs).mean()
        else:
            p = (perm_counts[phy] <= obs).mean()
        p_str = f"{p:.4f}" if p > 0 else "<0.0001"
        print(f"{phy:<28} {obs:>5.0f} {exp:>8.1f} {oe:>7.2f} {p_str:>10}")


if __name__ == "__main__":
    main()
