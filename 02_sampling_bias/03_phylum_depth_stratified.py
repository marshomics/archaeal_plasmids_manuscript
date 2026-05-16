#!/usr/bin/env python3
"""Halo vs non-Halo carrier rate within each depth bin."""
from common import load_data, header, DEPTH_BINS


def main():
    reps, _, _ = load_data()
    header("HALO vs NON-HALO BY DEPTH")
    reps = reps.copy()
    reps['is_halo'] = (reps['gtdb_phylum'] == 'p__Halobacteriota').astype(int)

    print(f"{'Depth':>6}  {'Halo carriers':>18}  {'Non-Halo carriers':>22}  {'Ratio':>6}")
    for low, high, label in DEPTH_BINS:
        sub = reps[(reps['n_genomes'] >= low) & (reps['n_genomes'] <= high)]
        h = sub[sub['is_halo'] == 1]
        o = sub[sub['is_halo'] == 0]
        h_rate = h['is_carrier'].mean() * 100 if len(h) else 0.0
        o_rate = o['is_carrier'].mean() * 100 if len(o) else 0.0
        ratio = h_rate / o_rate if o_rate > 0 else float('inf')
        print(f"{label:>6}  {int(h['is_carrier'].sum())}/{len(h)} ({h_rate:5.1f}%)"
              f"   {int(o['is_carrier'].sum())}/{len(o)} ({o_rate:5.1f}%)"
              f"   {ratio:>5.1f}x")


if __name__ == "__main__":
    main()
