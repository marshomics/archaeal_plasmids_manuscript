#!/usr/bin/env python3
"""Phylum carrier rates within isolate-only and complete-genome-only subsets.
"""
import pandas as pd
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests

from common import load_data, header


def _phylum_rate(df, phy):
    sub = df[df['gtdb_phylum'] == phy]
    return int(sub['is_carrier'].sum()), len(sub)


def _fisher_vs_halo(df, target):
    h_c, h_n = _phylum_rate(df, 'p__Halobacteriota')
    t_c, t_n = _phylum_rate(df, target)
    if h_n == 0 or t_n == 0:
        return None
    OR, p = stats.fisher_exact([[h_c, h_n - h_c], [t_c, t_n - t_c]])
    return h_c, h_n, t_c, t_n, OR, p


def main():
    reps, _, reps_meta = load_data()
    header("QUALITY-SUBSET ROBUSTNESS (BH-FDR across subsets × phyla)")

    carrier_phyla = sorted(
        reps.loc[reps['is_carrier'] == 1, 'gtdb_phylum'].dropna().unique())

    subsets = {
        'All species (with NCBI metadata)': reps_meta,
        'Isolate genomes only':            reps_meta[reps_meta['ncbi_genome_category'] == 'none'],
        'Complete genomes only':           reps_meta[reps_meta['ncbi_assembly_level'] == 'Complete Genome'],
    }

    # Print descriptive rates per subset
    for name, df in subsets.items():
        print(f"\n--- {name} (n = {len(df)}) ---")
        for phy in carrier_phyla:
            c, n = _phylum_rate(df, phy)
            rate = c / n * 100 if n else 0.0
            print(f"  {phy:<26} {c}/{n} ({rate:.1f}%)")

    # Collect all Fisher tests, then BH-correct
    halo_label = 'p__Halobacteriota'
    rows = []
    for name, df in subsets.items():
        for target in (p for p in carrier_phyla if p != halo_label):
            res = _fisher_vs_halo(df, target)
            if res is None:
                continue
            h_c, h_n, t_c, t_n, OR, p = res
            rows.append({
                'subset': name, 'target': target,
                'halo_carrier': h_c, 'halo_n': h_n,
                'target_carrier': t_c, 'target_n': t_n,
                'OR': OR, 'p_raw': p,
            })

    fisher = pd.DataFrame(rows)
    fisher['p_BH'] = multipletests(fisher['p_raw'], method='fdr_bh')[1]
    fisher = fisher.sort_values(['subset', 'p_BH'])

    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    fisher.to_csv(out_dir / "quality_subset_fisher_bh.csv", index=False)

    print("\n--- Fisher Halo vs each carrier phylum (BH-FDR across all tests) ---")
    for _, r in fisher.iterrows():
        marker = " *" if r['p_BH'] < 0.05 else ""
        print(f"  [{r['subset']:<35}] {r['target']:<26} "
              f"{r['halo_carrier']}/{r['halo_n']} vs {r['target_carrier']}/{r['target_n']} "
              f"OR = {r['OR']:.2f}, p = {r['p_raw']:.2e}, q = {r['p_BH']:.2e}{marker}")

    print("\nMAG proportion per carrier phylum:")
    for phy in carrier_phyla:
        sub = reps_meta[reps_meta['gtdb_phylum'] == phy]
        n_mag = int((sub['ncbi_genome_category'] == 'derived from metagenome').sum())
        n = len(sub)
        if n:
            print(f"  {phy:<26} {n_mag}/{n} ({n_mag/n*100:.1f}% MAG)")


if __name__ == "__main__":
    main()
