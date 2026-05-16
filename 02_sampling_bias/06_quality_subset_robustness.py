#!/usr/bin/env python3
"""Phylum carrier rates within isolate-only and complete-genome-only subsets."""
from scipy import stats

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
    header("QUALITY-SUBSET ROBUSTNESS")

    carrier_phyla = sorted(
        reps.loc[reps['is_carrier'] == 1, 'gtdb_phylum'].dropna().unique())

    subsets = {
        'All species (with NCBI metadata)': reps_meta,
        'Isolate genomes only':            reps_meta[reps_meta['ncbi_genome_category'] == 'none'],
        'Complete genomes only':           reps_meta[reps_meta['ncbi_assembly_level'] == 'Complete Genome'],
    }

    for name, df in subsets.items():
        print(f"\n--- {name} (n = {len(df)}) ---")
        for phy in carrier_phyla:
            c, n = _phylum_rate(df, phy)
            rate = c / n * 100 if n else 0.0
            print(f"  {phy:<26} {c}/{n} ({rate:.1f}%)")

        halo_label = 'p__Halobacteriota'
        for target in (p for p in carrier_phyla if p != halo_label):
            res = _fisher_vs_halo(df, target)
            if res is None:
                continue
            h_c, h_n, t_c, t_n, OR, p = res
            print(f"  Fisher Halo vs {target}: "
                  f"{h_c}/{h_n} vs {t_c}/{t_n}  OR = {OR:.2f}, p = {p:.2e}")

    print("\nMAG proportion per carrier phylum:")
    for phy in carrier_phyla:
        sub = reps_meta[reps_meta['gtdb_phylum'] == phy]
        n_mag = int((sub['ncbi_genome_category'] == 'derived from metagenome').sum())
        n = len(sub)
        if n:
            print(f"  {phy:<26} {n_mag}/{n} ({n_mag/n*100:.1f}% MAG)")


if __name__ == "__main__":
    main()
