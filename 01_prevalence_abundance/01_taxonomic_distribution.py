#!/usr/bin/env python3
"""Carrier counts by phylum and Halobacteriota class breakdown."""
from common import load_data, header


def main():
    reps, _ = load_data()
    header("PREVALENCE / TAXONOMIC DISTRIBUTION")

    carriers = reps[reps['is_carrier'] == 1]
    n_total = len(reps)
    n_carriers = len(carriers)

    print(f"Representative species:         {n_total}")
    print(f"Carriers:                       {n_carriers} ({n_carriers/n_total*100:.1f}%)")
    print(f"Phyla in GTDB r214:             {reps['gtdb_phylum'].nunique()}")
    print(f"Phyla with at least 1 carrier:  {carriers['gtdb_phylum'].nunique()}")

    halo = carriers[carriers['gtdb_phylum'] == 'p__Halobacteriota']
    nonhalo = carriers[carriers['gtdb_phylum'] != 'p__Halobacteriota']
    print(f"\nHalobacteriota carriers:     {len(halo)} ({len(halo)/n_carriers*100:.1f}%)")
    print(f"Non-Halobacteriota carriers: {len(nonhalo)} ({len(nonhalo)/n_carriers*100:.1f}%)")

    halo_all = reps[reps['gtdb_phylum'] == 'p__Halobacteriota']
    halo_class = halo_all.groupby('gtdb_class').agg(
        n=('is_carrier', 'size'),
        nc=('is_carrier', 'sum'),
        tp=('plasmid_abundance', 'sum'),
    ).reset_index()
    print("\nHalobacteriota class-level:")
    for _, r in halo_class[halo_class['nc'] > 0].sort_values('nc', ascending=False).iterrows():
        print(f"  {r['gtdb_class']}: {int(r['nc'])}/{int(r['n'])} carriers, "
              f"{int(r['tp'])} plasmids "
              f"({r['nc']/len(halo)*100:.1f}% of Halo carriers)")

    print("\nPer-phylum carrier counts:")
    for phy in carriers['gtdb_phylum'].value_counts().index:
        sub = carriers[carriers['gtdb_phylum'] == phy]
        print(f"  {phy}: {len(sub)} species, {int(sub['plasmid_abundance'].sum())} plasmids")


if __name__ == "__main__":
    main()
