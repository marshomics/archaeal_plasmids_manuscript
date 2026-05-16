#!/usr/bin/env python3
"""Phylum / family rollup for VirB4-T4CP plasmids."""
import pandas as pd

from common import FILTERED_HITS, excluded_replicons, OUT_DIR, header


def main():
    header("TAXONOMY OF CONJUGATIVE PLASMIDS")
    conj = pd.read_csv(FILTERED_HITS, sep='\t')
    conj = conj[~conj['replicon_name'].isin(excluded_replicons())]
    tax = conj.drop_duplicates('replicon_name')[
        ['replicon_name', 'gtdb_phylum', 'gtdb_class', 'gtdb_order',
         'gtdb_family', 'gtdb_genus', 'gtdb_species']
    ].copy()

    print("Phylum breakdown:")
    for phy, n in tax['gtdb_phylum'].value_counts().items():
        print(f"  {phy}: {n}")

    halo  = tax[tax['gtdb_phylum'] == 'p__Halobacteriota']
    sulfo = tax[tax['gtdb_family'] == 'f__Sulfolobaceae']

    print(f"\nSulfolobaceae:           {len(sulfo)}")
    print(f"Halobacteriota:          {len(halo)}")
    print(f"Halo families covered:   {halo['gtdb_family'].nunique()}")

    print("\nHalobacteriota by family:")
    for fam, n in halo['gtdb_family'].value_counts().items():
        print(f"  {fam}: {n}")

    tax.to_csv(OUT_DIR / "conjugative_plasmid_taxonomy.csv", index=False)


if __name__ == "__main__":
    main()
