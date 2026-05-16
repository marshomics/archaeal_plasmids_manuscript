#!/usr/bin/env python3
"""Per-phylum share of archaeal proteins in cross-domain clusters; Nanoarchaeota check."""
import pandas as pd
from common import CLUSTER_SUMMARY, header


def main():
    header("ARCHAEAL SHARING PER PHYLUM")
    cl = pd.read_csv(CLUSTER_SUMMARY)
    cl['archaea_phyla'] = cl['archaea_phyla'].fillna("")

    arch = cl[cl['n_archaea'] > 0].copy()
    cd = arch[arch['cluster_type'] == 'cross-domain']

    # split mixed-phylum cross-domain clusters equally across constituent phyla
    cd_exploded = cd.assign(phylum=cd['archaea_phyla'].str.split('|')).explode('phylum')
    cd_exploded['n_archaea'] = cd_exploded['n_archaea'] / cd_exploded.groupby(
        cd_exploded.index)['phylum'].transform('count')
    by_phy_cd = cd_exploded.groupby('phylum')['n_archaea'].sum().sort_values(ascending=False)
    total_arch_cd = by_phy_cd.sum()
    print("Per-phylum contribution to cross-domain clusters:")
    for phy, n in by_phy_cd.items():
        print(f"  {phy:<26} {n:>10,.0f}  ({n/total_arch_cd*100:.1f}%)")

    # Nanoarchaeota — all-or-nothing check (obligate-symbiont expectation)
    nano = cl[cl['archaea_phyla'].str.contains('Nanoarchaeota', na=False)]
    nano_cd = nano[nano['cluster_type'] == 'cross-domain']
    print("\nNanoarchaeota plasmid clusters:")
    print(f"  Total:        {len(nano)}")
    print(f"  Cross-domain: {len(nano_cd)}  "
          f"({len(nano_cd)/max(len(nano),1)*100:.1f}%)")


if __name__ == "__main__":
    main()
