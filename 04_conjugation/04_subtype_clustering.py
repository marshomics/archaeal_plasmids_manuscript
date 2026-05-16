#!/usr/bin/env python3
"""HDBSCAN subtype × family table (clustering is upstream)."""
import pandas as pd

from common import CLUSTER_CSV, FILTERED_HITS, excluded_replicons, OUT_DIR, header


def main():
    header("SUBTYPE × FAMILY")
    clust = pd.read_csv(CLUSTER_CSV)
    excl = excluded_replicons()
    if excl:
        # exclusion IDs may be partial stems
        mask = clust['plasmid'].apply(lambda p: any(e in p for e in excl))
        clust = clust[~mask]
        print(f"Fragments after excluding {len(excl)} replicons: {len(clust)}")
    else:
        print(f"Fragments: {len(clust)}")

    print("\nSubtype sizes:")
    sizes = clust['hdbscan_cluster'].value_counts().sort_index()
    for st, n in sizes.items():
        print(f"  ST{st}: {n}")
    print(f"  Total: {sizes.sum()}")

    conj = pd.read_csv(FILTERED_HITS, sep='\t')
    tax = conj.drop_duplicates('replicon_name')[
        ['replicon_name', 'gtdb_phylum', 'gtdb_family']]
    clust_tax = clust.merge(tax, left_on='short_label',
                            right_on='replicon_name', how='left')

    print("\nSubtype × Family:")
    for st in sorted(clust_tax['hdbscan_cluster'].dropna().unique()):
        sub = clust_tax[clust_tax['hdbscan_cluster'] == st]
        fams = sub['gtdb_family'].value_counts()
        phyla = sub['gtdb_phylum'].value_counts()
        n_phyla = sub['gtdb_phylum'].dropna().nunique()
        n_fams  = sub['gtdb_family'].dropna().nunique()
        confined = (f"single phylum ({phyla.index[0]})" if n_phyla == 1
                    else f"single family ({fams.index[0]})" if n_fams == 1
                    else "mixed")
        print(f"  ST{int(st)} (n = {len(sub)}, {confined}):")
        for fam, n in fams.items():
            print(f"    {fam}: {n}")

    clust_tax.to_csv(OUT_DIR / "subtype_family_assignments.csv", index=False)


if __name__ == "__main__":
    main()
