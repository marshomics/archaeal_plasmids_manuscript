#!/usr/bin/env python3
"""VirB4 + T4CP co-occurrence across the plasmid catalogue.

Paired-region count = min(VirB4 hits, T4CP hits) on the filtered table —
pairs every VirB4 ORF to one T4CP ORF on the same replicon.
"""
import pandas as pd

from common import (UNFILTERED_HITS, FILTERED_HITS,
                    total_plasmids_in_catalogue, excluded_replicons, header)


def main():
    header("VirB4 / T4CP CO-OCCURRENCE")
    n_catalogue = total_plasmids_in_catalogue()
    excl = excluded_replicons()
    print(f"Catalogue size: {n_catalogue}")
    print(f"Exclusion list: {len(excl)}")

    unfilt = pd.read_csv(UNFILTERED_HITS, sep='\t')

    virb4_reps = set(unfilt.loc[
        unfilt['gene_name'].str.contains('virb4', case=False), 'replicon_name'])
    t4cp_reps  = set(unfilt.loc[
        unfilt['gene_name'].str.contains('t4cp',  case=False), 'replicon_name'])
    both       = virb4_reps & t4cp_reps
    any_hit    = virb4_reps | t4cp_reps

    print(f"\nReplicons with any VirB4 or T4CP hit: {len(any_hit)}")
    print(f"Replicons encoding both:              {len(both)} "
          f"({len(both)/n_catalogue*100:.1f}% of catalogue)")

    conj = pd.read_csv(FILTERED_HITS, sep='\t')
    n_v4 = int((conj['gene_name'].str.contains('virb4', case=False)).sum())
    n_tc = int((conj['gene_name'].str.contains('t4cp',  case=False)).sum())
    n_regions = min(n_v4, n_tc)
    print(f"\nVirB4 hits in filtered set: {n_v4}")
    print(f"T4CP hits in filtered set:  {n_tc}")
    print(f"Paired VirB4-T4CP regions:  {n_regions}")
    print(f"Plasmids with ≥ 1 region:   {len(set(conj['replicon_name']))}")

    after_excl = set(conj['replicon_name']) - excl
    print(f"After exclusion list:       {len(after_excl)}")


if __name__ == "__main__":
    main()
