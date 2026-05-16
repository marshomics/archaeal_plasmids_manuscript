#!/usr/bin/env python3
"""Minimum ORF-position distance between VirB4 and T4CP on each plasmid."""
from itertools import product as iterproduct
import pandas as pd

from common import FILTERED_HITS, excluded_replicons, header


def main():
    header("VirB4 / T4CP PROXIMITY")
    conj = pd.read_csv(FILTERED_HITS, sep='\t')
    conj = conj[~conj['replicon_name'].isin(excluded_replicons())]

    virb4 = conj[conj['gene_name'].str.contains('virb4', case=False)]
    t4cp  = conj[conj['gene_name'].str.contains('t4cp',  case=False)]
    both_reps = set(virb4['replicon_name']) & set(t4cp['replicon_name'])

    rows = []
    for rep in sorted(both_reps):
        v4 = virb4.loc[virb4['replicon_name'] == rep, 'position_hit'].astype(int).values
        tc = t4cp .loc[t4cp ['replicon_name'] == rep, 'position_hit'].astype(int).values
        rows.append({'replicon': rep,
                     'min_distance': min(abs(v - t) for v, t in iterproduct(v4, tc))})
    prox = pd.DataFrame(rows)
    n = len(prox)

    fused    = (prox['min_distance'] == 0).sum()
    adjacent = (prox['min_distance'] == 1).sum()
    within_1 = (prox['min_distance'] <= 1).sum()

    print(f"Plasmids analysed:           {n}")
    print(f"Fused (distance 0):          {fused}  ({fused/n*100:.1f}%)")
    print(f"Adjacent (distance 1):       {adjacent}  ({adjacent/n*100:.1f}%)")
    print(f"Immediately adjacent (≤ 1):  {within_1}  ({within_1/n*100:.1f}%)")


if __name__ == "__main__":
    main()
