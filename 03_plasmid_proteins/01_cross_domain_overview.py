#!/usr/bin/env python3
"""Cluster-type counts, archaeal protein share, cross-domain vs domain-restricted sizes."""
import pandas as pd
from common import CLUSTER_SUMMARY, header


def main():
    header("CROSS-DOMAIN OVERVIEW")
    cl = pd.read_csv(CLUSTER_SUMMARY)

    ct = cl['cluster_type'].value_counts()
    n_total = len(cl)
    print("Cluster types:")
    for k, v in ct.items():
        print(f"  {k:<14} {v:>9,}  ({v/n_total*100:.2f}%)")

    cd = cl[cl['cluster_type'] == 'cross-domain']
    ao = cl[cl['cluster_type'] == 'archaea-only']
    arch_in_cd = cd['n_archaea'].sum()
    arch_in_ao = ao['n_archaea'].sum()
    arch_total = arch_in_cd + arch_in_ao
    print("\nArchaeal protein share:")
    print(f"  Cross-domain: {arch_in_cd:>9,}  ({arch_in_cd/arch_total*100:.1f}%)")
    print(f"  Archaea-only: {arch_in_ao:>9,}  ({arch_in_ao/arch_total*100:.1f}%)")

    domain_restricted = cl[cl['cluster_type'] != 'cross-domain']
    print("\nCluster size:")
    print(f"  Cross-domain:      median = {cd['n_total'].median():.0f}, "
          f"mean = {cd['n_total'].mean():.1f}, n = {len(cd):,}")
    print(f"  Domain-restricted: median = {domain_restricted['n_total'].median():.0f}, "
          f"mean = {domain_restricted['n_total'].mean():.1f}, n = {len(domain_restricted):,}")


if __name__ == "__main__":
    main()
