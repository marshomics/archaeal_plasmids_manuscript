#!/usr/bin/env python3
"""All-pairs Fisher exact on viral prevalence across phyla; Holm correction."""
from itertools import combinations
from scipy import stats

from common import load_data, header, holm_bonferroni


def main():
    header("PHYLUM PAIRWISE FISHER")
    df, mob, *_ = load_data()
    viral_ids = set(df['replicon'].unique())
    phyla = sorted(mob['gtdb_phylum'].unique())

    table = {}
    for phy in phyla:
        ids = set(mob.loc[mob['gtdb_phylum'] == phy, 'sample_id'])
        viral = sum(1 for pid in ids if pid in viral_ids)
        total = len(ids)
        table[phy] = (viral, total - viral, total)
        print(f"  {phy:<26} {viral:>3}/{total:<3} ({viral/total*100:.1f}%)")

    pairwise = []
    for p1, p2 in combinations(phyla, 2):
        v1, nv1, _ = table[p1]
        v2, nv2, _ = table[p2]
        OR, p = stats.fisher_exact([[v1, nv1], [v2, nv2]])
        label = f"{p1.replace('p__','')} vs {p2.replace('p__','')}"
        pairwise.append((label, p, OR))

    corrected = holm_bonferroni([(lab, p) for lab, p, _ in pairwise])
    or_map = {lab: OR for lab, _, OR in pairwise}
    print(f"\nAll {len(corrected)} pairwise tests (Holm-corrected):")
    for label, p_raw, p_adj, sig in corrected:
        marker = "*" if sig else ""
        print(f"  {label}: OR = {or_map[label]:.2f}, "
              f"p_adj = {p_adj:.2e} {marker}")


if __name__ == "__main__":
    main()
