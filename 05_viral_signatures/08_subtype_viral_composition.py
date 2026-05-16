#!/usr/bin/env python3
"""Viral complexity by VirB4-T4CP subtype: Kruskal-Wallis + Dunn post-hoc.

Uses the deduplicated cluster file (one row per plasmid).  Two plasmids
assigned to multiple subtypes (CP002989.1 → ST1/ST2; NZ_CP084474.1 →
ST2/ST4) are excluded as ambiguous.  Same-subtype duplicates are resolved
by retaining the row with the highest n_conserved_clusters.
"""
import warnings
warnings.filterwarnings('ignore')

from itertools import combinations
import scikit_posthocs as sp
from scipy import stats

from common import load_data, header


def main():
    header("VIRAL COMPLEXITY BY SUBTYPE (deduplicated)")
    df, _, clusters, conj_list, complexity, _ = load_data()

    # ── Deduplicate ────────────────────────────────────────────────
    # 1. Remove plasmids assigned to more than one subtype
    multi_st = (clusters.groupby('short_label')['hdbscan_cluster']
                .nunique().pipe(lambda s: s[s > 1]).index)
    if len(multi_st) > 0:
        print(f"Excluded {len(multi_st)} ambiguous plasmid(s) assigned to "
              f"multiple subtypes: {', '.join(sorted(multi_st))}")
        clusters = clusters[~clusters['short_label'].isin(multi_st)].copy()

    # 2. For same-subtype duplicates keep the best-supported row
    n_before = len(clusters)
    clusters = (clusters.sort_values('n_conserved_clusters', ascending=False)
                .drop_duplicates(subset='short_label', keep='first')
                .sort_values(['hdbscan_cluster', 'short_label'])
                .reset_index(drop=True))
    n_removed = n_before - len(clusters)
    if n_removed > 0:
        print(f"Removed {n_removed} same-subtype duplicate row(s) "
              f"(kept highest n_conserved_clusters)")

    print(f"Final: {len(clusters)} unique plasmids across "
          f"{clusters['hdbscan_cluster'].nunique()} subtypes\n")

    # ── Complexity assignment ──────────────────────────────────────
    clusters['complexity'] = (clusters['short_label']
                              .map(complexity).fillna(0).astype(int))
    subtypes = sorted(clusters['hdbscan_cluster'].unique())

    for st in subtypes:
        sub = clusters[clusters['hdbscan_cluster'] == st]
        c = sub['complexity']
        zero = int((c == 0).sum())
        print(f"  ST{st}: n = {len(sub)}, mean = {c.mean():.2f}, "
              f"median = {c.median():.1f}, range = {c.min()}-{c.max()}, "
              f"zero_viral = {zero}")

    # ── Kruskal-Wallis omnibus ─────────────────────────────────────
    groups = [clusters.loc[clusters['hdbscan_cluster'] == st,
                           'complexity'].values
              for st in subtypes]
    H, p = stats.kruskal(*groups)
    print(f"\nKruskal-Wallis: H = {H:.2f}, p = {p:.2e}")

    # ── Dunn post-hoc (Holm) ──────────────────────────────────────
    dunn_df = clusters[['hdbscan_cluster', 'complexity']].copy()
    dunn_df['hdbscan_cluster'] = 'ST' + dunn_df['hdbscan_cluster'].astype(str)
    dunn = sp.posthoc_dunn(dunn_df, val_col='complexity',
                           group_col='hdbscan_cluster', p_adjust='holm')
    print("\nDunn post-hoc (Holm):")
    labels = [f"ST{s}" for s in subtypes]
    for s1, s2 in combinations(labels, 2):
        adj = dunn.loc[s1, s2]
        marker = "*" if adj < 0.05 else ""
        print(f"  {s1} vs {s2}: p_adj = {adj:.2e} {marker}")

    # ── Category prevalence per subtype ────────────────────────────
    all_cats = sorted(df['new_category'].unique())
    print("\nCategory prevalence per subtype:")
    for st in subtypes:
        sub_ids = set(clusters.loc[clusters['hdbscan_cluster'] == st,
                                   'short_label'])
        sub_viral = df[df['replicon'].isin(sub_ids)]
        n = len(sub_ids)
        print(f"  ST{st} (n = {n}):")
        for cat in all_cats:
            k = sub_viral[sub_viral['new_category'] == cat]['replicon'].nunique()
            if k > 0 or cat in ('Capsid & head', 'DNA packaging'):
                print(f"    {cat:<24} {k:>3} ({k/n*100:.1f}%)")

    # ── Capsid / packaging exclusion ──────────────────────────────
    print("\nCapsid / packaging on conjugative plasmids:")
    conj_viral = df[df['replicon'].isin(conj_list)]
    for cat in ('Capsid & head', 'DNA packaging'):
        k = conj_viral[conj_viral['new_category'] == cat]['replicon'].nunique()
        print(f"  {cat}: {k}/{len(conj_list)}")


if __name__ == "__main__":
    main()
