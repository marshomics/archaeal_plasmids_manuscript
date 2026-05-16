#!/usr/bin/env python3
"""Validate that five-subtype topology is recovered from Clinker sequence identity.

Loads the 154×154 Clinker BLAST-derived pairwise distance matrix and the
MMseqs2 protein-cluster-based subtype assignments.  Performs hierarchical
clustering (Ward's method) on the distance matrix and cuts the dendrogram
at k=5 to compare with the HDBSCAN subtypes.  Reports Adjusted Rand Index
(ARI) and a confusion matrix between the two partitions.

This supports the Extended Data Fig 4A,B claim that the same five-subtype
topology emerges regardless of whether gene-content (MMseqs2) or direct
sequence identity (Clinker) is used.
"""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from common import CLINKER_MATRIX, CLUSTER_CSV, excluded_replicons, OUT_DIR, header


def _ari(labels_a, labels_b):
    """Adjusted Rand Index — sklearn-free implementation."""
    from collections import Counter
    from math import comb

    pairs = list(zip(labels_a, labels_b))
    contingency = Counter(pairs)

    # Row and column sums
    row_sums = Counter()
    col_sums = Counter()
    for (a, b), n in contingency.items():
        row_sums[a] += n
        col_sums[b] += n

    n = len(labels_a)
    sum_comb_c = sum(comb(v, 2) for v in contingency.values())
    sum_comb_a = sum(comb(v, 2) for v in row_sums.values())
    sum_comb_b = sum(comb(v, 2) for v in col_sums.values())
    comb_n = comb(n, 2)

    expected = sum_comb_a * sum_comb_b / comb_n if comb_n else 0
    max_index = 0.5 * (sum_comb_a + sum_comb_b)
    denom = max_index - expected

    if denom == 0:
        return 1.0 if sum_comb_c == expected else 0.0
    return (sum_comb_c - expected) / denom


def main():
    header("DUAL-METRIC TOPOLOGY VALIDATION (Clinker vs. MMseqs2)")

    if not CLINKER_MATRIX.exists():
        print("  [skipped — filtered_gbk_matrix.csv not found]")
        return

    # Load distance matrix
    dist_df = pd.read_csv(CLINKER_MATRIX, index_col=0)
    print(f"Clinker distance matrix: {dist_df.shape[0]} × {dist_df.shape[1]}")

    # Load subtype assignments and apply exclusions
    clust = pd.read_csv(CLUSTER_CSV)
    excl = excluded_replicons()
    if excl:
        # Exclude based on accession prefix (before _v suffix)
        clust = clust[~clust['plasmid'].apply(
            lambda x: x.rsplit('_v', 1)[0] if '_v' in x else x
        ).isin(excl)].copy()
    print(f"Subtype assignments (post-exclusion): {len(clust)}")

    # Align: keep only matrix rows/cols that have subtype labels
    clust_ids = set(clust['plasmid'])
    matrix_ids = [idx for idx in dist_df.index if idx in clust_ids]
    n_aligned = len(matrix_ids)
    print(f"Fragments in both matrix and cluster file: {n_aligned}")

    if n_aligned < 10:
        print("  [too few overlapping fragments — cannot validate]")
        return

    # Subset and ensure symmetric
    D = dist_df.loc[matrix_ids, matrix_ids].values
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0)

    # Hierarchical clustering (Ward's method on the distance matrix)
    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method='ward')
    hier_labels = fcluster(Z, t=5, criterion='maxclust')

    # Get HDBSCAN labels in the same order
    label_map = dict(zip(clust['plasmid'], clust['hdbscan_cluster']))
    hdbscan_labels = np.array([label_map[pid] for pid in matrix_ids])

    # Compute ARI
    ari = _ari(hdbscan_labels.tolist(), hier_labels.tolist())
    print(f"\nAdjusted Rand Index (HDBSCAN vs Ward k=5): {ari:.3f}")

    # Confusion matrix
    st_names = sorted(set(hdbscan_labels))
    hier_names = sorted(set(hier_labels))
    print(f"\nConfusion matrix (rows = HDBSCAN subtype, cols = Ward cluster):")
    print(f"{'':>6}", end='')
    for hc in hier_names:
        print(f"  W{hc}", end='')
    print()
    for st in st_names:
        mask_st = hdbscan_labels == st
        print(f"  ST{st}", end='')
        for hc in hier_names:
            mask_hc = hier_labels == hc
            count = int((mask_st & mask_hc).sum())
            print(f"  {count:3d}", end='')
        print()

    # Per-subtype purity (fraction of HDBSCAN subtype in majority Ward cluster)
    print(f"\nPer-subtype purity (majority overlap):")
    purities = []
    for st in st_names:
        mask_st = hdbscan_labels == st
        st_hier = hier_labels[mask_st]
        majority = max(np.bincount(st_hier)[1:])  # fcluster is 1-indexed
        purity = majority / mask_st.sum()
        purities.append(purity)
        print(f"  ST{st}: {majority}/{mask_st.sum()} = {purity:.1%}")
    mean_purity = np.mean(purities)
    print(f"  Mean purity: {mean_purity:.1%}")

    # Save alignment table
    out_df = pd.DataFrame({
        'plasmid': matrix_ids,
        'hdbscan_subtype': [f"ST{s}" for s in hdbscan_labels],
        'ward_cluster': [f"W{c}" for c in hier_labels],
    })
    out_df.to_csv(OUT_DIR / "dual_metric_cluster_comparison.csv", index=False)
    print(f"\nSaved: outputs/dual_metric_cluster_comparison.csv")
    print(f"       (ARI = {ari:.3f}; mean purity = {mean_purity:.1%})")


if __name__ == "__main__":
    main()
