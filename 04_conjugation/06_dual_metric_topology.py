#!/usr/bin/env python3
"""Validate the five-subtype topology against an independent metric.

"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "streamlined_conjugation"))

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform, pdist

from common import CLINKER_MATRIX, CLUSTER_CSV, excluded_replicons, header

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def _ari(labels_a, labels_b):
    from collections import Counter
    from math import comb
    contingency = Counter(zip(labels_a, labels_b))
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
    denom = 0.5 * (sum_comb_a + sum_comb_b) - expected
    if denom == 0:
        return 1.0 if sum_comb_c == expected else 0.0
    return (sum_comb_c - expected) / denom


def _silhouette(D, labels):
    """Mean silhouette over all points; D is a square distance matrix."""
    labels = np.asarray(labels)
    unique = np.unique(labels)
    n = len(labels)
    s = np.zeros(n)
    for i in range(n):
        own = labels[i]
        same = (labels == own) & (np.arange(n) != i)
        if same.sum() == 0:
            s[i] = 0.0
            continue
        a = D[i, same].mean()
        b_options = []
        for u in unique:
            if u == own:
                continue
            mask = labels == u
            if mask.sum() == 0:
                continue
            b_options.append(D[i, mask].mean())
        if not b_options:
            s[i] = 0.0
            continue
        b = min(b_options)
        s[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return float(s.mean())


def main():
    header("DUAL-METRIC TOPOLOGY (updated: independent k-selection)")

    if not CLINKER_MATRIX.exists():
        print("  [skipped — filtered_gbk_matrix.csv not found]")
        return

    dist_df = pd.read_csv(CLINKER_MATRIX, index_col=0)
    print(f"Clinker distance matrix: {dist_df.shape[0]} × {dist_df.shape[1]}")

    clust = pd.read_csv(CLUSTER_CSV)
    excl = excluded_replicons()
    if excl:
        clust = clust[~clust['plasmid'].apply(
            lambda x: x.rsplit('_v', 1)[0] if '_v' in x else x
        ).isin(excl)].copy()

    clust_ids = set(clust['plasmid'])
    matrix_ids = [idx for idx in dist_df.index if idx in clust_ids]
    n = len(matrix_ids)
    print(f"Fragments in both matrix and cluster file: {n}")
    if n < 10:
        print("  [too few overlapping fragments]")
        return

    D = dist_df.loc[matrix_ids, matrix_ids].values
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0)
    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method='average')   # average linkage — does not
                                               # assume Euclidean, unlike Ward

    label_map = dict(zip(clust['plasmid'], clust['hdbscan_cluster']))
    hdbscan_labels = np.array([label_map[pid] for pid in matrix_ids])

    # Sweep k.
    print(f"\nk-sweep (k = 2..10) on Clinker distance matrix:")
    print(f"  {'k':>3}  {'silhouette':>11}  {'ARI vs HDBSCAN':>15}  "
          f"{'Ward heights (top 3)':>21}")
    sweep = []
    for k in range(2, 11):
        labels = fcluster(Z, t=k, criterion='maxclust')
        sil = _silhouette(D, labels)
        ari = _ari(hdbscan_labels.tolist(), labels.tolist())
        sweep.append({'k': k, 'silhouette': sil, 'ari_vs_hdbscan': ari})
        print(f"  {k:>3}  {sil:>11.3f}  {ari:>15.3f}")

    best_sil = max(sweep, key=lambda r: r['silhouette'])
    best_ari = max(sweep, key=lambda r: r['ari_vs_hdbscan'])
    print(f"\nClinker-preferred k by silhouette:  k = {best_sil['k']}  "
          f"(silhouette = {best_sil['silhouette']:.3f})")
    print(f"Clinker-preferred k by ARI to HDBSCAN: k = {best_ari['k']}  "
          f"(ARI = {best_ari['ari_vs_hdbscan']:.3f})")
    if best_sil['k'] == 5:
        print("  → Clinker independently selects k=5; topology claim supported.")
    else:
        print(f"  → Clinker does not select k=5 on its own. "
              f"The 'same five-subtype topology' claim should be reframed as "
              f"\"with k=5 imposed, the two partitions agree at ARI = "
              f"{[r for r in sweep if r['k']==5][0]['ari_vs_hdbscan']:.3f}\".")

    pd.DataFrame(sweep).to_csv(OUT_DIR / 'k_sweep_clinker.csv', index=False)

    # Imposed-k=5 confusion matrix (kept for diagnostic).
    hier_labels = fcluster(Z, t=5, criterion='maxclust')
    print("\nConfusion (HDBSCAN rows × Ward-k5 cols, IMPOSED at k=5):")
    print(f"{'':>6}", end='')
    for hc in sorted(set(hier_labels)):
        print(f"  W{hc}", end='')
    print()
    for st in sorted(set(hdbscan_labels)):
        mask_st = hdbscan_labels == st
        print(f"  ST{st}", end='')
        for hc in sorted(set(hier_labels)):
            count = int((mask_st & (hier_labels == hc)).sum())
            print(f"  {count:3d}", end='')
        print()


if __name__ == "__main__":
    main()
