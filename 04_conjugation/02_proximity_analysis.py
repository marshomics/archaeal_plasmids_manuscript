#!/usr/bin/env python3
"""Nucleotide-distance proximity between VirB4 and T4CP on each plasmid.

"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "streamlined_conjugation"))

from itertools import product as iterproduct
import pandas as pd

from common import FILTERED_HITS, excluded_replicons, header


def _ngap(v_begin, v_end, t_begin, t_end):
    """Nucleotide gap between two intervals; 0 if they overlap."""
    return max(0, max(v_begin, t_begin) - min(v_end, t_end))


def main():
    header("VirB4 / T4CP PROXIMITY (updated: nucleotide distance)")
    conj = pd.read_csv(FILTERED_HITS, sep='\t')
    conj = conj[~conj['replicon_name'].isin(excluded_replicons())]

    virb4 = conj[conj['gene_name'].str.contains('virb4', case=False)]
    t4cp  = conj[conj['gene_name'].str.contains('t4cp',  case=False)]
    both_reps = set(virb4['replicon_name']) & set(t4cp['replicon_name'])

    rows = []
    for rep in sorted(both_reps):
        v_sub = virb4.loc[virb4['replicon_name'] == rep,
                          ['position_hit', 'begin', 'end']].values
        t_sub = t4cp .loc[t4cp ['replicon_name'] == rep,
                          ['position_hit', 'begin', 'end']].values
        best_orf = None
        best_nt = None
        for vrow in v_sub:
            for trow in t_sub:
                orf_d = abs(int(vrow[0]) - int(trow[0]))
                nt_d  = _ngap(int(vrow[1]), int(vrow[2]),
                              int(trow[1]), int(trow[2]))
                if best_orf is None or orf_d < best_orf:
                    best_orf = orf_d
                if best_nt is None or nt_d < best_nt:
                    best_nt = nt_d
        rows.append({'replicon': rep,
                     'min_orf_distance': best_orf,
                     'min_nt_gap': best_nt})
    prox = pd.DataFrame(rows)
    n = len(prox)
    print(f"Plasmids with both VirB4 and T4CP:  {n}")

    # ORF-index thresholds.
    fused_orf       = (prox['min_orf_distance'] == 0).sum()
    consecutive_orf = (prox['min_orf_distance'] == 1).sum()
    within_2_orf    = (prox['min_orf_distance'] <= 2).sum()
    print(f"\nORF-index distance (manuscript-style):")
    print(f"  Fused ORFs (orf_d == 0):        {fused_orf}  "
          f"({fused_orf/n*100:.1f}%)")
    print(f"  Consecutive ORFs (orf_d == 1):  {consecutive_orf}  "
          f"({consecutive_orf/n*100:.1f}%)")
    print(f"  ≤ 1 ORF apart (fused+consec):   "
          f"{fused_orf + consecutive_orf}  "
          f"({(fused_orf + consecutive_orf)/n*100:.1f}%)")
    print(f"  ≤ 2 ORFs apart:                 "
          f"{within_2_orf}  ({within_2_orf/n*100:.1f}%)")

    # Nucleotide thresholds.
    tight_nt_500  = (prox['min_nt_gap'] <= 500).sum()
    nt_2kb       = (prox['min_nt_gap'] <= 2000).sum()
    nt_5kb       = (prox['min_nt_gap'] <= 5000).sum()
    print(f"\nNucleotide gap between gene boundaries:")
    print(f"  Overlapping or ≤ 500 bp:        "
          f"{tight_nt_500}  ({tight_nt_500/n*100:.1f}%)")
    print(f"  ≤ 2 kb:                          "
          f"{nt_2kb}  ({nt_2kb/n*100:.1f}%)")
    print(f"  ≤ 5 kb:                          "
          f"{nt_5kb}  ({nt_5kb/n*100:.1f}%)")
    print(f"  Median nt gap:                   "
          f"{prox['min_nt_gap'].median():.0f} bp")

    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    prox.to_csv(out_dir / 'proximity_per_replicon.csv', index=False)
    pd.DataFrame([{
        'n_replicons': n,
        'pct_consec_orf_or_fused': round(
            (fused_orf + consecutive_orf) / n * 100, 1),
        'pct_nt_le_500bp': round(tight_nt_500 / n * 100, 1),
        'pct_nt_le_2kb':  round(nt_2kb / n * 100, 1),
        'pct_nt_le_5kb':  round(nt_5kb / n * 100, 1),
        'median_nt_gap':  int(prox['min_nt_gap'].median()),
    }]).to_csv(out_dir / 'proximity_summary.csv', index=False)


if __name__ == "__main__":
    main()
