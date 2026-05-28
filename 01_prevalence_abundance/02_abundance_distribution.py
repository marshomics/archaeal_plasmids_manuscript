#!/usr/bin/env python3
"""Per-carrier plasmid abundance, Halo vs non-Halo, Fisher on single-plasmid carriers.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "streamlined_results"))

from scipy import stats
import numpy as np
from common import load_data, header


def _fisher_single_vs_multi(halo, nonhalo, cutoff_label, halo_max=None,
                            nonhalo_max=None):
    halo_in = halo if halo_max is None else halo[halo['plasmid_abundance'] <= halo_max]
    nonhalo_in = nonhalo if nonhalo_max is None else nonhalo[nonhalo['plasmid_abundance'] <= nonhalo_max]
    s_h = int((halo_in['plasmid_abundance'] == 1).sum())
    m_h = int((halo_in['plasmid_abundance'] > 1).sum())
    s_n = int((nonhalo_in['plasmid_abundance'] == 1).sum())
    m_n = int((nonhalo_in['plasmid_abundance'] > 1).sum())
    OR, p = stats.fisher_exact([[s_h, m_h], [s_n, m_n]])
    return {
        'rule': cutoff_label,
        'halo_n': len(halo_in),
        'halo_single_pct': 100 * s_h / max(1, s_h + m_h),
        'nonhalo_n': len(nonhalo_in),
        'nonhalo_single_pct': 100 * s_n / max(1, s_n + m_n),
        'OR': OR,
        'p': p,
    }


def main():
    reps, _ = load_data()
    header("ABUNDANCE DISTRIBUTION (updated: Tukey rule + sensitivity sweep)")
    carriers = reps[reps['is_carrier'] == 1]
    halo = carriers[carriers['gtdb_phylum'] == 'p__Halobacteriota']
    nonhalo = carriers[carriers['gtdb_phylum'] != 'p__Halobacteriota']

    # Descriptive stats first (unchanged).
    print(f"Total carriers:           {len(carriers)}")
    print(f"Halobacteriota carriers:  {len(halo)}")
    print(f"Non-Halo carriers:        {len(nonhalo)}")

    # Tukey upper-fence computed on the full carrier distribution.
    q1, q3 = np.percentile(carriers['plasmid_abundance'], [25, 75])
    iqr = q3 - q1
    tukey_15 = q3 + 1.5 * iqr
    tukey_30 = q3 + 3.0 * iqr
    print(f"\nTukey fences on full carrier distribution:")
    print(f"  Q1 = {q1:.1f}, Q3 = {q3:.1f}, IQR = {iqr:.1f}")
    print(f"  Upper fence (Q3 + 1.5*IQR): {tukey_15:.1f}")
    print(f"  Far-outlier fence (Q3 + 3*IQR): {tukey_30:.1f}")

    # Sensitivity sweep: same Fisher test under four cutoffs.
    rules = [
        ('no exclusion', None),
        (f'Tukey 1.5*IQR (≤ {tukey_15:.1f})', tukey_15),
        (f'Tukey 3*IQR (≤ {tukey_30:.1f})', tukey_30),
        ('manuscript cutoff (≤ 4)', 4),
    ]
    print("\nSensitivity sweep — Fisher single vs >1 plasmid:")
    print(f"  {'rule':<32}  {'Halo single %':>13}  {'Non-Halo single %':>17}  "
          f"{'OR':>6}  {'p':>10}")
    sweep_rows = []
    for label, cutoff in rules:
        r = _fisher_single_vs_multi(halo, nonhalo, label, cutoff, cutoff)
        sweep_rows.append(r)
        print(f"  {r['rule']:<32}  {r['halo_single_pct']:>12.1f}%  "
              f"{r['nonhalo_single_pct']:>16.1f}%  {r['OR']:>6.2f}  "
              f"{r['p']:>10.2e}")

    # Save sweep.
    import pandas as pd
    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    pd.DataFrame(sweep_rows).to_csv(out_dir / 'abundance_fisher_sensitivity.csv',
                                    index=False)

    # Headline number for manuscript: use Tukey 1.5*IQR.
    headline = sweep_rows[1]
    print(f"\nHEADLINE (Tukey 1.5*IQR rule, replaces manuscript p = 0.0004):")
    print(f"  Halo single: {headline['halo_single_pct']:.1f}%; "
          f"Non-Halo single: {headline['nonhalo_single_pct']:.1f}%; "
          f"OR = {headline['OR']:.2f}; p = {headline['p']:.4f}")


if __name__ == "__main__":
    main()
