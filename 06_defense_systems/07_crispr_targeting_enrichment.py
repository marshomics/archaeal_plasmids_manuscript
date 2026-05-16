#!/usr/bin/env python3
"""Plasmid- vs viral-targeting: per-array test as primary, pooled as secondary.

UPDATE vs streamlined_defense_systems/07_crispr_targeting_enrichment.py
-----------------------------------------------------------------------
Original computed both a pooled χ² (treating every spacer hit as
independent) and a per-array Wilcoxon. The manuscript leads with the
pooled p = 4.7e-270, which inflates significance via pseudo-replication
within arrays.

This version inverts the emphasis:
  (a) the per-array Wilcoxon is the headline test;
  (b) the per-array median plasmid-target fraction and its
      bootstrap 95% CI are the headline effect size;
  (c) the pooled χ² is retained but flagged as secondary / inflated by
      within-array dependence.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "streamlined_defense_systems"))

import numpy as np
import pandas as pd
from scipy.stats import chisquare, wilcoxon

from common import BLAST_TSV, SEQ_SPACE, header

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)
N_BOOT = 5000
SEED = 42


def main():
    header("CRISPR TARGETING (updated: per-array as primary)")
    hits = pd.read_csv(BLAST_TSV, sep='\t')
    seq  = pd.read_csv(SEQ_SPACE, sep='\t')

    n_plasmid_hits = int((hits['target_category'] == 'plasmid').sum())
    n_virus_hits   = int((hits['target_category'] == 'virus').sum())
    total = n_plasmid_hits + n_virus_hits
    plasmid_kb = float(seq.loc[seq['category'] == 'plasmid', 'total_kb'].iloc[0])
    virus_kb   = float(seq.loc[seq['category'] == 'virus',   'total_kb'].iloc[0])
    total_kb   = plasmid_kb + virus_kb
    expected_p_frac = plasmid_kb / total_kb

    # Per-array first: each source plasmid contributes one observation.
    per_array = (
        hits.groupby('source_plasmid')['target_category']
            .value_counts()
            .unstack(fill_value=0)
            .reset_index()
    )
    for c in ('plasmid', 'virus'):
        if c not in per_array.columns:
            per_array[c] = 0
    per_array['total'] = per_array['plasmid'] + per_array['virus']
    per_array = per_array[per_array['total'] > 0].copy()
    per_array['frac_plasmid'] = per_array['plasmid'] / per_array['total']
    per_array['diff_from_expected'] = per_array['frac_plasmid'] - expected_p_frac

    # Wilcoxon signed-rank against expected fraction.
    W, p_w = wilcoxon(per_array['diff_from_expected'], alternative='greater')

    # Bootstrap 95% CI for median plasmid-target fraction across arrays.
    rng = np.random.default_rng(SEED)
    arr_frac = per_array['frac_plasmid'].values
    boot_medians = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, len(arr_frac), len(arr_frac))
        boot_medians[i] = np.median(arr_frac[idx])
    ci_lo, ci_hi = np.percentile(boot_medians, [2.5, 97.5])

    print(f"HEADLINE (per-array test, replaces pooled p = 4.7e-270):")
    print(f"  Arrays with ≥ 1 hit:               {len(per_array)}")
    print(f"  Expected plasmid-target fraction:  {expected_p_frac:.3f}  "
          f"(plasmid_kb / total_kb)")
    print(f"  Median per-array plasmid fraction: {np.median(arr_frac):.3f}")
    print(f"  Bootstrap 95% CI for median:       "
          f"[{ci_lo:.3f}, {ci_hi:.3f}]  ({N_BOOT} resamples)")
    print(f"  Median deviation from expected:    "
          f"{per_array['diff_from_expected'].median():.3f}")
    print(f"  Wilcoxon signed-rank (greater):    "
          f"W = {W:.1f}, p = {p_w:.2e}")

    # Arrays exclusively targeting plasmids.
    excl = int((per_array['virus'] == 0).sum())
    print(f"  Arrays hitting only plasmids:      "
          f"{excl}/{len(per_array)} ({excl/len(per_array)*100:.0f}%)")

    # Secondary: pooled χ² with flag.
    expected = [expected_p_frac * total, (1 - expected_p_frac) * total]
    chi2, p = chisquare([n_plasmid_hits, n_virus_hits], f_exp=expected)
    fold = (n_plasmid_hits / total) / expected_p_frac
    print(f"\nSECONDARY (pooled χ², inflated by within-array dependence):")
    print(f"  Plasmid hits: {n_plasmid_hits}, virus hits: {n_virus_hits}")
    print(f"  Fold enrichment: {fold:.2f}x")
    print(f"  χ² = {chi2:.1f}, p = {p:.2e}  [DO NOT use as primary evidence]")

    rate_p = n_plasmid_hits / plasmid_kb
    rate_v = n_virus_hits / virus_kb if virus_kb > 0 else float('nan')
    rate_ratio = rate_p / rate_v if rate_v > 0 else float('inf')
    print(f"  Per-kb rate ratio (plasmid/virus): {rate_ratio:.1f}x")

    per_array.to_csv(OUT_DIR / 'per_array_targeting.csv', index=False)
    pd.DataFrame([{
        'primary_test': 'per-array Wilcoxon signed-rank (greater)',
        'n_arrays': len(per_array),
        'expected_plasmid_frac': round(expected_p_frac, 4),
        'median_array_frac': round(np.median(arr_frac), 3),
        'median_array_frac_ci_lo': round(ci_lo, 3),
        'median_array_frac_ci_hi': round(ci_hi, 3),
        'wilcoxon_W': round(W, 1),
        'wilcoxon_p': p_w,
        'arrays_only_plasmid': excl,
        'secondary_pooled_chi2': round(chi2, 2),
        'secondary_pooled_p': p,
        'fold_enrichment_pooled': round(fold, 2),
        'rate_ratio_per_kb': round(rate_ratio, 1),
    }]).to_csv(OUT_DIR / 'crispr_targeting_enrichment.csv', index=False)


if __name__ == "__main__":
    main()
