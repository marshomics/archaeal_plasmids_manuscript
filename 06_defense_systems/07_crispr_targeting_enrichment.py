#!/usr/bin/env python3
"""Plasmid- vs viral-targeting: pooled χ² + per-kb rate ratio + per-array Wilcoxon.

Wilcoxon signed-rank tests each array's observed plasmid-target fraction
against the null expected fraction (plasmid_kb / total_kb), which controls
for the fact that a few high-hit arrays dominate the pooled χ².
"""
import numpy as np
import pandas as pd
from scipy.stats import chisquare, wilcoxon

from common import BLAST_TSV, SEQ_SPACE, OUT_DIR, header


def main():
    header("PLASMID vs VIRAL TARGETING ENRICHMENT")
    hits = pd.read_csv(BLAST_TSV, sep='\t')
    seq  = pd.read_csv(SEQ_SPACE, sep='\t')

    n_plasmid_hits = int((hits['target_category'] == 'plasmid').sum())
    n_virus_hits   = int((hits['target_category'] == 'virus').sum())
    total = n_plasmid_hits + n_virus_hits
    plasmid_kb = float(seq.loc[seq['category'] == 'plasmid', 'total_kb'].iloc[0])
    virus_kb   = float(seq.loc[seq['category'] == 'virus',   'total_kb'].iloc[0])
    total_kb   = plasmid_kb + virus_kb

    expected_p_frac = plasmid_kb / total_kb
    expected_v_frac = virus_kb / total_kb
    expected = [expected_p_frac * total, expected_v_frac * total]
    chi2, p = chisquare([n_plasmid_hits, n_virus_hits], f_exp=expected)
    fold = (n_plasmid_hits / total) / expected_p_frac

    print(f"Plasmid hits:           {n_plasmid_hits}")
    print(f"Virus hits:             {n_virus_hits}")
    print(f"Plasmid reference Mb:   {plasmid_kb/1000:.1f}")
    print(f"Virus reference Mb:     {virus_kb/1000:.1f}  "
          f"({virus_kb/total_kb*100:.1f}%)")
    print(f"Expected plasmid frac:  {expected_p_frac:.3f}")
    print(f"\nExpected plasmid hits if random: {expected[0]:.1f}")
    print(f"Observed:                        {n_plasmid_hits}")
    print(f"Fold enrichment:                 {fold:.2f}x")
    print(f"χ² goodness-of-fit:              χ² = {chi2:.1f}, p = {p:.2e}")

    rate_p = n_plasmid_hits / plasmid_kb
    rate_v = n_virus_hits / virus_kb if virus_kb > 0 else float('nan')
    rate_ratio = rate_p / rate_v if rate_v > 0 else float('inf')
    print(f"\nHits / kb (plasmid): {rate_p:.4f}")
    print(f"Hits / kb (virus):   {rate_v:.4f}")
    print(f"Per-kb rate ratio:   {rate_ratio:.1f}x")

    # per-array Wilcoxon: each source plasmid's observed plasmid-target
    # fraction vs the expected fraction from reference composition
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

    W, p_w = wilcoxon(per_array['diff_from_expected'], alternative='greater')
    print(f"\nWilcoxon signed-rank (per-array plasmid fraction vs "
          f"{expected_p_frac:.3f}):")
    print(f"  Arrays:        {len(per_array)}")
    print(f"  Median frac:   {per_array['frac_plasmid'].median():.3f}")
    print(f"  Median diff:   {per_array['diff_from_expected'].median():.3f}")
    print(f"  W = {W:.1f}, p = {p_w:.2e}")

    per_array.to_csv(OUT_DIR / 'per_array_targeting.csv', index=False)
    pd.DataFrame([{
        'n_plasmid_hits': n_plasmid_hits, 'n_virus_hits': n_virus_hits,
        'plasmid_kb': plasmid_kb, 'virus_kb': virus_kb,
        'expected_plasmid_frac': round(expected_p_frac, 4),
        'fold_enrichment': round(fold, 2),
        'chi2': round(chi2, 2), 'p_value': p,
        'rate_ratio_per_kb': round(rate_ratio, 1),
        'wilcoxon_W': round(W, 1), 'wilcoxon_p': p_w,
        'n_arrays': len(per_array),
    }]).to_csv(OUT_DIR / 'crispr_targeting_enrichment.csv', index=False)


if __name__ == "__main__":
    main()
