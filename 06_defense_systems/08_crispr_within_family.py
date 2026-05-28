#!/usr/bin/env python3
"""Same-family CRISPR targeting with per-array (not per-hit) testing.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "streamlined_defense_systems"))

from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

from common import BLAST_TSV, header

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)
N_PERM = 5000
SEED = 42


def main():
    header("WITHIN-FAMILY CRISPR TARGETING (updated: per-array)")
    hits = pd.read_csv(BLAST_TSV, sep='\t')
    plasmid_hits = hits[hits['target_category'] == 'plasmid'].copy()
    n_total = len(plasmid_hits)
    if n_total == 0:
        print("No plasmid-target hits.")
        return

    plasmid_hits['same_family'] = (
        plasmid_hits['source_family'] == plasmid_hits['target_family'])

    # Reference: per-hit rate (kept for comparison).
    same_hit = int(plasmid_hits['same_family'].sum())
    print(f"Per-hit (anti-conservative, manuscript's framing):")
    print(f"  {same_hit}/{n_total} hits same-family "
          f"({same_hit/n_total*100:.1f}%)")
    fam_counts = plasmid_hits['target_family'].value_counts(normalize=True)
    expected_rate = float((fam_counts ** 2).sum())
    print(f"  Simpson null:    {expected_rate*100:.1f}%")
    print(f"  Fold enrichment: {(same_hit/n_total) / expected_rate:.2f}x")

    # Per-array: each source plasmid contributes one fraction.
    per_array = (plasmid_hits
                 .groupby('source_plasmid')
                 .agg(n_hits=('same_family', 'size'),
                      same_family=('same_family', 'sum'),
                      source_family=('source_family', 'first'))
                 .reset_index())
    per_array['frac_same'] = per_array['same_family'] / per_array['n_hits']
    print(f"\nPer-array headline (replaces manuscript's per-hit rate):")
    print(f"  Arrays:                       {len(per_array)}")
    print(f"  Median within-family frac:    "
          f"{per_array['frac_same'].median():.3f}")
    print(f"  Mean within-family frac:      "
          f"{per_array['frac_same'].mean():.3f}")
    print(f"  Arrays hitting only same family: "
          f"{int((per_array['frac_same'] == 1).sum())}/{len(per_array)}")

    # Permutation null: for each array, resample its n_hits target families
    # from the empirical target-family distribution.
    rng = np.random.default_rng(SEED)
    target_families = plasmid_hits['target_family'].values
    obs_mean = per_array['frac_same'].mean()
    null_means = np.empty(N_PERM)
    for i in range(N_PERM):
        fracs = []
        for _, r in per_array.iterrows():
            draws = rng.choice(target_families, size=int(r['n_hits']),
                               replace=True)
            fracs.append(np.mean(draws == r['source_family']))
        null_means[i] = float(np.mean(fracs))
    p_perm = (np.sum(null_means >= obs_mean) + 1) / (N_PERM + 1)
    print(f"  Permutation null (N = {N_PERM}, resample each array's hits):")
    print(f"    null mean ± sd: "
          f"{null_means.mean():.3f} ± {null_means.std():.3f}")
    print(f"    p (observed ≥ null): {p_perm:.4f}")

    # Per-family per-array test with BH-FDR.
    print("\nPer-family per-array enrichment (BH-FDR over families):")
    n_arrays_total = len(per_array)
    rows = []
    for fam, grp in per_array.groupby('source_family'):
        if len(grp) < 3:
            continue
        # Arrays in this family with any same-family hit vs without.
        n_fam_arrays = len(grp)
        n_fam_with_same = int((grp['same_family'] > 0).sum())
        # Compare to arrays NOT in this family: do they hit this family?
        other_arrays = per_array[per_array['source_family'] != fam]
        n_other = len(other_arrays)
        if n_other == 0:
            continue
        # Did each other-family array hit `fam` at all?
        other_hit_fam = 0
        for sp in other_arrays['source_plasmid']:
            sub = plasmid_hits[plasmid_hits['source_plasmid'] == sp]
            if (sub['target_family'] == fam).any():
                other_hit_fam += 1
        table = [[n_fam_with_same, n_fam_arrays - n_fam_with_same],
                 [other_hit_fam, n_other - other_hit_fam]]
        OR, p = fisher_exact(table, alternative='greater')
        rows.append({'source_family': fam,
                     'n_arrays': n_fam_arrays,
                     'arrays_with_same_family_hit': n_fam_with_same,
                     'other_arrays': n_other,
                     'other_arrays_hitting_this_family': other_hit_fam,
                     'OR': OR, 'p_raw': p})
    fam_df = pd.DataFrame(rows)
    if len(fam_df):
        fam_df['p_adj'] = multipletests(fam_df['p_raw'], method='fdr_bh')[1]
        fam_df = fam_df.sort_values('p_adj')
        print(fam_df.to_string(index=False))
        fam_df.to_csv(OUT_DIR / 'within_family_per_array.csv', index=False)

    pd.DataFrame([{
        'observed_mean_array_frac': round(obs_mean, 3),
        'null_mean_array_frac': round(null_means.mean(), 3),
        'permutation_p': round(p_perm, 4),
        'n_arrays': len(per_array),
        'per_hit_pct_same_family': round(same_hit/n_total*100, 1),
        'simpson_null_pct': round(expected_rate*100, 1),
    }]).to_csv(OUT_DIR / 'within_family_summary.csv', index=False)


if __name__ == "__main__":
    main()
