#!/usr/bin/env python3
"""Defence count vs plasmid size with a size-permutation null for density.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "streamlined_defense_systems"))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import load_defense_tables, MOB_FILE, header

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)
N_PERM = 5000
SEED = 42


def main():
    header("DEFENCE × PLASMID SIZE (updated: permutation null for density)")
    type_df, *_ = load_defense_tables()

    mob = pd.read_csv(MOB_FILE, sep='\t')
    size_col = next((c for c in ('size', 'replicon_size', 'plasmid_size',
                                  'size_bp', 'length') if c in mob.columns), None)
    sizes = mob[['sample_id', size_col]].rename(
        columns={'sample_id': 'replicon', size_col: 'size_bp'})
    df = type_df.merge(sizes, on='replicon', how='inner')
    df['size_kb'] = df['size_bp'] / 1000.0

    rho_count, p_count = spearmanr(df['size_kb'], df['n_instances'])
    print(f"Spearman (count vs size): ρ = {rho_count:.3f}, "
          f"p = {p_count:.2e}, n = {len(df)}")

    with_def = df[df['n_instances'] > 0].copy()
    with_def['density_per_kb'] = with_def['n_instances'] / with_def['size_kb']
    rho_d, p_d = spearmanr(with_def['size_kb'], with_def['density_per_kb'])
    print(f"\nSpearman (density vs size on carriers):")
    print(f"  Observed ρ = {rho_d:.3f}, parametric p = {p_d:.2e}, "
          f"n = {len(with_def)}")

    # Permutation null: shuffle size labels and recompute density and ρ.
    rng = np.random.default_rng(SEED)
    counts = with_def['n_instances'].values.astype(float)
    sizes_arr = with_def['size_kb'].values.astype(float)
    null_rhos = np.empty(N_PERM)
    for i in range(N_PERM):
        s_perm = rng.permutation(sizes_arr)
        d_perm = counts / s_perm
        null_rhos[i] = spearmanr(s_perm, d_perm)[0]
    null_p = (np.sum(np.abs(null_rhos) >= np.abs(rho_d)) + 1) / (N_PERM + 1)
    pct95 = np.percentile(np.abs(null_rhos), 95)
    print(f"  Permutation null (N = {N_PERM}, size labels shuffled):")
    print(f"    null ρ mean ± sd:   {null_rhos.mean():.3f} ± "
          f"{null_rhos.std():.3f}")
    print(f"    null ρ 95th |abs|:  {pct95:.3f}")
    print(f"    empirical p:        {null_p:.4f}")
    print(f"    |observed| / |null 95th|: {abs(rho_d)/pct95:.2f}x")
    if abs(rho_d) > pct95:
        print(f"  → Observed correlation exceeds the arithmetic null; "
              f"reports as ρ = {rho_d:.3f} (permutation p = {null_p:.3g}).")
    else:
        print(f"  → Observed correlation lies within the arithmetic null; "
              f"the manuscript's density-vs-size negative trend is largely "
              f"explained by shared denominator.")

    pd.DataFrame({
        'metric': ['count_vs_size', 'density_vs_size'],
        'spearman_rho': [rho_count, rho_d],
        'parametric_p': [p_count, p_d],
        'permutation_p': [np.nan, null_p],
    }).to_csv(OUT_DIR / 'defense_vs_size_perm.csv', index=False)


if __name__ == "__main__":
    main()
