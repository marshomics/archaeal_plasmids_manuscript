#!/usr/bin/env python3
"""Defence count vs plasmid size (Spearman); density vs size on carriers."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import load_defense_tables, MOB_FILE, OUT_DIR, header

SIZE_BINS  = [0, 10, 50, 100, 200, 500, np.inf]
SIZE_LABELS = ['<10', '10-50', '50-100', '100-200', '200-500', '>500']


def main():
    header("DEFENCE × PLASMID SIZE")
    type_df, *_ = load_defense_tables()

    mob = pd.read_csv(MOB_FILE, sep='\t')
    size_col = next((c for c in ('size', 'replicon_size', 'plasmid_size',
                                  'size_bp', 'length') if c in mob.columns), None)
    if size_col is None:
        raise RuntimeError("no size column in mobsuite table")
    sizes = mob[['sample_id', size_col]].rename(
        columns={'sample_id': 'replicon', size_col: 'size_bp'})
    df = type_df.merge(sizes, on='replicon', how='inner')
    df['size_kb'] = df['size_bp'] / 1000.0

    rho, p = spearmanr(df['size_kb'], df['n_instances'])
    print(f"Spearman (count vs size): ρ = {rho:.3f}, p = {p:.2e}, n = {len(df)}")

    df['size_bin'] = pd.cut(df['size_kb'], bins=SIZE_BINS, labels=SIZE_LABELS,
                            include_lowest=True)
    bin_summary = df.groupby('size_bin').agg(
        n=('n_instances', 'size'),
        n_with_defense=('n_instances', lambda x: (x > 0).sum()),
    ).reset_index()
    bin_summary['pct_with_defense'] = (
        100 * bin_summary['n_with_defense'] / bin_summary['n'])
    print("\nCarriage by size bin:")
    print(bin_summary.to_string(index=False))

    with_def = df[df['n_instances'] > 0].copy()
    with_def['density_per_kb'] = with_def['n_instances'] / with_def['size_kb']
    rho_d, p_d = spearmanr(with_def['size_kb'], with_def['density_per_kb'])
    print(f"\nSpearman (density vs size on carriers): "
          f"ρ = {rho_d:.3f}, p = {p_d:.2e}, n = {len(with_def)}")

    bin_summary.to_csv(OUT_DIR / 'carriage_by_size_bin.csv', index=False)
    with_def[['replicon', 'size_kb', 'n_instances', 'density_per_kb']].to_csv(
        OUT_DIR / 'density_per_plasmid.csv', index=False)


if __name__ == "__main__":
    main()
