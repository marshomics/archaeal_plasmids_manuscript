#!/usr/bin/env python3
"""Phylum-level defence carriage; Halo vs non-Halo Fisher."""
import pandas as pd
from scipy.stats import fisher_exact

from common import load_defense_tables, OUT_DIR, header


def main():
    header("DEFENCE CARRIAGE BY PHYLUM")
    type_df, *_ = load_defense_tables()

    rows = []
    for phy in type_df['phylum'].unique():
        sub = type_df[type_df['phylum'] == phy]
        n = len(sub)
        with_def = int((sub['n_instances'] > 0).sum())
        rows.append({'phylum': phy, 'n': n, 'n_with_defense': with_def,
                     'pct_with_defense': round(100 * with_def / n, 1)})
    df = pd.DataFrame(rows).sort_values('pct_with_defense', ascending=False)
    df.to_csv(OUT_DIR / 'defense_pct_by_phylum.csv', index=False)
    print(df.to_string(index=False))

    halo = type_df[type_df['phylum'] == 'Halobacteriota']
    non  = type_df[type_df['phylum'] != 'Halobacteriota']
    h_yes = int((halo['n_instances'] > 0).sum())
    n_yes = int((non['n_instances']  > 0).sum())
    OR, p = fisher_exact([[h_yes, len(halo) - h_yes],
                          [n_yes, len(non)  - n_yes]])
    print(f"\nHalo:     {h_yes}/{len(halo)} ({h_yes/len(halo)*100:.1f}%)")
    print(f"Non-Halo: {n_yes}/{len(non)}  ({n_yes/len(non)*100:.1f}%)")
    print(f"Fisher OR = {OR:.2f}, p = {p:.2e}")


if __name__ == "__main__":
    main()
