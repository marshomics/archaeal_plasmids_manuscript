#!/usr/bin/env python3
"""Viral protein prevalence across the plasmid catalogue."""
import pandas as pd
from common import load_data, header, OUT_DIR


def main():
    header("VIRAL PROTEIN PREVALENCE")
    df, mob, *_ = load_data()
    viral_ids = set(df['replicon'].unique())
    n_total = mob['sample_id'].nunique()
    n_viral = len(viral_ids)
    print(f"Total plasmids:        {n_total}")
    print(f"With viral proteins:   {n_viral} ({n_viral/n_total*100:.1f}%)")
    print(f"Without:               {n_total - n_viral} "
          f"({(n_total - n_viral)/n_total*100:.1f}%)")

    summary = pd.DataFrame([
        {'metric': 'total_plasmids',       'n': n_total,           'pct': 100.0},
        {'metric': 'with_viral_proteins',  'n': n_viral,           'pct': n_viral / n_total * 100},
        {'metric': 'without_viral',        'n': n_total - n_viral, 'pct': (n_total - n_viral) / n_total * 100},
    ])
    summary.to_csv(OUT_DIR / "01_viral_prevalence_summary.csv", index=False)

    per_plasmid = pd.DataFrame({
        'replicon': sorted(mob['sample_id'].unique()),
    })
    per_plasmid['has_viral_protein'] = per_plasmid['replicon'].isin(viral_ids).astype(int)
    per_plasmid.to_csv(OUT_DIR / "01_viral_prevalence_per_plasmid.csv", index=False)


if __name__ == "__main__":
    main()
