#!/usr/bin/env python3
"""Viral protein prevalence across the plasmid catalogue."""
from common import load_data, header


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


if __name__ == "__main__":
    main()
