#!/usr/bin/env python3
"""Read the pre-computed family × category Fisher table and report sig rows."""
import pandas as pd
from common import FISHER_CSV, header


def main():
    header("FAMILY × CATEGORY FISHER ENRICHMENT")
    fisher_df = pd.read_csv(FISHER_CSV)

    all_fam   = fisher_df[fisher_df['analysis'] == 'all_families']
    halo_only = fisher_df[fisher_df['analysis'] == 'halobacteriota_only']
    print(f"All-families tests:        {len(all_fam)}")
    print(f"Halobacteriota-only tests: {len(halo_only)}")
    print(f"Total tests:               {len(fisher_df)}")

    sig = fisher_df[fisher_df['significant'] == True]
    print(f"\nSignificant (FDR < 0.05): {len(sig)}")
    for _, row in sig.iterrows():
        direction = "enriched" if row['OR'] > 1 else "depleted"
        print(f"  {row['family']} × {row['category']} ({row['analysis']}): "
              f"OR = {row['OR']:.2f}, FDR = {row['p_adj']:.3f} ({direction})")


if __name__ == "__main__":
    main()
