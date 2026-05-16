#!/usr/bin/env python3
"""Non-Halobacteriota viral content: complexity ceiling and per-family rollup."""
import numpy as np
from common import load_data, header


def main():
    header("NON-HALOBACTERIOTA VIRAL CONTENT")
    df, mob, *_ = load_data()

    non_halo_mob = mob[mob['gtdb_phylum'] != 'p__Halobacteriota']
    non_halo_ids = set(non_halo_mob['sample_id'])
    non_halo_viral = df[df['replicon'].isin(non_halo_ids)]
    non_halo_viral_ids = set(non_halo_viral['replicon'])

    print(f"Non-Halobacteriota plasmids: {len(non_halo_ids)}")
    print(f"  With viral proteins:       {len(non_halo_viral_ids)}")

    comp = non_halo_viral.groupby('replicon')['new_category'].nunique()
    print(f"  Complexity range:          {comp.min()}-{comp.max()}")
    print(f"  All at complexity 1:       {(comp == 1).all()}")

    print("\nPer family:")
    fam_sizes = []
    for fam in sorted(non_halo_mob['gtdb_family'].unique()):
        fam_ids = set(non_halo_mob.loc[non_halo_mob['gtdb_family'] == fam,
                                       'sample_id'])
        n_total = len(fam_ids)
        fam_sizes.append(n_total)
        fam_viral = non_halo_viral[non_halo_viral['replicon'].isin(fam_ids)]
        n_viral_fam = fam_viral['replicon'].nunique()
        cats = sorted(fam_viral['new_category'].unique())
        cat_str = ", ".join(cats) if cats else "none"
        print(f"  {fam:<28} n = {n_total:>3}, viral = {n_viral_fam:>3} "
              f"({n_viral_fam/n_total*100:.1f}%) — {cat_str}")

    print(f"\nPer-family sample size: "
          f"median = {int(np.median(fam_sizes))}, "
          f"range = {min(fam_sizes)}-{max(fam_sizes)} "
          f"across {len(fam_sizes)} families")


if __name__ == "__main__":
    main()
