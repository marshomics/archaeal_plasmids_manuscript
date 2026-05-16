#!/usr/bin/env python3
"""Per-family carrier counts, prevalence and mean abundance."""
from common import load_data, header


def main():
    reps, _ = load_data()
    header("FAMILY-LEVEL")

    fam = reps.groupby(['gtdb_phylum', 'gtdb_family']).agg(
        n_species=('is_carrier', 'size'),
        n_carriers=('is_carrier', 'sum'),
        total_plasmids=('plasmid_abundance', 'sum'),
    ).reset_index()
    fam['prevalence'] = fam['n_carriers'] / fam['n_species'] * 100

    carrier_mean = (
        reps[reps['is_carrier'] == 1]
        .groupby('gtdb_family')['plasmid_abundance']
        .mean()
    )
    fam = fam.merge(carrier_mean.rename('mean_abundance'),
                    on='gtdb_family', how='left')

    carrier_fam = fam[fam['n_carriers'] > 0].sort_values('prevalence', ascending=False)
    print(f"Families with carriers: {len(carrier_fam)}")
    for _, r in carrier_fam.iterrows():
        print(f"  {r['gtdb_family']}: "
              f"{int(r['n_carriers'])}/{int(r['n_species'])} "
              f"({r['prevalence']:.1f}%), "
              f"{int(r['total_plasmids'])} plasmids, "
              f"mean = {r['mean_abundance']:.2f}")


if __name__ == "__main__":
    main()
