#!/usr/bin/env python3
"""Per-carrier plasmid abundance, Halo vs non-Halo, Fisher on single-plasmid carriers."""
from scipy import stats

from common import load_data, header


def main():
    reps, _ = load_data()
    header("ABUNDANCE DISTRIBUTION")
    carriers = reps[reps['is_carrier'] == 1]

    total_plas = int(carriers['plasmid_abundance'].sum())
    print(f"Total plasmids:      {total_plas}")
    print(f"Range per carrier:   {carriers['plasmid_abundance'].min()}-"
          f"{carriers['plasmid_abundance'].max()}")
    print(f"Median:              {carriers['plasmid_abundance'].median():.0f}")
    print(f"Mean:                {carriers['plasmid_abundance'].mean():.2f}")

    multi = carriers[carriers['plasmid_abundance'] > 1]
    print(f"\nMulti-plasmid carriers: {len(multi)}/{len(carriers)} "
          f"({len(multi)/len(carriers)*100:.1f}%)")

    halo = carriers[carriers['gtdb_phylum'] == 'p__Halobacteriota']
    nonhalo = carriers[carriers['gtdb_phylum'] != 'p__Halobacteriota']

    print(f"\nHalobacteriota (n={len(halo)}):")
    print(f"  median {halo['plasmid_abundance'].median():.0f}  "
          f"mean {halo['plasmid_abundance'].mean():.2f}  "
          f"range {halo['plasmid_abundance'].min()}-{halo['plasmid_abundance'].max()}")
    print(f"\nNon-Halobacteriota (n={len(nonhalo)}):")
    print(f"  median {nonhalo['plasmid_abundance'].median():.0f}  "
          f"mean {nonhalo['plasmid_abundance'].mean():.2f}  "
          f"range {nonhalo['plasmid_abundance'].min()}-{nonhalo['plasmid_abundance'].max()}")

    print("\nMean per non-Halo phylum:")
    for phy in ['p__Methanobacteriota', 'p__Methanobacteriota_B',
                'p__Thermoproteota', 'p__Thermoplasmatota']:
        sub = nonhalo[nonhalo['gtdb_phylum'] == phy]
        if len(sub):
            print(f"  {phy}: mean = {sub['plasmid_abundance'].mean():.2f} (n = {len(sub)})")

    # Fisher on single-plasmid carriers — drop the high-count tail (>4) so the
    # comparison isn't dominated by a handful of intensively sequenced species.
    halo_in = halo[halo['plasmid_abundance'] <= 4]
    nonhalo_in = nonhalo[nonhalo['plasmid_abundance'] <= 4]
    s_h = int((halo_in['plasmid_abundance'] == 1).sum())
    m_h = int((halo_in['plasmid_abundance'] > 1).sum())
    s_n = int((nonhalo_in['plasmid_abundance'] == 1).sum())
    m_n = int((nonhalo_in['plasmid_abundance'] > 1).sum())
    OR, p = stats.fisher_exact([[s_h, m_h], [s_n, m_n]])
    print("\nSingle vs >1 plasmid (outliers > 4 excluded):")
    print(f"  Halo single:     {s_h}/{s_h+m_h} ({s_h/(s_h+m_h)*100:.1f}%)")
    print(f"  Non-Halo single: {s_n}/{s_n+m_n} ({s_n/(s_n+m_n)*100:.1f}%)")
    print(f"  Fisher OR = {OR:.2f}, p = {p:.4f}")

    print("\nSpecies with > 4 plasmids:")
    top = carriers[carriers['plasmid_abundance'] > 4].sort_values(
        'plasmid_abundance', ascending=False)
    for _, r in top.iterrows():
        print(f"  {r['gtdb_species']}: {int(r['plasmid_abundance'])}")


if __name__ == "__main__":
    main()
