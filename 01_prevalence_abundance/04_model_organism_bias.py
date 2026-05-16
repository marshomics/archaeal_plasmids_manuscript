#!/usr/bin/env python3
"""Plasmid count contributed by reference-strain species (from data/)."""
from common import load_data, load_model_species, header


def main():
    reps, _ = load_data()
    nonhalo_models, halo_models = load_model_species()
    header("MODEL-ORGANISM CONTRIBUTION")
    print(f"Non-Halo model species: {len(nonhalo_models)}")
    print(f"Halo model species:     {len(halo_models)}\n")

    carriers = reps[reps['is_carrier'] == 1]
    halo = carriers[carriers['gtdb_phylum'] == 'p__Halobacteriota']
    nonhalo = carriers[carriers['gtdb_phylum'] != 'p__Halobacteriota']

    nh_mod = nonhalo[nonhalo['gtdb_species'].isin(nonhalo_models)]
    h_mod = halo[halo['gtdb_species'].isin(halo_models)]

    nh_share = nh_mod['plasmid_abundance'].sum() / nonhalo['plasmid_abundance'].sum() * 100
    h_share  = h_mod['plasmid_abundance'].sum()  / halo['plasmid_abundance'].sum()  * 100

    print(f"Non-Halo carriers:   {len(nonhalo)} species, "
          f"{int(nonhalo['plasmid_abundance'].sum())} plasmids")
    print(f"Halo carriers:       {len(halo)} species, "
          f"{int(halo['plasmid_abundance'].sum())} plasmids")
    print()
    print(f"Non-Halo models:     "
          f"{int(nh_mod['plasmid_abundance'].sum())}/"
          f"{int(nonhalo['plasmid_abundance'].sum())} ({nh_share:.1f}%)")
    print(f"  {', '.join(nh_mod['gtdb_species'].values)}")
    print()
    print(f"Halo models:         "
          f"{int(h_mod['plasmid_abundance'].sum())}/"
          f"{int(halo['plasmid_abundance'].sum())} ({h_share:.1f}%)")
    print(f"  {', '.join(h_mod['gtdb_species'].values)}")


if __name__ == "__main__":
    main()
