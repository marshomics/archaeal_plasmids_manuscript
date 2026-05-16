#!/usr/bin/env python3
"""Detection power per non-carrier phylum, given complete-genome base rate.

Power = 1 - (1 - p)^k, with p = complete-genome carrier rate and k = number
of complete-genome species in that phylum.
"""
from common import load_data, header


def main():
    reps, meta, _ = load_data()
    header("DETECTION POWER FOR NON-CARRIER PHYLA")

    carrier_species = set(reps[reps['plasmid_prevalence'] == 1]['gtdb_species'])
    carrier_phyla = set(reps[reps['plasmid_prevalence'] == 1]['gtdb_phylum'])
    all_phyla = sorted(reps['gtdb_phylum'].unique())
    non_carrier = [p for p in all_phyla if p not in carrier_phyla]

    cg = meta[meta['ncbi_assembly_level'] == 'Complete Genome']
    cg_species = cg.groupby('gtdb_species').size().index
    cg_carrier = len(set(cg_species) & carrier_species)
    base_rate = cg_carrier / len(cg_species)
    print(f"Non-carrier phyla:                  {len(non_carrier)}")
    print(f"Complete-genome species total:      {len(cg_species)}")
    print(f"Complete-genome carrier base rate:  "
          f"{cg_carrier}/{len(cg_species)} ({base_rate*100:.1f}%)")

    print("\nPer non-carrier phylum:")
    no_complete = 0
    print(f"{'Phylum':<30} {'Complete spp.':>14} {'Power':>8}")
    for phy in non_carrier:
        sub = meta[(meta['gtdb_phylum'] == phy) &
                   (meta['ncbi_assembly_level'] == 'Complete Genome')]
        n_species = sub['gtdb_species'].nunique()
        if n_species == 0:
            no_complete += 1
        power = (1 - (1 - base_rate) ** n_species) * 100
        print(f"{phy:<30} {n_species:>14} {power:>7.1f}%")
    print(f"\nPhyla with 0 complete genomes: {no_complete}/{len(non_carrier)}")


if __name__ == "__main__":
    main()
