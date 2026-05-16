#!/usr/bin/env python3
"""Carrier rate by NCBI assembly level."""
from common import load_data, header


def main():
    _, _, reps_meta = load_data()
    header("CARRIER RATE BY ASSEMBLY LEVEL")
    print(f"Reps joined to NCBI metadata: {len(reps_meta)}")

    for level in ['Complete Genome', 'Chromosome', 'Scaffold', 'Contig']:
        sub = reps_meta[reps_meta['ncbi_assembly_level'] == level]
        n = len(sub)
        if n == 0:
            print(f"  {level:<16}: 0/0")
            continue
        nc = int(sub['is_carrier'].sum())
        print(f"  {level:<16}: {nc}/{n} ({nc/n*100:.2f}%)")


if __name__ == "__main__":
    main()
