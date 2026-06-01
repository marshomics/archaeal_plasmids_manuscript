#!/usr/bin/env python3
"""Filter a multi-FASTA to retain only Archaeal plasmid sequences per PLSDB summary."""

import csv
import sys

FASTA_IN = "/ebio/abt3_scratch/jmarsh/archaea_plasmids/db/plsdb/sequences.fasta"
SUMMARY = "/ebio/abt3_scratch/jmarsh/archaea_plasmids/db/plsdb/plsdb_summary.csv"
FASTA_OUT = "/ebio/abt3_scratch/jmarsh/archaea_plasmids/db/plsdb/sequences_archaea.fasta"

ACC_COL = "NUCCORE_ACC"
KINGDOM_COL = "TAXONOMY.TAXONOMY_superkingdom"
ARCHAEA = "Archaea (2157)"


def load_archaeal_accessions(path):
    """Return the set of NUCCORE_ACC values whose superkingdom is Archaea."""
    keep = set()
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for col in (ACC_COL, KINGDOM_COL):
            if col not in reader.fieldnames:
                sys.exit(f"Column '{col}' not found in {path}. "
                         f"Columns present: {reader.fieldnames}")
        for row in reader:
            if row[KINGDOM_COL].strip() == ARCHAEA:
                acc = row[ACC_COL].strip()
                if acc:
                    keep.add(acc)
    return keep


def filter_fasta(fasta_in, fasta_out, keep_accessions):
    """Stream the FASTA, writing records whose header (pre-space) is in keep_accessions."""
    written = 0
    with open(fasta_in, encoding="utf-8") as fin, \
         open(fasta_out, "w", encoding="utf-8") as fout:
        keep_current = False
        for line in fin:
            if line.startswith(">"):
                acc = line[1:].split(None, 1)[0]
                keep_current = acc in keep_accessions
                if keep_current:
                    fout.write(line)
                    written += 1
            elif keep_current:
                fout.write(line)
    return written


def main():
    keep = load_archaeal_accessions(SUMMARY)
    print(f"{len(keep)} archaeal accessions in summary table", file=sys.stderr)

    n = filter_fasta(FASTA_IN, FASTA_OUT, keep)
    print(n)


if __name__ == "__main__":
    main()