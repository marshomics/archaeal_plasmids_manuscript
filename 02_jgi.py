#!/usr/bin/env python3
"""Filter the IMG/PR plasmid multi-FASTA down to putatively complete,
isolate-taxonomy archaeal plasmids and report phage-plasmid counts.

stdlib only. Streams the FASTA so memory stays flat on large inputs.
"""

import csv
import sys

FASTA_IN = "/ebio/abt3_scratch/jmarsh/archaea_plasmids/db/jgi/Cus_PR/IMG_VR_2023-08-08_1/IMGPR_nucl.fna"
TABLE_IN = "/ebio/abt3_scratch/jmarsh/archaea_plasmids/db/jgi/Cus_PR/IMG_VR_2023-08-08_1/IMGPR_plasmid_data.tsv"
FASTA_OUT = "/ebio/abt3_scratch/jmarsh/archaea_plasmids/db/jgi/Cus_PR/IMG_VR_2023-08-08_1/IMGPR_nucl.archaea.fna"

# Required table columns.
COL_ID = "plasmid_id"
COL_TAX = "host_taxonomy"
COL_METHOD = "host_prediction_method"
COL_COMPLETE = "putatively_complete"
COL_PHAGE = "putative_phage_plasmid"


def header_to_id(header_line):
    """Strip '>', take the field before the first '|'."""
    return header_line[1:].strip().split("|", 1)[0]


def load_passing(table_path):
    """Return dict: plasmid_id -> putative_phage_plasmid value, for rows
    that pass all three filters."""
    passing = {}
    with open(table_path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for col in (COL_ID, COL_TAX, COL_METHOD, COL_COMPLETE, COL_PHAGE):
            if col not in reader.fieldnames:
                sys.exit(f"ERROR: column '{col}' not found in {table_path}\n"
                         f"Found: {reader.fieldnames}")
        for row in reader:
            tax = (row[COL_TAX] or "").strip()
            method = (row[COL_METHOD] or "").strip()
            complete = (row[COL_COMPLETE] or "").strip()
            # "at least d__Archaea" -> host domain is Archaea
            if not tax.startswith("d__Archaea"):
                continue
            if method != "Isolate taxonomy":
                continue
            if complete != "Yes":
                continue
            passing[row[COL_ID].strip()] = (row[COL_PHAGE] or "").strip()
    return passing


def main():
    passing = load_passing(TABLE_IN)

    n_written = 0
    n_phage = 0

    with open(FASTA_IN, encoding="utf-8") as fin, \
         open(FASTA_OUT, "w", encoding="utf-8") as fout:
        keep = False
        current_phage = ""
        for line in fin:
            if line.startswith(">"):
                seq_id = header_to_id(line)
                keep = seq_id in passing
                if keep:
                    n_written += 1
                    current_phage = passing[seq_id]
                    if current_phage == "Yes":
                        n_phage += 1
                    fout.write(line)
            elif keep:
                fout.write(line)

    print(f"Sequences in filtered file: {n_written}")
    print(f"  of which putative_phage_plasmid == 'Yes': {n_phage}")


if __name__ == "__main__":
    main()