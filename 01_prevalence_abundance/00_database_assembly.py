#!/usr/bin/env python3
"""Catalogue sizes (source plasmid list and non-redundant set).
"""
from pathlib import Path
import pandas as pd

from common import DATA_DIR, header

CATALOGUE = DATA_DIR / "plasmid_catalogue.tsv"
SOURCE_LIST = DATA_DIR / "source_plasmid_list.txt"


def _count_unique(path, col_candidates):
    df = pd.read_csv(path, sep='\t')
    for c in col_candidates:
        if c in df.columns:
            return int(df[c].nunique())
    raise RuntimeError(f"{path.name}: no plasmid-ID column")


def main():
    header("DATABASE ASSEMBLY")

    if SOURCE_LIST.exists():
        with open(SOURCE_LIST) as f:
            n_source = sum(1 for ln in f if ln.strip() and not ln.startswith('#'))
        print(f"NCBI source plasmids: {n_source}")
    else:
        print(f"({SOURCE_LIST.name} not provided)")

    if CATALOGUE.exists():
        n_dedup = _count_unique(CATALOGUE,
                                ('replicon_name', 'sample_id', 'plasmid', 'accession'))
        print(f"Non-redundant set:    {n_dedup}")
    else:
        print(f"({CATALOGUE.name} not provided)")


if __name__ == "__main__":
    main()
