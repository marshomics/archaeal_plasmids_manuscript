"""Loaders for the defence + CRISPR pipeline.

Inputs (data/):
  - defense_finder_output_combined_taxonomy_defense_only_reshaped_type.txt
  - defense_finder_output_combined_taxonomy_defense_only_reshaped_subtype.txt
  - mobsuite_combined_conj_taxonomy.txt
  - putative_conjugative_plasmids.txt
  - crispr_spacers.fasta
  - crispr_blast_hits.tsv
  - sequence_space_summary.tsv
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
OUT_DIR  = SCRIPT_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

TYPE_FILE = DATA_DIR / "defense_finder_output_combined_taxonomy_defense_only_reshaped_type.txt"
SUB_FILE  = DATA_DIR / "defense_finder_output_combined_taxonomy_defense_only_reshaped_subtype.txt"
MOB_FILE  = DATA_DIR / "mobsuite_combined_conj_taxonomy.txt"
CONJ_TXT  = DATA_DIR / "putative_conjugative_plasmids.txt"
SPACER_FASTA = DATA_DIR / "crispr_spacers.fasta"
BLAST_TSV    = DATA_DIR / "crispr_blast_hits.tsv"
SEQ_SPACE    = DATA_DIR / "sequence_space_summary.tsv"

SEED = 42
N_PERM = 10_000


def header(title):
    bar = "=" * 70
    print(f"\n{bar}\n  {title}\n{bar}\n")


def load_defense_tables():
    """Return (type_df, binary_type, sub_df, type_cols, sub_cols).

    Adds n_instances (total defence count), n_types (distinct types) and
    inverse-species-frequency weights normalised so sum(w) = N.
    """
    TAX_COLS = ['replicon', 'gtdb_phylum', 'gtdb_class', 'gtdb_order',
                'gtdb_family', 'gtdb_genus', 'gtdb_species']
    type_df = pd.read_csv(TYPE_FILE, sep='\t')
    sub_df  = pd.read_csv(SUB_FILE,  sep='\t')
    type_cols = [c for c in type_df.columns if c not in TAX_COLS]
    sub_cols  = [c for c in sub_df.columns  if c not in TAX_COLS]
    for c in type_cols:
        type_df[c] = pd.to_numeric(type_df[c], errors='coerce').fillna(0).astype(int)
    for c in sub_cols:
        sub_df[c]  = pd.to_numeric(sub_df[c],  errors='coerce').fillna(0).astype(int)

    binary_type = type_df.copy()
    for c in type_cols:
        binary_type[c] = (binary_type[c] > 0).astype(int)

    for df in (type_df, binary_type, sub_df):
        df['phylum']  = df['gtdb_phylum'].str.replace('p__', '', regex=False)
    for df in (type_df, binary_type):
        df['family']  = df['gtdb_family'].str.replace('f__', '', regex=False)
        df['species'] = df['gtdb_species'].str.replace('s__', '', regex=False)

    type_df['n_instances'] = type_df[type_cols].sum(axis=1)
    type_df['n_types']     = binary_type[type_cols].sum(axis=1)
    binary_type['n_instances'] = type_df['n_instances']
    binary_type['n_types']     = type_df['n_types']

    N = len(type_df)
    species_counts = type_df['species'].value_counts()
    type_df['weight_raw'] = type_df['species'].map(lambda s: 1.0 / species_counts[s])
    type_df['weight']     = type_df['weight_raw'] / type_df['weight_raw'].sum() * N
    binary_type['weight'] = type_df['weight']
    return type_df, binary_type, sub_df, type_cols, sub_cols
