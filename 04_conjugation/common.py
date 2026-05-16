"""Paths and helpers for the VirB4-T4CP / conjugation pipeline.

Methodological inputs (loaded from data/, not hard-coded):
  - plasmid_catalogue.tsv     full non-redundant plasmid set; used as the
                              denominator in step 01
  - exclude_replicons.txt     one replicon ID per line; optional
  - mechanism_patterns.tsv    columns `mechanism`, `pattern`; optional regex
                              labels for step 05
  - filtered_gbk_matrix.csv  154×154 Clinker BLAST-derived pairwise distance
                              matrix; used in step 06 for dual-metric topology
                              validation
"""
from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
OUT_DIR  = SCRIPT_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

UNFILTERED_HITS = DATA_DIR / "virb4_vird4_combined_prokka_filtered.txt"
FILTERED_HITS   = DATA_DIR / "virb4_vird4_combined_prokka_filtered_potential_conjugative.txt"
CATALOGUE_FILE  = DATA_DIR / "plasmid_catalogue.tsv"
EXCLUDE_FILE    = DATA_DIR / "exclude_replicons.txt"
MECHANISM_FILE  = DATA_DIR / "mechanism_patterns.tsv"
CLUSTER_CSV     = DATA_DIR / "mmseqs_clusters" / "final_cluster_assignments_5st.csv"
CORE_ENRICH_TSV = DATA_DIR / "mmseqs_clusters" / "core_genes_enrichment.tsv"
CLINKER_MATRIX  = DATA_DIR / "filtered_gbk_matrix.csv"


def header(title):
    bar = "=" * 70
    print(f"\n{bar}\n  {title}\n{bar}\n")


def total_plasmids_in_catalogue():
    if not CATALOGUE_FILE.exists():
        raise FileNotFoundError(f"missing catalogue: {CATALOGUE_FILE}")
    df = pd.read_csv(CATALOGUE_FILE, sep='\t')
    for col in ('replicon_name', 'sample_id', 'plasmid', 'accession'):
        if col in df.columns:
            return int(df[col].nunique())
    raise RuntimeError(f"{CATALOGUE_FILE.name}: no plasmid-ID column")


def excluded_replicons():
    if not EXCLUDE_FILE.exists():
        return set()
    with open(EXCLUDE_FILE) as f:
        return {ln.strip() for ln in f if ln.strip() and not ln.startswith('#')}


def mechanism_patterns():
    if not MECHANISM_FILE.exists():
        return {}
    tab = pd.read_csv(MECHANISM_FILE, sep='\t')
    return dict(zip(tab['mechanism'], tab['pattern']))


def subtypes_present(clust_df):
    return [f"ST{int(s)}" for s in sorted(clust_df['hdbscan_cluster'].unique())]
