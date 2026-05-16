"""Loader returning reps, full metadata and reps joined to assembly metadata."""
from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
REPS_FILE = DATA_DIR / "ar53_metadata_r214_reps_nospeciesincluded_matched_plasmids.txt"
META_FILE = DATA_DIR / "ar53_metadata_r214.tsv"

DEPTH_BINS = [(1, 1, '1'), (2, 3, '2-3'), (4, 10, '4-10'), (11, 9999, '≥11')]


def _tax(tax_string, prefix):
    for part in tax_string.split(';'):
        if part.startswith(prefix):
            return part
    return None


def load_data():
    reps = pd.read_csv(REPS_FILE, sep="\t")
    reps['is_carrier'] = (reps['plasmid_abundance'] > 0).astype(int)

    meta = pd.read_csv(META_FILE, sep="\t", low_memory=False)
    meta['gtdb_phylum']  = meta['gtdb_taxonomy'].apply(lambda x: _tax(x, 'p__'))
    meta['gtdb_species'] = meta['gtdb_taxonomy'].apply(lambda x: _tax(x, 's__'))

    depth = meta.groupby('gtdb_species').size().reset_index(name='n_genomes')
    reps = reps.merge(depth, on='gtdb_species', how='left')
    reps['n_genomes'] = reps['n_genomes'].fillna(1).astype(int)

    if 'gtdb_phylum' not in reps.columns:
        reps['gtdb_phylum'] = reps['gtdb_taxonomy'].apply(lambda x: _tax(x, 'p__'))

    # inner join → ~4.4k of 5.5k reps will carry NCBI assembly metadata
    reps_meta = reps.merge(
        meta[['accession', 'ncbi_assembly_level', 'ncbi_genome_category']],
        on='accession', how='inner',
    )
    reps_meta['is_carrier'] = (reps_meta['plasmid_prevalence'] == 1).astype(int)
    return reps, meta, reps_meta


def header(title):
    bar = "=" * 70
    print(f"\n{bar}\n  {title}\n{bar}\n")
