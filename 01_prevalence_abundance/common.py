"""GTDB r214 archaeal reps + full metadata loader."""
from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
REPS_FILE = DATA_DIR / "ar53_metadata_r214_reps_nospeciesincluded_matched_plasmids.txt"
META_FILE = DATA_DIR / "ar53_metadata_r214.tsv"
NONHALO_MODEL_FILE = DATA_DIR / "nonhalo_model_organisms.txt"
HALO_MODEL_FILE    = DATA_DIR / "halo_model_organisms.txt"

PHYLUM_COLORS = {
    'p__Halobacteriota':      '#4DAF4A',
    'p__Methanobacteriota':   '#E69F00',
    'p__Methanobacteriota_B': '#56B4E9',
    'p__Thermoproteota':      '#CC79A7',
    'p__Thermoplasmatota':    '#009E73',
}


def _parse_tax_field(tax_string, prefix):
    for part in tax_string.split(';'):
        if part.startswith(prefix):
            return part
    return None


def load_data():
    reps = pd.read_csv(REPS_FILE, sep="\t")
    reps['is_carrier'] = (reps['plasmid_abundance'] > 0).astype(int)

    meta = pd.read_csv(META_FILE, sep="\t", low_memory=False)
    meta['gtdb_phylum']  = meta['gtdb_taxonomy'].apply(lambda x: _parse_tax_field(x, 'p__'))
    meta['gtdb_species'] = meta['gtdb_taxonomy'].apply(lambda x: _parse_tax_field(x, 's__'))

    # per-species genome count = sampling depth
    genome_counts = meta.groupby('gtdb_species').size().reset_index(name='n_genomes')
    reps = reps.merge(genome_counts, on='gtdb_species', how='left')
    reps['n_genomes'] = reps['n_genomes'].fillna(1).astype(int)
    return reps, meta


def _load_species_list(path):
    if not path.exists():
        raise FileNotFoundError(f"missing species list: {path}")
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith('#')]


def load_model_species():
    return _load_species_list(NONHALO_MODEL_FILE), _load_species_list(HALO_MODEL_FILE)


def header(title):
    bar = "=" * 70
    print(f"\n{bar}\n  {title}\n{bar}\n")
