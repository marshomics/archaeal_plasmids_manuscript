"""Paths and COG vocabulary for the cross-domain pipeline."""
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
OUT_DIR  = SCRIPT_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

CLUSTER_TSV = DATA_DIR / "archaea_vs_bacteria_cluster_taxonomy_eggnog.tsv"
CLUSTER_SUMMARY = OUT_DIR / "cluster_summary.csv"
ARCHAEAL_EGGNOG = DATA_DIR / "proteins_combined_taxonomy_eggnog.txt"

COG_DESCRIPTIONS = {
    'J': 'Translation, ribosomal structure and biogenesis',
    'A': 'RNA processing and modification',
    'K': 'Transcription',
    'L': 'Replication, recombination and repair',
    'B': 'Chromatin structure and dynamics',
    'D': 'Cell cycle control, cell division',
    'Y': 'Nuclear structure',
    'V': 'Defense mechanisms',
    'T': 'Signal transduction mechanisms',
    'M': 'Cell wall/membrane/envelope biogenesis',
    'N': 'Cell motility',
    'Z': 'Cytoskeleton',
    'W': 'Extracellular structures',
    'U': 'Intracellular trafficking, secretion, vesicular transport',
    'O': 'Post-translational modification, chaperones',
    'C': 'Energy production and conversion',
    'G': 'Carbohydrate transport and metabolism',
    'E': 'Amino acid transport and metabolism',
    'F': 'Nucleotide transport and metabolism',
    'H': 'Coenzyme transport and metabolism',
    'I': 'Lipid transport and metabolism',
    'P': 'Inorganic ion transport and metabolism',
    'Q': 'Secondary metabolites',
    'R': 'General function prediction only',
    'S': 'Function unknown',
}

N_PERM = 1000     # CLR permutation iterations
N_BOOT = 100      # species-level bootstrap iterations
FAM_MIN = 5       # min plasmids per family for family-level tests
PSEUDO = 0.5      # CLR pseudocount


def header(title):
    bar = "=" * 70
    print(f"\n{bar}\n  {title}\n{bar}\n")
