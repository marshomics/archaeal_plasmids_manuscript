"""Loaders and helpers for the viral-content pipeline."""
import math
from pathlib import Path
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
OUT_DIR  = SCRIPT_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

FINAL_CSV   = DATA_DIR / "final_classified_data.csv"
MOB_TSV     = DATA_DIR / "mobsuite_combined_conj_taxonomy.txt"
CONJ_TXT    = DATA_DIR / "putative_conjugative_plasmids.txt"
CLUSTER_CSV = DATA_DIR / "final_cluster_assignments_5st.csv"
FISHER_CSV  = DATA_DIR / "fisher_enrichment_final.csv"


def header(title):
    bar = "=" * 70
    print(f"\n{bar}\n  {title}\n{bar}\n")


def fisher_exact_with_ci(table, alpha=0.05):
    """Woolf-logit 95% CI on the OR + Fisher exact p."""
    OR, p = stats.fisher_exact(table)
    a, b, c, d = table[0][0], table[0][1], table[1][0], table[1][1]
    a_, b_, c_, d_ = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    log_or = math.log(a_ * d_ / (b_ * c_))
    se = math.sqrt(1/a_ + 1/b_ + 1/c_ + 1/d_)
    z = stats.norm.ppf(1 - alpha / 2)
    return OR, p, math.exp(log_or - z * se), math.exp(log_or + z * se)


def holm_bonferroni(p_values):
    """Holm-Bonferroni step-down on a list of (label, p_raw) tuples."""
    n = len(p_values)
    sorted_pvals = sorted(p_values, key=lambda x: x[1])
    out = []
    for rank, (label, p_raw) in enumerate(sorted_pvals):
        out.append((label, p_raw, min(p_raw * (n - rank), 1.0)))
    # enforce monotonicity
    for i in range(1, len(out)):
        if out[i][2] < out[i-1][2]:
            out[i] = (out[i][0], out[i][1], out[i-1][2])
    return [(lab, raw, adj, adj < 0.05) for lab, raw, adj in out]


def load_data():
    df  = pd.read_csv(FINAL_CSV)
    mob = pd.read_csv(MOB_TSV, sep='\t')
    clusters = pd.read_csv(CLUSTER_CSV)
    with open(CONJ_TXT) as f:
        conj_list = {ln.strip() for ln in f if ln.strip()}

    all_ids = mob['sample_id'].unique()
    complexity = df.groupby('replicon')['new_category'].nunique()
    full_complexity = pd.Series(0, index=all_ids)
    full_complexity.update(complexity)
    full_complexity = full_complexity.astype(int)
    return df, mob, clusters, conj_list, complexity, full_complexity
