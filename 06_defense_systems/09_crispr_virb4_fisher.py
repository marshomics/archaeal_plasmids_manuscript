#!/usr/bin/env python3
"""Is VirB4-T4CP status enriched among CRISPR-targeted plasmids? Fisher exact."""
import pandas as pd
from scipy.stats import fisher_exact

from common import BLAST_TSV, CONJ_TXT, MOB_FILE, OUT_DIR, header


def main():
    header("VirB4-T4CP × CRISPR TARGETING")
    hits = pd.read_csv(BLAST_TSV, sep='\t')
    mob  = pd.read_csv(MOB_FILE, sep='\t')
    with open(CONJ_TXT) as f:
        conj = {ln.strip() for ln in f if ln.strip()}

    plasmid_hits = hits[hits['target_category'] == 'plasmid'].copy()
    targets = set(plasmid_hits['target_plasmid'].dropna())

    n_targeted_conj    = len(targets & conj)
    n_targeted_nonconj = len(targets - conj)

    all_plasmid_ids = set(mob['sample_id'])
    n_total_conj     = len(all_plasmid_ids & conj)
    n_total_nonconj  = len(all_plasmid_ids - conj)
    n_untargeted_conj    = n_total_conj    - n_targeted_conj
    n_untargeted_nonconj = n_total_nonconj - n_targeted_nonconj

    OR, p = fisher_exact([[n_targeted_conj, n_untargeted_conj],
                          [n_targeted_nonconj, n_untargeted_nonconj]])

    print(f"VirB4-T4CP+ plasmids: {n_total_conj} total, "
          f"{n_targeted_conj} targeted ({n_targeted_conj/n_total_conj*100:.1f}%)")
    print(f"VirB4-T4CP- plasmids: {n_total_nonconj} total, "
          f"{n_targeted_nonconj} targeted "
          f"({n_targeted_nonconj/n_total_nonconj*100:.1f}%)")
    print(f"\nFisher OR = {OR:.2f}, p = {p:.2f}")

    pd.DataFrame([{
        'n_target_conj':      n_targeted_conj,
        'n_untarget_conj':    n_untargeted_conj,
        'n_target_nonconj':   n_targeted_nonconj,
        'n_untarget_nonconj': n_untargeted_nonconj,
        'odds_ratio':         OR,
        'p_value':            p,
    }]).to_csv(OUT_DIR / 'crispr_virb4_fisher.csv', index=False)


if __name__ == "__main__":
    main()
