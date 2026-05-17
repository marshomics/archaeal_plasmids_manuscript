#!/usr/bin/env python3
"""Is VirB4-T4CP status enriched among CRISPR-targeted plasmids?

Reports both the unconditional two-sided Fisher exact (kept for reference)
and a size-conditioned logistic regression of targeting status on plasmid
size and VirB4-T4CP status. Plasmid size is a known confounder for both
targeting probability (more sequence = more BLAST opportunity) and
conjugative status (conjugative plasmids tend to be larger), so the
size-adjusted OR is the primary inference.
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import fisher_exact

from common import BLAST_TSV, CONJ_TXT, MOB_FILE, OUT_DIR, header


def main():
    header("VirB4-T4CP × CRISPR TARGETING (size-conditioned)")
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

    OR_f, p_f = fisher_exact([[n_targeted_conj, n_untargeted_conj],
                              [n_targeted_nonconj, n_untargeted_nonconj]],
                             alternative='two-sided')

    print("Unconditional Fisher exact (two-sided):")
    print(f"  VirB4-T4CP+: {n_total_conj} total, "
          f"{n_targeted_conj} targeted ({n_targeted_conj/n_total_conj*100:.1f}%)")
    print(f"  VirB4-T4CP-: {n_total_nonconj} total, "
          f"{n_targeted_nonconj} targeted "
          f"({n_targeted_nonconj/n_total_nonconj*100:.1f}%)")
    print(f"  OR = {OR_f:.2f}, p = {p_f:.2f}")

    # Size-conditioned logistic regression
    df = mob[['sample_id', 'size']].dropna().copy()
    df['size'] = pd.to_numeric(df['size'], errors='coerce')
    df = df[df['size'] > 0].copy()
    df['is_conj']     = df['sample_id'].isin(conj).astype(int)
    df['is_targeted'] = df['sample_id'].isin(targets).astype(int)
    df['log10_size']  = np.log10(df['size'])

    print(f"\nLogistic regression on n = {len(df)} plasmids "
          f"(targeted = {int(df['is_targeted'].sum())}, "
          f"conjugative = {int(df['is_conj'].sum())})")

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        X = sm.add_constant(df[['log10_size', 'is_conj']].values)
        y = df['is_targeted'].values
        try:
            logit = sm.Logit(y, X).fit(disp=False, maxiter=200)
            converged = logit.mle_retvals.get('converged', True)
        except Exception as e:
            print(f"  Logit failed: {e}")
            converged = False
            logit = None

    out = {
        'unconditional_fisher_OR': OR_f,
        'unconditional_fisher_p_two_sided': p_f,
        'n_target_conj':      n_targeted_conj,
        'n_untarget_conj':    n_untargeted_conj,
        'n_target_nonconj':   n_targeted_nonconj,
        'n_untarget_nonconj': n_untargeted_nonconj,
    }
    if logit is not None and converged:
        beta = logit.params
        se   = logit.bse
        OR_logit_size = np.exp(beta[1])
        OR_logit_conj = np.exp(beta[2])
        ci_lo_size, ci_hi_size = np.exp(beta[1] - 1.96 * se[1]), np.exp(beta[1] + 1.96 * se[1])
        ci_lo_conj, ci_hi_conj = np.exp(beta[2] - 1.96 * se[2]), np.exp(beta[2] + 1.96 * se[2])
        p_size = logit.pvalues[1]
        p_conj = logit.pvalues[2]
        # LRT for is_conj (drop term, refit)
        X_red = sm.add_constant(df[['log10_size']].values)
        logit_red = sm.Logit(y, X_red).fit(disp=False, maxiter=200)
        lrt = 2 * (logit.llf - logit_red.llf)
        from scipy.stats import chi2
        p_lrt = 1 - chi2.cdf(lrt, df=1)
        print(f"  log10(size):  OR (per 1-log10 unit) = {OR_logit_size:.2f} "
              f"[{ci_lo_size:.2f}, {ci_hi_size:.2f}], p = {p_size:.2e}")
        print(f"  VirB4-T4CP:   OR (adj for size)     = {OR_logit_conj:.2f} "
              f"[{ci_lo_conj:.2f}, {ci_hi_conj:.2f}], p (Wald) = {p_conj:.2e}, "
              f"p (LRT) = {p_lrt:.2e}")
        print(f"  Model: pseudo-R² = {logit.prsquared:.3f}, "
              f"converged = {converged}")
        out.update({
            'logit_size_OR': OR_logit_size,
            'logit_size_CI_lo': ci_lo_size, 'logit_size_CI_hi': ci_hi_size,
            'logit_size_p': p_size,
            'logit_conj_OR_adj': OR_logit_conj,
            'logit_conj_CI_lo': ci_lo_conj, 'logit_conj_CI_hi': ci_hi_conj,
            'logit_conj_p_wald': p_conj,
            'logit_conj_p_lrt': p_lrt,
            'logit_pseudo_R2': logit.prsquared,
        })

    pd.DataFrame([out]).to_csv(OUT_DIR / 'crispr_virb4_size_adjusted.csv',
                               index=False)


if __name__ == "__main__":
    main()
