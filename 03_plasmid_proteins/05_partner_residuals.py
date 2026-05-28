#!/usr/bin/env python3
"""Archaeal-family × bacterial-phylum contingency: χ² + Pearson residuals.
"""
from collections import Counter
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from common import CLUSTER_SUMMARY, OUT_DIR, header

MIN_MARGINAL = 5
N_PERM = 9999
RNG_SEED = 0


def _split_count(series):
    counter = Counter()
    for s in series.dropna():
        for tok in s.split('|'):
            if tok:
                counter[tok] += 1
    return counter


def main():
    header("PARTNER SPECIFICITY")
    cl = pd.read_csv(CLUSTER_SUMMARY)
    cd = cl[cl['cluster_type'] == 'cross-domain'].copy()
    cd['archaea_families'] = cd['archaea_families'].fillna('')
    cd['bacteria_phyla']   = cd['bacteria_phyla'].fillna('')

    fam_counts  = _split_count(cd['archaea_families'])
    phy_counts  = _split_count(cd['bacteria_phyla'])
    valid_fams  = sorted([f for f, n in fam_counts.items() if n >= MIN_MARGINAL],
                         key=lambda x: -fam_counts[x])
    valid_phyla = sorted([p for p, n in phy_counts.items() if n >= MIN_MARGINAL],
                         key=lambda x: -phy_counts[x])
    print(f"Archaeal families: {len(fam_counts)} total, "
          f"{len(valid_fams)} with marginal ≥ {MIN_MARGINAL}")
    print(f"Bacterial phyla:   {len(phy_counts)} total, "
          f"{len(valid_phyla)} with marginal ≥ {MIN_MARGINAL}")

    cooc = pd.DataFrame(0, index=valid_fams, columns=valid_phyla)
    for _, row in cd.iterrows():
        fams = [f for f in row['archaea_families'].split('|') if f in cooc.index]
        phyla = [p for p in row['bacteria_phyla'].split('|') if p in cooc.columns]
        for f in fams:
            for p in phyla:
                cooc.loc[f, p] += 1
    print(f"Contingency table: {cooc.shape[0]} × {cooc.shape[1]}")

    chi2, p_asym, dof, expected = chi2_contingency(cooc.values)
    residuals = (cooc.values - expected) / np.sqrt(expected)
    res_df = pd.DataFrame(residuals,
                          index=[f.replace('f__', '') for f in cooc.index],
                          columns=[c.replace('p__', '') for c in cooc.columns])
    pct_low = (expected < 5).mean() * 100
    print(f"\nχ² = {chi2:.1f}, df = {dof}, asymptotic p = {p_asym:.2e}")
    print(f"Cells with expected < 5: {pct_low:.1f}% — running permutation test")

    # permutation: shuffle phylum labels across all (family, phylum) pairs
    pair_list = [(f, p) for i, f in enumerate(cooc.index)
                 for j, p in enumerate(cooc.columns)
                 for _ in range(int(cooc.iat[i, j]))]
    if not pair_list:
        print("  no pairs to permute")
        return
    fam_idx = {f: i for i, f in enumerate(cooc.index)}
    phy_idx = {p: j for j, p in enumerate(cooc.columns)}
    f_arr = np.array([fam_idx[f] for f, _ in pair_list])
    p_arr = np.array([phy_idx[p] for _, p in pair_list])
    nrow, ncol = cooc.shape

    rng = np.random.default_rng(RNG_SEED)
    perm_chi2 = np.empty(N_PERM)
    for k in range(N_PERM):
        shuf = rng.permutation(p_arr)
        table = np.zeros((nrow, ncol), dtype=np.int64)
        np.add.at(table, (f_arr, shuf), 1)
        try:
            perm_chi2[k] = chi2_contingency(table)[0]
        except ValueError:
            perm_chi2[k] = 0.0
    p_perm = (np.sum(perm_chi2 >= chi2) + 1) / (N_PERM + 1)
    print(f"Permutation p = {p_perm:.2e} ({N_PERM} perms; "
          f"null mean χ² = {perm_chi2.mean():.1f})")

    flat = []
    for i, f in enumerate(res_df.index):
        for j, c in enumerate(res_df.columns):
            flat.append({'archaea_family': f, 'bacteria_phylum': c,
                         'observed': int(cooc.iat[i, j]),
                         'expected': round(expected[i, j], 2),
                         'residual': round(residuals[i, j], 3)})
    flat = pd.DataFrame(flat).sort_values('residual', ascending=False)
    flat.to_csv(OUT_DIR / "partner_specificity_residuals.csv", index=False)

    summary = pd.DataFrame([{
        'n_families': cooc.shape[0],
        'n_phyla':    cooc.shape[1],
        'chi2':       round(chi2, 2),
        'df':         dof,
        'p_asym':     p_asym,
        'p_perm':     p_perm,
        'n_perm':     N_PERM,
        'pct_expected_lt5': round(pct_low, 1),
        'min_marginal':    MIN_MARGINAL,
    }])
    summary.to_csv(OUT_DIR / "partner_specificity_chi2_summary.csv", index=False)

    print("\nTop 10 over-represented partnerships:")
    print(flat.head(10).to_string(index=False))
    print("\nTop 10 under-represented partnerships:")
    print(flat.tail(10).iloc[::-1].to_string(index=False))


if __name__ == "__main__":
    main()
