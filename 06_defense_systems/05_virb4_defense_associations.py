#!/usr/bin/env python3
"""NB GLM: defence burden / type richness ~ log10(size) + VirB4-T4CP, Halo only.

Uses sm.NegativeBinomial (NB2 MLE) to estimate dispersion, then fits a
species-weighted GLM-NB with freq_weights to control for unequal species
representation. Defence burden and type richness are computed from
subtype-level columns (matching original analysis).

Plus a per-defence-subtype Fisher test (BH-corrected) against VirB4-T4CP
status within Halobacteriota.
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

from common import SUB_FILE, MOB_FILE, CONJ_TXT, OUT_DIR, header


def _fit_weighted_nb(y, X, weights):
    """Fit species-weighted NB GLM.

    Step 1: estimate dispersion (alpha) from unweighted NB2-MLE.
    Step 2: fit GLM-NB with that fixed alpha and freq_weights.
    """
    # Estimate alpha from unweighted NB2
    nb_unw = sm.NegativeBinomial(y, X).fit(disp=False, maxiter=500)
    alpha = np.exp(nb_unw.params[-1])  # NB2 stores log(alpha)

    # Weighted GLM with estimated alpha
    glm = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=alpha),
                 freq_weights=weights).fit()
    return glm, alpha


def main():
    header("VirB4-T4CP × DEFENCE (Halo only, species-weighted)")

    # Load subtype file (burden and richness from subtypes, matching original)
    TAX_COLS = ['replicon', 'gtdb_phylum', 'gtdb_class', 'gtdb_order',
                'gtdb_family', 'gtdb_genus', 'gtdb_species']
    sub_df = pd.read_csv(SUB_FILE, sep='\t')
    defense_cols = [c for c in sub_df.columns if c not in TAX_COLS]
    for c in defense_cols:
        sub_df[c] = pd.to_numeric(sub_df[c], errors='coerce').fillna(0).astype(int)

    # Get sizes from mobsuite
    mob = pd.read_csv(MOB_FILE, sep='\t')
    sizes = mob[['sample_id', 'size']].rename(
        columns={'sample_id': 'replicon', 'size': 'size_bp'})

    # Get conjugative status
    with open(CONJ_TXT) as f:
        conj = {ln.strip() for ln in f if ln.strip()}

    # Merge and compute response variables
    df = sub_df.merge(sizes, on='replicon', how='inner')
    df['is_conjugative'] = df['replicon'].isin(conj).astype(int)
    df['defense_burden'] = df[defense_cols].sum(axis=1)
    df['type_richness'] = (df[defense_cols] > 0).sum(axis=1)
    df['log10_size'] = np.log10(df['size_bp'])
    df['species'] = df['gtdb_species'].str.replace('s__', '', regex=False)

    # Filter to Halobacteriota
    df = df[df['gtdb_phylum'] == 'p__Halobacteriota'].copy()
    df = df[df['size_bp'] > 0]

    # Species-level inverse-frequency weights, normalised so sum(w) = N
    species_counts = df['species'].value_counts()
    df['n_species_reps'] = df['species'].map(species_counts)
    df['w'] = 1.0 / df['n_species_reps']
    df['w'] = df['w'] / df['w'].sum() * len(df)

    print(f"Halo plasmids: {len(df)}  (VirB4-T4CP+ = {int(df['is_conjugative'].sum())})")
    print(f"Unique species: {df['species'].nunique()}")

    # Design matrix: const + log10_size + is_conjugative
    X = sm.add_constant(df[['log10_size', 'is_conjugative']].values)
    w = df['w'].values

    # Defence burden
    y_burden = df['defense_burden'].values
    print("\nNB GLM (species-weighted): burden ~ log10(size) + VirB4-T4CP")
    glm_b, alpha_b = _fit_weighted_nb(y_burden, X, w)
    irr_b = np.exp(glm_b.params[2])
    ci_lo_b = np.exp(glm_b.params[2] - 1.96 * glm_b.bse[2])
    ci_hi_b = np.exp(glm_b.params[2] + 1.96 * glm_b.bse[2])
    p_b = glm_b.pvalues[2]
    print(f"  alpha = {alpha_b:.4f}")
    print(f"  IRR = {irr_b:.2f} (95% CI [{ci_lo_b:.2f}, {ci_hi_b:.2f}]), p = {p_b:.2e}")

    # Type richness
    y_richness = df['type_richness'].values
    print("\nNB GLM (species-weighted): richness ~ log10(size) + VirB4-T4CP")
    glm_r, alpha_r = _fit_weighted_nb(y_richness, X, w)
    irr_r = np.exp(glm_r.params[2])
    ci_lo_r = np.exp(glm_r.params[2] - 1.96 * glm_r.bse[2])
    ci_hi_r = np.exp(glm_r.params[2] + 1.96 * glm_r.bse[2])
    p_r = glm_r.pvalues[2]
    print(f"  alpha = {alpha_r:.4f}")
    print(f"  IRR = {irr_r:.2f} (95% CI [{ci_lo_r:.2f}, {ci_hi_r:.2f}]), p = {p_r:.2e}")

    # Save GLM summary
    pd.DataFrame([{
        'response': 'defense_burden',
        'conj_IRR': round(irr_b, 3), 'conj_CI_lo': round(ci_lo_b, 3),
        'conj_CI_hi': round(ci_hi_b, 3), 'conj_P': p_b,
        'nb_alpha': round(alpha_b, 4),
    }, {
        'response': 'type_richness',
        'conj_IRR': round(irr_r, 3), 'conj_CI_lo': round(ci_lo_r, 3),
        'conj_CI_hi': round(ci_hi_r, 3), 'conj_P': p_r,
        'nb_alpha': round(alpha_r, 4),
    }]).to_csv(OUT_DIR / 'virb4_defense_glm.csv', index=False)

    # Per-subtype Fisher (BH-corrected) within Halo
    rows = []
    for col in defense_cols:
        present = (df[col] > 0).astype(int)
        ct = pd.crosstab(present, df['is_conjugative'])
        a = ct.loc[1, 1] if (1 in ct.index and 1 in ct.columns) else 0
        b = ct.loc[1, 0] if (1 in ct.index and 0 in ct.columns) else 0
        c = ct.loc[0, 1] if (0 in ct.index and 1 in ct.columns) else 0
        d = ct.loc[0, 0] if (0 in ct.index and 0 in ct.columns) else 0
        if a + c < 1:
            continue
        OR_s, p_s = fisher_exact([[a, b], [c, d]])
        rows.append({'subtype': col, 'a': a, 'b': b, 'c': c, 'd': d,
                     'OR': OR_s, 'p_raw': p_s})
    if rows:
        out = pd.DataFrame(rows)
        out['p_adj_bh'] = multipletests(out['p_raw'], method='fdr_bh')[1]
        out['significant'] = out['p_adj_bh'] < 0.05
        out = out.sort_values('p_adj_bh')
        out.to_csv(OUT_DIR / 'subtype_virb4_fisher.csv', index=False)
        print("\nPer-subtype × VirB4-T4CP Fisher (BH-corrected):")
        sig = out[out['significant']]
        if len(sig) > 0:
            print(sig[['subtype', 'OR', 'p_adj_bh']].to_string(index=False))
        else:
            print("  No significant subtypes after BH correction.")
        # Report pAgo subtypes specifically
        pago = out[out['subtype'].str.contains('pAgo', case=False)]
        if len(pago) > 0:
            print("\npAgo subtypes:")
            print(pago[['subtype', 'OR', 'p_raw', 'p_adj_bh']].to_string(index=False))


if __name__ == "__main__":
    main()
