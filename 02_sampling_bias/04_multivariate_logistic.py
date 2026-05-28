#!/usr/bin/env python3
"""Logistic regression: phylum + depth + genome quality covariates.
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import roc_auc_score

from common import load_data, header


def _fit(y, X):
    return sm.Logit(y, sm.add_constant(X.astype(float))).fit(disp=0, maxiter=300)


def _diagnostics(model, y):
    mcfadden = 1 - model.llf / model.llnull
    auc = roc_auc_score(y, model.fittedvalues)
    return mcfadden, auc


def _report_model(name, model, y):
    mcf, auc = _diagnostics(model, y)
    print(f"\n{name}:")
    for var in model.params.index:
        if var == 'const':
            continue
        OR = np.exp(model.params[var])
        p = model.pvalues[var]
        print(f"  {var:<25}  OR = {OR:8.3f}   p = {p:.3e}")
    print(f"  McFadden R² = {mcf:.4f}   AUC-ROC = {auc:.4f}")
    print(f"  LL = {model.llf:.2f}   AIC = {model.aic:.1f}   BIC = {model.bic:.1f}")


def _lrt(m_big, m_small, df_diff, label):
    lr = 2 * (m_big.llf - m_small.llf)
    p = 1 - stats.chi2.cdf(lr, df=df_diff)
    print(f"  {label}:  LR χ² = {lr:.2f}, df = {df_diff}, p = {p:.3e}")
    return lr, p


def _prepare(reps, meta):
    """Merge reps with full genome-quality metadata."""
    covs_to_grab = [
        'accession', 'ncbi_assembly_level', 'ncbi_genome_category',
        'genome_size', 'gc_percentage', 'checkm_completeness',
        'checkm_contamination', 'coding_density', 'contig_count',
    ]
    df = reps.merge(meta[covs_to_grab], on='accession', how='inner')
    df['is_carrier']  = (df['plasmid_prevalence'] == 1).astype(int)
    df['is_halo']     = (df['gtdb_phylum'] == 'p__Halobacteriota').astype(int)
    df['log_depth']   = np.log1p(df['n_genomes'])
    df['log_genome_size'] = np.log(df['genome_size'])
    df['log_contigs'] = np.log1p(df['contig_count'])
    df['is_complete'] = (df['ncbi_assembly_level'] == 'Complete Genome').astype(int)
    df['is_isolate']  = (df['ncbi_genome_category'] == 'none').astype(int)
    return df


# Covariate sets
GENOME_COVS = [
    'log_depth', 'log_genome_size', 'gc_percentage',
    'checkm_completeness', 'checkm_contamination',
    'coding_density', 'log_contigs', 'is_complete', 'is_isolate',
]
SUBSET_COVS = [
    'log_depth', 'log_genome_size', 'gc_percentage',
    'checkm_completeness', 'checkm_contamination',
    'coding_density', 'log_contigs',
]


def main():
    reps, meta, _ = load_data()

    df = _prepare(reps, meta)
    df = df.dropna(subset=GENOME_COVS).copy()

    # ── Part 1: All species ────────────────────────────────────────
    header("PART 1: ALL SPECIES WITH METADATA")
    y = df['is_carrier']
    print(f"N = {len(df)} (carriers = {int(y.sum())})")

    m_depth   = _fit(y, df[['log_depth']])
    m_phylum  = _fit(y, df[['is_halo']])
    m_phy_dep = _fit(y, df[['is_halo', 'log_depth']])
    m_genome  = _fit(y, df[GENOME_COVS])
    m_full    = _fit(y, df[['is_halo'] + GENOME_COVS])

    _report_model("Depth only", m_depth, y)
    _report_model("Phylum only (is_halo)", m_phylum, y)
    _report_model("Phylum + Depth", m_phy_dep, y)
    _report_model("Genome covariates only (no phylum)", m_genome, y)
    _report_model("Phylum + ALL genome covariates", m_full, y)

    print("\nLikelihood-ratio tests (all species):")
    _lrt(m_phy_dep, m_depth, 1, "phylum | depth")
    _lrt(m_phy_dep, m_phylum, 1, "depth | phylum")
    _lrt(m_full, m_genome, 1, "phylum | genome covariates")
    _lrt(m_full, m_phy_dep, len(GENOME_COVS) - 1,
         "genome covariates | phylum + depth")

    # ── Part 2: Complete genomes only ──────────────────────────────
    header("PART 2: COMPLETE GENOMES ONLY")
    cg = df[df['is_complete'] == 1].copy()
    y_cg = cg['is_carrier']
    print(f"N = {len(cg)} (carriers = {int(y_cg.sum())})")

    m_cg_phy  = _fit(y_cg, cg[['is_halo']])
    m_cg_covs = _fit(y_cg, cg[['is_halo'] + SUBSET_COVS])
    m_cg_nophy = _fit(y_cg, cg[SUBSET_COVS])

    _report_model("Phylum only (complete genomes)", m_cg_phy, y_cg)
    _report_model("Phylum + genome covariates (complete genomes)", m_cg_covs, y_cg)
    _report_model("Genome covariates only (complete genomes)", m_cg_nophy, y_cg)

    print("\nLikelihood-ratio tests (complete genomes):")
    _lrt(m_cg_covs, m_cg_nophy, 1, "phylum | genome covariates")

    # ── Part 3: Isolates only ─────────────────────────────────────
    header("PART 3: ISOLATES ONLY")
    iso = df[df['is_isolate'] == 1].copy()
    y_iso = iso['is_carrier']
    print(f"N = {len(iso)} (carriers = {int(y_iso.sum())})")

    m_iso_phy  = _fit(y_iso, iso[['is_halo']])
    m_iso_covs = _fit(y_iso, iso[['is_halo'] + SUBSET_COVS])
    m_iso_nophy = _fit(y_iso, iso[SUBSET_COVS])

    _report_model("Phylum only (isolates)", m_iso_phy, y_iso)
    _report_model("Phylum + genome covariates (isolates)", m_iso_covs, y_iso)
    _report_model("Genome covariates only (isolates)", m_iso_nophy, y_iso)

    print("\nLikelihood-ratio tests (isolates):")
    _lrt(m_iso_covs, m_iso_nophy, 1, "phylum | genome covariates")

    # ── Summary table ──────────────────────────────────────────────
    header("MODEL COMPARISON SUMMARY")
    rows = [
        ("All spp — Depth only", m_depth, y),
        ("All spp — Phylum only", m_phylum, y),
        ("All spp — Phylum + Depth", m_phy_dep, y),
        ("All spp — Genome covs (no phylum)", m_genome, y),
        ("All spp — Phylum + genome covs", m_full, y),
        ("Complete — Phylum only", m_cg_phy, y_cg),
        ("Complete — Phylum + genome covs", m_cg_covs, y_cg),
        ("Complete — Genome covs (no phylum)", m_cg_nophy, y_cg),
        ("Isolates — Phylum only", m_iso_phy, y_iso),
        ("Isolates — Phylum + genome covs", m_iso_covs, y_iso),
        ("Isolates — Genome covs (no phylum)", m_iso_nophy, y_iso),
    ]
    print(f"{'Model':<42} {'N':>5} {'McF.R²':>8} {'AUC':>7} {'AIC':>8}")
    print("-" * 75)
    for name, m, yy in rows:
        mcf, auc = _diagnostics(m, yy)
        print(f"{name:<42} {len(yy):>5} {mcf:>8.4f} {auc:>7.4f} {m.aic:>8.1f}")


if __name__ == "__main__":
    main()
