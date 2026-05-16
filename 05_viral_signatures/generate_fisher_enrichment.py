#!/usr/bin/env python3
"""Generate the family × category Fisher enrichment/depletion table.

Reconstructs `fisher_enrichment_final.csv`, which was previously a frozen
pre-computed file with no upstream generation script.

Method
------
For each analysis stratum (all_families, halobacteriota_only):
  1. Restrict universe to viral-carrying plasmids (those present in
     final_classified_data.csv) that belong to families with >= 7
     viral-carrying members.  "Unclassified" families are excluded.
  2. For each (family, viral category) pair, build a 2 × 2 contingency
     table:  family-with-cat / family-without-cat / other-with-cat /
     other-without-cat, where "other" = all remaining viral-carrying
     plasmids in the universe.
  3. Run Fisher's exact test (two-sided) to obtain OR and p.
  4. Correct p-values within each stratum using Benjamini–Hochberg FDR.

The halobacteriota_only stratum excludes Sulfolobaceae (which is
Thermoproteota) but retains Methanosarcinaceae (which is in
Halobacteriota under GTDB).
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from common import FINAL_CSV, MOB_TSV, FISHER_CSV, OUT_DIR, header


def build_fisher_table(df, mob, families, analysis_label, universe=None):
    """Run family × category Fisher tests for a given set of families.

    Parameters
    ----------
    df : DataFrame
        Viral protein hits (one row per replicon × protein).
    mob : DataFrame
        Plasmid metadata with gtdb_family column.
    families : list of str
        GTDB family names to test (with f__ prefix).
    analysis_label : str
        'all_families' or 'halobacteriota_only'.
    universe : set or None
        If provided, restrict all comparisons to this set of plasmid
        accessions.  For all_families this is all viral-carrying plasmids;
        for halobacteriota_only it is Halobacteriota viral-carrying only.

    Returns
    -------
    DataFrame with columns: family, category, OR, p, plasmids_with,
        plasmids_total, analysis.
    """
    if universe is None:
        universe = set(df['replicon'].unique())
    cats = sorted(df['new_category'].unique())

    # Category → set of plasmid accessions (within the universe)
    cat_plasmids = {}
    for cat in cats:
        cat_plasmids[cat] = (
            set(df[df['new_category'] == cat]['replicon'].unique())
            & universe
        )

    rows = []
    for fam in families:
        # Viral-carrying plasmids in this family (within universe)
        fam_ids = set(
            mob[(mob['gtdb_family'] == fam)
                & (mob['sample_id'].isin(universe))]['sample_id']
        )
        other_ids = universe - fam_ids
        n_fam = len(fam_ids)

        for cat in cats:
            a = len(fam_ids & cat_plasmids[cat])      # family with cat
            b = n_fam - a                               # family without cat
            c = len(other_ids & cat_plasmids[cat])     # other with cat
            d = len(other_ids) - c                      # other without cat

            OR, p = stats.fisher_exact([[a, b], [c, d]])

            rows.append({
                'family':         fam.replace('f__', ''),
                'category':       cat,
                'OR':             OR,
                'p':              p,
                'plasmids_with':  a,
                'plasmids_total': n_fam,
                'analysis':       analysis_label,
            })

    return pd.DataFrame(rows)


def main():
    header("GENERATE FISHER ENRICHMENT TABLE")

    df  = pd.read_csv(FINAL_CSV)
    mob = pd.read_csv(MOB_TSV, sep='\t')

    viral_ids = set(df['replicon'].unique())
    mob_viral = mob[mob['sample_id'].isin(viral_ids)]

    # ── Identify eligible families ────────────────────────────────
    fam_counts = (mob_viral.groupby('gtdb_family')['sample_id']
                  .nunique().sort_values(ascending=False))

    # Exclude empty / unclassified family labels
    fam_counts = fam_counts[fam_counts.index.str.len() > 3]  # drop 'f__'

    # Threshold: >= 7 viral-carrying plasmids
    MIN_N = 7
    eligible = fam_counts[fam_counts >= MIN_N].index.tolist()
    print(f"Families with >= {MIN_N} viral-carrying plasmids: {len(eligible)}")
    for fam in eligible:
        print(f"  {fam}: {fam_counts[fam]}")

    # ── Halobacteriota families (GTDB) ────────────────────────────
    halo_fams = set(
        mob_viral[mob_viral['gtdb_phylum'] == 'p__Halobacteriota']
        ['gtdb_family'].unique()
    )
    eligible_halo = [f for f in eligible if f in halo_fams]
    print(f"\nHalobacteriota families in test set: {len(eligible_halo)}")
    for fam in eligible_halo:
        print(f"  {fam}: {fam_counts[fam]}")

    # ── Define universes ──────────────────────────────────────────
    all_viral = set(df['replicon'].unique())

    halo_plasmids = set(
        mob[mob['gtdb_phylum'] == 'p__Halobacteriota']['sample_id']
    )
    halo_viral = all_viral & halo_plasmids
    print(f"\nUniverse — all viral-carrying: {len(all_viral)}")
    print(f"Universe — Halobacteriota viral-carrying: {len(halo_viral)}")

    # ── Build tables ──────────────────────────────────────────────
    df_all  = build_fisher_table(df, mob, eligible, 'all_families',
                                 universe=all_viral)
    df_halo = build_fisher_table(df, mob, eligible_halo, 'halobacteriota_only',
                                 universe=halo_viral)

    # ── BH FDR within each stratum ────────────────────────────────
    for sub in [df_all, df_halo]:
        _, padj, _, _ = multipletests(sub['p'], method='fdr_bh')
        sub['p_adj'] = padj
        sub['significant'] = padj < 0.05

    result = pd.concat([df_all, df_halo], ignore_index=True)

    # ── Summary ───────────────────────────────────────────────────
    print(f"\nAll-families tests:        {len(df_all)}")
    print(f"Halobacteriota-only tests: {len(df_halo)}")
    print(f"Total tests:               {len(result)}")

    sig = result[result['significant']]
    print(f"\nSignificant (FDR < 0.05): {len(sig)}")
    for _, row in sig.iterrows():
        direction = "enriched" if row['OR'] > 1 else "depleted"
        print(f"  {row['family']} × {row['category']} ({row['analysis']}): "
              f"OR = {row['OR']:.2f}, FDR = {row['p_adj']:.3f} ({direction})")

    # ── Write CSV ─────────────────────────────────────────────────
    result.to_csv(FISHER_CSV, index=False)
    print(f"\nWrote {FISHER_CSV}")


if __name__ == "__main__":
    main()
