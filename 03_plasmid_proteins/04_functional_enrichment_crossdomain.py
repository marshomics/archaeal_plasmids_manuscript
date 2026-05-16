#!/usr/bin/env python3
"""Per-COG Fisher enrichment, cross-domain vs archaea-only. BH-FDR.

Adds a bootstrap on the pooled metabolic-OR (categories E + G + I): 500
resamples with replacement of the protein-level counts, Shapiro-Wilk on the
bootstrap distribution, and the 95% percentile CI.

COG annotations come from the archaeal eggNOG file (ARCHAEAL_EGGNOG),
joined to cluster membership via protein IDs from the cluster TSV.
"""
import csv, re
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, shapiro
from statsmodels.stats.multitest import multipletests

from common import (CLUSTER_TSV, CLUSTER_SUMMARY, ARCHAEAL_EGGNOG,
                    OUT_DIR, COG_DESCRIPTIONS, header)

METABOLIC_COGS = ['E', 'G', 'I']    # amino acid, carbohydrate, lipid
N_BOOT = 500


def _normalize_id(pid):
    m = re.match(r'^(.+)_(\d+)$', str(pid))
    if m:
        return f'{m.group(1)}_{int(m.group(2)):05d}'
    return pid


def main():
    header("CROSS-DOMAIN COG ENRICHMENT")

    # Step 1: cluster_rep → cluster_type
    cluster_type = dict(pd.read_csv(CLUSTER_SUMMARY,
                                    usecols=['cluster_rep', 'cluster_type']).itertuples(
        index=False, name=None))

    # Step 2: member protein → cluster_rep (from cluster TSV)
    print("  Building member → cluster_rep map...")
    member_cluster = {}
    with open(CLUSTER_TSV) as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            member_cluster[row[1]] = row[0]
    print(f"  Member→cluster mappings: {len(member_cluster):,}")

    # Step 3: Load COG annotations from eggNOG file
    print("  Loading COG annotations from eggNOG file...")
    annot = pd.read_csv(ARCHAEAL_EGGNOG, sep='\t',
                        usecols=['proteins', 'COG_category'],
                        dtype=str, low_memory=False)
    annot['pid'] = annot['proteins'].apply(_normalize_id)
    annot['cluster_rep'] = annot['pid'].map(member_cluster)
    annot = annot.dropna(subset=['cluster_rep'])
    annot['cluster_type'] = annot['cluster_rep'].map(cluster_type)
    annot = annot[annot['cluster_type'].isin(['cross-domain', 'archaea-only'])]
    annot['COG_category'] = annot['COG_category'].fillna('S')
    annot.loc[annot['COG_category'] == '-', 'COG_category'] = 'S'
    annot.loc[annot['COG_category'].str.strip() == '', 'COG_category'] = 'S'
    print(f"  Mapped proteins with cluster assignment: {len(annot):,}")

    # Step 4: Count COG occurrences per cluster type
    records = []
    cross_counts = {c: 0 for c in COG_DESCRIPTIONS}
    only_counts = {c: 0 for c in COG_DESCRIPTIONS}
    for _, row in annot.iterrows():
        ct = row['cluster_type']
        cog_str = row['COG_category']
        target = cross_counts if ct == 'cross-domain' else only_counts
        for ch in cog_str:
            if ch in target:
                target[ch] += 1
                records.append((ct, ch))

    records = pd.DataFrame(records, columns=['cluster_type', 'COG'])

    total_cross = sum(cross_counts.values())
    total_only  = sum(only_counts.values())
    rows = []
    for cog in COG_DESCRIPTIONS:
        a = cross_counts[cog]
        c = only_counts[cog]
        if a + c < 10:
            continue
        b = total_cross - a
        d = total_only - c
        OR, p = fisher_exact([[a, b], [c, d]])
        rows.append({'COG': cog, 'description': COG_DESCRIPTIONS[cog],
                     'n_cross': a, 'n_only': c,
                     'odds_ratio': OR, 'p_raw': p})
    out = pd.DataFrame(rows)
    out['p_adj'] = multipletests(out['p_raw'], method='fdr_bh')[1]
    out = out.sort_values('odds_ratio', ascending=False)
    out.to_csv(OUT_DIR / "cog_enrichment_crossdomain.csv", index=False)
    pd.set_option('display.float_format', '{:.3g}'.format)
    print(out[['COG', 'description', 'odds_ratio', 'p_adj']].to_string(index=False))

    # pooled metabolic-OR bootstrap (E + G + I, the transport-and-metabolism categories)
    print(f"\nMetabolic-OR bootstrap (COGs {METABOLIC_COGS}, {N_BOOT} replicates):")
    pooled = records[records['COG'].isin(METABOLIC_COGS)]
    cross_pooled = int((pooled['cluster_type'] == 'cross-domain').sum())
    only_pooled  = int((pooled['cluster_type'] == 'archaea-only').sum())
    b_pool = total_cross - cross_pooled
    d_pool = total_only - only_pooled
    OR_obs, _ = fisher_exact([[cross_pooled, b_pool], [only_pooled, d_pool]])
    print(f"  Observed pooled OR: {OR_obs:.3f}")

    rng = np.random.default_rng(0)
    records_arr = records.to_numpy()
    n = len(records_arr)
    boot_ors = np.empty(N_BOOT)
    for i in range(N_BOOT):
        sample = records_arr[rng.integers(0, n, n)]
        s_cross = (sample[:, 0] == 'cross-domain')
        s_metab = np.isin(sample[:, 1], METABOLIC_COGS)
        a_b = int((s_cross & s_metab).sum())
        c_b = int((~s_cross & s_metab).sum())
        b_b = int(s_cross.sum()) - a_b
        d_b = int((~s_cross).sum()) - c_b
        try:
            boot_ors[i], _ = fisher_exact([[a_b, b_b], [c_b, d_b]])
        except Exception:
            boot_ors[i] = np.nan

    boot_ors = boot_ors[~np.isnan(boot_ors)]
    log_ors = np.log(boot_ors[boot_ors > 0])
    sw_stat, sw_p = shapiro(boot_ors[:5000]) if len(boot_ors) > 3 else (np.nan, np.nan)
    sw_stat_log, sw_p_log = shapiro(log_ors[:5000]) if len(log_ors) > 3 else (np.nan, np.nan)

    ci_lo, ci_hi = np.percentile(boot_ors, [2.5, 97.5])
    print(f"  Bootstrap mean OR:  {boot_ors.mean():.3f}")
    print(f"  Bootstrap 95% CI:   [{ci_lo:.3f}, {ci_hi:.3f}]")
    print(f"  Shapiro-Wilk on OR:     W = {sw_stat:.3f}, p = {sw_p:.2e}")
    print(f"  Shapiro-Wilk on log OR: W = {sw_stat_log:.3f}, p = {sw_p_log:.2e}")

    pd.DataFrame({'bootstrap_OR': boot_ors}).to_csv(
        OUT_DIR / 'metabolic_or_bootstrap.csv', index=False)


if __name__ == "__main__":
    main()
