#!/usr/bin/env python3
"""Intra-archaea CLR enrichment — single-COG attribution per protein.

UPDATE vs streamlined_cross_domain/06_archaea_only_clr_enrichment.py
--------------------------------------------------------------------
Original `_explode_cog` produced one row per (replicon, COG-letter),
so a protein with k COG letters contributed k times to that replicon's
COG totals. After grouping to a plasmid × COG matrix this inflated the
row sums above the true protein count and biased the CLR geometric mean
and the per-letter deltas.

This version uses weighted contributions: a protein's contribution to
each of its k COG letters is 1/k. The plasmid × COG count matrix is then
the sum of these per-letter weights per plasmid, preserving the
invariant that each plasmid's total weight equals its annotated protein
count.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "streamlined_cross_domain"))

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from common import (ARCHAEAL_EGGNOG, COG_DESCRIPTIONS,
                    N_PERM, PSEUDO, FAM_MIN, header)

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)
MIN_TOTAL_COUNT = 50    # drop sparse COGs


def _clr_transform(count_df, pseudo):
    log_counts = np.log(count_df + pseudo)
    geo_mean = log_counts.mean(axis=1)
    return log_counts.sub(geo_mean, axis=0)


EXCLUDE_COGS = {'R', 'S'}   # R = general prediction; S = unknown function


def _weighted_cog_rows(df):
    """One row per (replicon, COG) with weight = 1/k for each protein.

    R and S are excluded so the CLR is computed over characterised
    functional categories only.  Proteins whose only COG letter is R or S
    (or that are unannotated) are dropped entirely.
    """
    df = df.copy()
    rows = []
    for _, r in df.iterrows():
        cogs = [c for c in str(r['COG_category'])
                if c in COG_DESCRIPTIONS and c not in EXCLUDE_COGS]
        if not cogs:
            continue          # unannotated / S-only / R-only → skip
        w = 1.0 / len(cogs)
        for c in cogs:
            rows.append({'replicon': r['replicon'],
                         'gtdb_phylum': r['gtdb_phylum'],
                         'gtdb_family': r['gtdb_family'],
                         'gtdb_species': r['gtdb_species'],
                         'COG': c, 'w': w})
    return pd.DataFrame(rows)


def _clr_perm_test(plasmid_clr, weights, groups, n_perm, min_plas=1):
    rng = np.random.default_rng(0)
    common = plasmid_clr.index.intersection(groups.index).intersection(weights.index)
    clr = plasmid_clr.loc[common]
    w   = weights.loc[common].to_numpy()
    g   = groups.loc[common]
    results = []
    for grp_name, members in g.groupby(g).groups.items():
        focal = list(members)
        if len(focal) < min_plas:
            continue
        mask_focal = clr.index.isin(focal)
        for cog in clr.columns:
            x = clr[cog].to_numpy()
            obs_focal = np.average(x[mask_focal], weights=w[mask_focal])
            obs_bg    = np.average(x[~mask_focal], weights=w[~mask_focal])
            obs_delta = obs_focal - obs_bg
            count = 0
            for _ in range(n_perm):
                perm = rng.permutation(mask_focal)
                pd_delta = (np.average(x[perm], weights=w[perm]) -
                            np.average(x[~perm], weights=w[~perm]))
                if abs(pd_delta) >= abs(obs_delta):
                    count += 1
            p_raw = (count + 1) / (n_perm + 1)
            results.append({'group': grp_name, 'COG': cog,
                            'description': COG_DESCRIPTIONS[cog],
                            'delta': obs_delta, 'p_raw': p_raw,
                            'n_plasmids': len(focal)})
    res = pd.DataFrame(results)
    res['p_adj'] = multipletests(res['p_raw'], method='fdr_bh')[1]
    res['significant'] = res['p_adj'] < 0.05
    return res


def main():
    header("INTRA-ARCHAEA CLR ENRICHMENT (updated: weighted COG attribution)")
    df = pd.read_csv(ARCHAEAL_EGGNOG, sep='\t', low_memory=False)
    print(f"Archaeal plasmid proteins: {len(df):,}")
    df['phylum_short'] = df['gtdb_phylum'].str.replace('p__', '', regex=False)
    df['family_short'] = df['gtdb_family'].str.replace('f__', '', regex=False)

    weighted = _weighted_cog_rows(df)
    print(f"Weighted (replicon, COG) rows: {len(weighted):,}  "
          f"(total weight ≈ {weighted['w'].sum():,.0f} = annotated protein count)")

    # Plasmid × COG weighted-count matrix.
    pc = (weighted
          .groupby(['replicon', 'COG'])['w']
          .sum()
          .unstack(fill_value=0))
    sparse = pc.columns[pc.sum(0) < MIN_TOTAL_COUNT].tolist()
    pc = pc.drop(columns=sparse)

    sp = weighted.groupby('replicon')['gtdb_species'].first()
    counts_per_sp = weighted.groupby('gtdb_species')['replicon'].nunique()
    weights_per_plasmid = sp.map(lambda s: 1.0 / counts_per_sp.get(s, 1))

    plasmid_clr = _clr_transform(pc, PSEUDO)
    phylum = df.groupby('replicon')['phylum_short'].first()
    family = df.groupby('replicon')['family_short'].first()

    res_p = _clr_perm_test(plasmid_clr, weights_per_plasmid, phylum, N_PERM)
    res_f = _clr_perm_test(plasmid_clr, weights_per_plasmid, family, N_PERM,
                           min_plas=FAM_MIN)

    res_p.to_csv(OUT_DIR / "CLR_enrichment_phylum.csv", index=False)
    res_f.to_csv(OUT_DIR / "CLR_enrichment_family.csv", index=False)

    print("\nPhylum-level (significant):")
    print(res_p[res_p['significant']]
          .sort_values(['group', 'delta'], ascending=[True, False])
          .to_string(index=False))
    print("\nFamily-level (significant):")
    print(res_f[res_f['significant']]
          .sort_values(['group', 'delta'], ascending=[True, False])
          .to_string(index=False))


if __name__ == "__main__":
    main()
