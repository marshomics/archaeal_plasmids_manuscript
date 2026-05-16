#!/usr/bin/env python3
"""Intra-archaea CLR enrichment, phylum and family levels, with permutation test.

CLR(x_i) = log((x_i + pseudo) / G) per plasmid × COG matrix, with G the
geometric mean across categories. Permutation shuffles focal vs background
group labels; inverse-species-frequency weighting on plasmid means.
"""
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from common import (ARCHAEAL_EGGNOG, OUT_DIR, COG_DESCRIPTIONS,
                    N_PERM, PSEUDO, FAM_MIN, header)

MIN_TOTAL_COUNT = 50    # drop sparse COGs


def _clr_transform(count_df, pseudo):
    log_counts = np.log(count_df + pseudo)
    geo_mean = log_counts.mean(axis=1)
    return log_counts.sub(geo_mean, axis=0)


def _explode_cog(df):
    df = df.copy()
    df['COG_category'] = df['COG_category'].fillna('S')
    rows = []
    for _, r in df.iterrows():
        cogs = [c for c in str(r['COG_category']) if c in COG_DESCRIPTIONS]
        if not cogs:
            cogs = ['S']
        for c in cogs:
            rows.append({'replicon': r['replicon'], 'gtdb_phylum': r['gtdb_phylum'],
                         'gtdb_family': r['gtdb_family'],
                         'gtdb_species': r['gtdb_species'], 'COG': c})
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
                m1 = perm; m2 = ~perm
                pd_delta = (np.average(x[m1], weights=w[m1]) -
                            np.average(x[m2], weights=w[m2]))
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
    header("INTRA-ARCHAEA CLR ENRICHMENT")
    df = pd.read_csv(ARCHAEAL_EGGNOG, sep='\t', low_memory=False)
    print(f"Archaeal plasmid proteins: {len(df):,}")
    df['phylum_short'] = df['gtdb_phylum'].str.replace('p__', '', regex=False)
    df['family_short'] = df['gtdb_family'].str.replace('f__', '', regex=False)

    exploded = _explode_cog(df)

    pc = exploded.groupby(['replicon', 'COG']).size().unstack(fill_value=0)
    sparse = pc.columns[pc.sum(0) < MIN_TOTAL_COUNT].tolist()
    pc = pc.drop(columns=sparse)

    sp = exploded.groupby('replicon')['gtdb_species'].first()
    counts_per_sp = exploded.groupby('gtdb_species')['replicon'].nunique()
    weights = sp.map(lambda s: 1.0 / counts_per_sp.get(s, 1))

    plasmid_clr = _clr_transform(pc, PSEUDO)
    phylum = df.groupby('replicon')['phylum_short'].first()
    family = df.groupby('replicon')['family_short'].first()

    res_p = _clr_perm_test(plasmid_clr, weights, phylum, N_PERM)
    res_f = _clr_perm_test(plasmid_clr, weights, family, N_PERM, min_plas=FAM_MIN)

    res_p.to_csv(OUT_DIR / "CLR_enrichment_phylum.csv", index=False)
    res_f.to_csv(OUT_DIR / "CLR_enrichment_family.csv", index=False)

    print("\nPhylum-level (significant rows):")
    print(res_p[res_p['significant']]
          .sort_values(['group', 'delta'], ascending=[True, False])
          .to_string(index=False))
    print("\nFamily-level (significant rows):")
    print(res_f[res_f['significant']]
          .sort_values(['group', 'delta'], ascending=[True, False])
          .to_string(index=False))


if __name__ == "__main__":
    main()
