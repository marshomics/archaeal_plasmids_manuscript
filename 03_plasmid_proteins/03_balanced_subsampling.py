#!/usr/bin/env python3
"""Bootstrap balanced subsampling at three levels:
  1. Species-level:  subsample each phylum to the smallest viable phylum size
  2. Protein-level:  subsample proteins per phylum to the smallest viable count
  3. Plasmid-level:  subsample plasmids per phylum to the smallest viable count

All three compute cross-domain sharing fraction with N_BOOT iterations and 95% CIs.
"""
import csv, re
from collections import defaultdict
import numpy as np
import pandas as pd

from common import CLUSTER_TSV, CLUSTER_SUMMARY, ARCHAEAL_EGGNOG, OUT_DIR, N_BOOT, header


def _normalize_id(pid):
    m = re.match(r'^(.+)_(\d+)$', str(pid))
    if m:
        return f'{m.group(1)}_{int(m.group(2)):05d}'
    return pid


def _build_phylum_species_clusters():
    """Return {phylum: {species: set((cluster_rep, cluster_type))}} for archaea."""
    ct_map = dict(pd.read_csv(CLUSTER_SUMMARY,
                              usecols=['cluster_rep', 'cluster_type']).itertuples(
        index=False, name=None))
    psc = defaultdict(lambda: defaultdict(set))
    with open(CLUSTER_TSV) as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            rep = row[0]
            domain = row[2] if len(row) > 2 and row[2] else 'd__Archaea'
            if domain != 'd__Archaea':
                continue
            phylum = row[3] if len(row) > 3 else ''
            species = (row[8] if len(row) > 8 and row[8]
                       else (row[7] if len(row) > 7 else ''))
            if not (phylum and species):
                continue
            psc[phylum][species].add((rep, ct_map.get(rep, 'unknown')))
    return psc


def _species_subsampling(psc, n_boot, rng):
    """Species-level balanced subsampling."""
    sp_counts = {p: len(sps) for p, sps in psc.items()}
    viable = {p: n for p, n in sp_counts.items() if n >= 5}
    if not viable:
        return pd.DataFrame()
    min_sp = min(viable.values())
    print(f"  Species-level: {min_sp} species/phylum, {len(viable)} viable phyla")

    rows = []
    for phylum, sp_clusters in psc.items():
        sp_list = list(sp_clusters)
        n_sp = len(sp_list)
        if n_sp < 5:
            continue
        for b in range(n_boot):
            sampled = (rng.choice(sp_list, size=min_sp, replace=False)
                       if n_sp > min_sp else sp_list)
            seen = set().union(*(sp_clusters[sp] for sp in sampled))
            n_cl = len(seen)
            n_cd = sum(1 for _, ct in seen if ct == 'cross-domain')
            rows.append({'phylum': phylum, 'level': 'species',
                         'frac_cross': n_cd / n_cl if n_cl else 0.0,
                         'n_sampled': min_sp, 'n_total': n_sp})

    return pd.DataFrame(rows)


def _protein_subsampling(n_boot, rng):
    """Protein-level balanced subsampling."""
    # Load annotations with phylum
    annot = pd.read_csv(ARCHAEAL_EGGNOG, sep='\t',
                        usecols=['proteins', 'gtdb_phylum'],
                        dtype=str, low_memory=False)
    annot['pid'] = annot['proteins'].apply(_normalize_id)
    annot = annot.dropna(subset=['gtdb_phylum'])

    # Build member → cluster map
    member_cluster = {}
    with open(CLUSTER_TSV) as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            member_cluster[row[1]] = row[0]

    # Load cluster types
    ct_map = dict(pd.read_csv(CLUSTER_SUMMARY,
                              usecols=['cluster_rep', 'cluster_type']).itertuples(
        index=False, name=None))

    annot['cluster_rep'] = annot['pid'].map(member_cluster)
    annot = annot.dropna(subset=['cluster_rep'])
    annot['cluster_type'] = annot['cluster_rep'].map(ct_map)
    annot = annot.dropna(subset=['cluster_type'])

    # Group by phylum
    phylum_data = {}
    for phy in annot['gtdb_phylum'].unique():
        sub = annot[annot['gtdb_phylum'] == phy]
        phylum_data[phy] = {'ct': sub['cluster_type'].values, 'n': len(sub)}

    viable = {p: d['n'] for p, d in phylum_data.items() if d['n'] >= 25}
    if not viable:
        return pd.DataFrame()
    min_prot = min(viable.values())
    print(f"  Protein-level: {min_prot} proteins/phylum, {len(viable)} viable phyla")

    rows = []
    for phy, data in phylum_data.items():
        if data['n'] < 25:
            continue
        ct_arr = data['ct']
        n_total = data['n']
        sample_size = min(min_prot, n_total)
        for b in range(n_boot):
            if n_total > min_prot:
                idx = rng.choice(n_total, size=sample_size, replace=False)
                sampled = ct_arr[idx]
            else:
                sampled = ct_arr
            n_cross = (sampled == 'cross-domain').sum()
            rows.append({'phylum': phy, 'level': 'protein',
                         'frac_cross': n_cross / sample_size,
                         'n_sampled': sample_size, 'n_total': n_total})

    return pd.DataFrame(rows)


def _plasmid_subsampling(n_boot, rng):
    """Plasmid-level balanced subsampling."""
    ct_map = dict(pd.read_csv(CLUSTER_SUMMARY,
                              usecols=['cluster_rep', 'cluster_type']).itertuples(
        index=False, name=None))

    # Build replicon → (phylum, list of cluster_types)
    replicon_data = defaultdict(lambda: {'phylum': '', 'types': []})
    with open(CLUSTER_TSV) as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            rep = row[0]
            member = row[1]
            domain = row[2] if len(row) > 2 and row[2] else 'd__Archaea'
            if domain != 'd__Archaea':
                continue
            phylum = row[3] if len(row) > 3 else ''
            # Extract replicon (everything before last _ and protein number)
            parts = member.rsplit('_', 1)
            replicon = parts[0] if len(parts) == 2 else member
            ct = ct_map.get(rep, 'unknown')
            replicon_data[replicon]['types'].append(ct)
            if not replicon_data[replicon]['phylum'] and phylum:
                replicon_data[replicon]['phylum'] = phylum

    # Group replicons by phylum
    phylum_replicons = defaultdict(list)
    for repl, data in replicon_data.items():
        if data['phylum']:
            types_arr = np.array(data['types'])
            n_total = len(types_arr)
            n_cross = (types_arr == 'cross-domain').sum()
            phylum_replicons[data['phylum']].append({
                'replicon': repl, 'n_total': n_total, 'n_cross': n_cross,
                'frac_cross': n_cross / n_total if n_total > 0 else 0
            })

    viable = {p: len(repls) for p, repls in phylum_replicons.items() if len(repls) >= 3}
    if not viable:
        return pd.DataFrame()
    min_plas = min(viable.values())
    print(f"  Plasmid-level: {min_plas} plasmids/phylum, {len(viable)} viable phyla")

    rows = []
    for phy, repls in phylum_replicons.items():
        n_repls = len(repls)
        if n_repls < 3:
            continue
        sample_size = min(min_plas, n_repls)
        for b in range(n_boot):
            if n_repls > min_plas:
                sampled = rng.choice(repls, size=sample_size, replace=False)
            else:
                sampled = repls
            total_prot = sum(r['n_total'] for r in sampled)
            total_cross = sum(r['n_cross'] for r in sampled)
            frac = total_cross / total_prot if total_prot > 0 else 0
            rows.append({'phylum': phy, 'level': 'plasmid',
                         'frac_cross': frac,
                         'n_sampled': sample_size, 'n_total': n_repls})

    return pd.DataFrame(rows)


def _summarize(df):
    """Summarize bootstrap results with mean and 95% CI."""
    return df.groupby(['phylum', 'level']).agg(
        mean_frac_cross=('frac_cross', 'mean'),
        ci_lo=('frac_cross', lambda x: np.percentile(x, 2.5)),
        ci_hi=('frac_cross', lambda x: np.percentile(x, 97.5)),
        n_sampled=('n_sampled', 'first'),
        n_total=('n_total', 'first'),
    ).reset_index().sort_values(['level', 'mean_frac_cross'], ascending=[True, False])


def main():
    header("BALANCED SUBSAMPLING (3 levels)")
    rng = np.random.default_rng(42)

    # 1. Species-level
    psc = _build_phylum_species_clusters()
    sp_df = _species_subsampling(psc, N_BOOT, rng)

    # 2. Protein-level
    prot_df = _protein_subsampling(N_BOOT, rng)

    # 3. Plasmid-level
    plas_df = _plasmid_subsampling(N_BOOT, rng)

    # Combine and summarize
    all_df = pd.concat([sp_df, prot_df, plas_df], ignore_index=True)
    summary = _summarize(all_df)

    # Save
    summary.to_csv(OUT_DIR / "balanced_subsampling_sharing.csv", index=False)

    for level in ['species', 'protein', 'plasmid']:
        sub = summary[summary['level'] == level]
        if sub.empty:
            continue
        print(f"\n  {level.capitalize()}-level subsampling:")
        for _, r in sub.iterrows():
            short = r['phylum'].replace('p__', '')
            print(f"    {short:<26} frac_cross = {r['mean_frac_cross']:.4f}  "
                  f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]  "
                  f"(n={int(r['n_sampled'])} of {int(r['n_total'])})")

    print(f"\nWrote {(OUT_DIR / 'balanced_subsampling_sharing.csv').name}")


if __name__ == "__main__":
    main()
