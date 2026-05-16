#!/usr/bin/env python3
"""Build the per-cluster summary table from the raw MMseqs2 + eggNOG TSV."""
import gc
import numpy as np
import pandas as pd

from common import CLUSTER_TSV, CLUSTER_SUMMARY, OUT_DIR, header


def main():
    header("BUILD CLUSTER SUMMARY")
    df = pd.read_csv(
        CLUSTER_TSV, sep="\t", usecols=[0, 1, 2, 3, 4, 5, 6],
        names=["cluster_rep", "member", "domain", "phylum", "class_", "order", "family"],
        header=0, dtype=str, low_memory=True,
    )
    print(f"  Rows: {len(df):,}")

    # archaeal accession-style rows have no domain assigned
    df.loc[df['domain'].isna(), 'domain'] = 'd__Archaea'

    # propagate phylum/family from labelled archaea members to unlabelled ones
    arch_lab = df[(df['domain'] == 'd__Archaea') & df['phylum'].notna()]
    tax_lookup = (arch_lab.drop_duplicates(subset='cluster_rep', keep='first')
                          .set_index('cluster_rep')[['phylum', 'class_', 'order', 'family']])
    mask = (df['domain'] == 'd__Archaea') & df['phylum'].isna()
    if mask.any():
        reps = df.loc[mask, 'cluster_rep']
        for col in ['phylum', 'class_', 'order', 'family']:
            df.loc[mask, col] = reps.map(tax_lookup[col]).values
    del arch_lab, tax_lookup
    gc.collect()

    df['is_arch'] = (df['domain'] == 'd__Archaea')
    df['is_bact'] = (df['domain'] == 'd__Bacteria')

    counts = df.groupby('cluster_rep').agg(
        n_total=('member', 'size'),
        n_archaea=('is_arch', 'sum'),
        n_bacteria=('is_bact', 'sum'),
    )
    counts['cluster_type'] = np.where(
        (counts['n_archaea'] > 0) & (counts['n_bacteria'] > 0), 'cross-domain',
        np.where(counts['n_archaea'] > 0, 'archaea-only', 'bacteria-only'),
    )

    arch = df[df['is_arch']]
    bact = df[df['is_bact']]
    arch_phyla = arch.groupby('cluster_rep')['phylum'].apply(
        lambda x: "|".join(sorted(x.dropna().unique()))).rename('archaea_phyla')
    bact_phyla = bact.groupby('cluster_rep')['phylum'].apply(
        lambda x: "|".join(sorted(x.dropna().unique()))).rename('bacteria_phyla')
    arch_fams = arch.groupby('cluster_rep')['family'].apply(
        lambda x: "|".join(sorted(x.dropna().unique()))).rename('archaea_families')

    clusters = counts.join(arch_phyla).join(bact_phyla).join(arch_fams).reset_index()
    clusters[['archaea_phyla', 'bacteria_phyla', 'archaea_families']] = (
        clusters[['archaea_phyla', 'bacteria_phyla', 'archaea_families']].fillna("")
    )

    clusters.to_csv(CLUSTER_SUMMARY, index=False)
    print(f"\n  Proteins total:    {len(df):,}")
    print(f"  Archaeal:          {int(df['is_arch'].sum()):,}")
    print(f"  Bacterial:         {int(df['is_bact'].sum()):,}")
    print(f"  Archaeal phyla:    {arch['phylum'].dropna().nunique()}")
    print(f"  Bacterial phyla:   {bact['phylum'].dropna().nunique()}")
    print(f"  Clusters:          {len(clusters):,}")
    print(clusters['cluster_type'].value_counts().to_string())
    print(f"\n  Wrote {CLUSTER_SUMMARY.relative_to(OUT_DIR.parent)}")


if __name__ == "__main__":
    main()
