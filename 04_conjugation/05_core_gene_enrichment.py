#!/usr/bin/env python3
"""Per-subtype core families; OR enrichment; optional mechanism labels."""
import pandas as pd

from common import CORE_ENRICH_TSV, OUT_DIR, mechanism_patterns, header


def main():
    header("CORE GENE ENRICHMENT")
    enr = pd.read_csv(CORE_ENRICH_TSV, sep='\t')

    subtypes = sorted(enr['subtype'].dropna().unique())
    print("Core families per subtype (excl. VirB4/T4CP):")
    for st in subtypes:
        n = (enr['subtype'] == st).sum()
        print(f"  {st}: {n}")

    # how many subtypes does each cluster bridge?
    shared = enr.groupby('cluster_id')['subtype'].apply(list).reset_index()
    shared['n_subtypes'] = shared['subtype'].apply(len)
    multi = shared[shared['n_subtypes'] > 1]
    print(f"\nClusters spanning > 1 subtype: {len(multi)}")
    print(f"Max subtypes bridged:           {shared['n_subtypes'].max()}")

    enriched = enr[enr['enriched_on_conjugative'] == True]
    finite = enriched[enriched['odds_ratio'] != float('inf')]
    if not finite.empty:
        mx = finite.loc[finite['odds_ratio'].idxmax()]
        mn = finite.loc[finite['odds_ratio'].idxmin()]
        print(f"\nFinite OR range: {mn['odds_ratio']:.1f} ({mn['subtype']}, "
              f"{mn.get('eggnog_description', '')})")
        print(f"            to:  {mx['odds_ratio']:.1f} ({mx['subtype']}, "
              f"{mx.get('eggnog_description', '')})")
    inf_or = enriched[enriched['odds_ratio'] == float('inf')]
    print(f"Clusters exclusive to conjugative plasmids (OR = ∞): {len(inf_or)}")

    patterns = mechanism_patterns()
    if patterns:
        print("\nMechanism labels (from mechanism_patterns.tsv):")
        desc = enr['eggnog_description'].fillna('')
        for label, pat in patterns.items():
            hits = enr[desc.str.contains(pat, case=False, regex=True)]
            if hits.empty:
                print(f"  {label}: no match")
                continue
            for _, r in hits.iterrows():
                print(f"  {label}: {r['subtype']} {r['cluster_id']} "
                      f"(OR = {r['odds_ratio']:.1f}; {r['eggnog_description']})")
    else:
        print("\n(no mechanism_patterns.tsv provided)")

    enr.to_csv(OUT_DIR / "core_genes_enrichment_full.csv", index=False)


if __name__ == "__main__":
    main()
