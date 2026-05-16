#!/usr/bin/env python3
"""Same-family CRISPR targeting rate, per-family breakdown, and convergent targets.

Expected same-family rate under random sampling is Simpson's sum(p_f^2) over
target families; binomial test against that null gives the fold-enrichment p.
"""
from collections import defaultdict
import pandas as pd
from scipy.stats import binomtest

from common import BLAST_TSV, OUT_DIR, header


def main():
    header("WITHIN-FAMILY CRISPR TARGETING")
    hits = pd.read_csv(BLAST_TSV, sep='\t')

    plasmid_hits = hits[hits['target_category'] == 'plasmid'].copy()
    n_total = len(plasmid_hits)
    if n_total == 0:
        print("No plasmid-target hits.")
        return

    plasmid_hits['same_family'] = (
        plasmid_hits['source_family'] == plasmid_hits['target_family'])
    same = int(plasmid_hits['same_family'].sum())
    print(f"Plasmid hits:        {n_total}")
    print(f"Same-family hits:    {same}  ({same/n_total*100:.1f}%)")

    fam_counts = plasmid_hits['target_family'].value_counts(normalize=True)
    expected_rate = float((fam_counts ** 2).sum())
    print(f"Expected same-family rate: {expected_rate*100:.1f}%")
    print(f"Fold enrichment:           {(same/n_total) / expected_rate:.2f}x")

    bt = binomtest(same, n_total, expected_rate, alternative='greater')
    print(f"Binomial test (greater):   p = {bt.pvalue:.2e}")

    print("\nPer-family within-family rate:")
    fam_rates = []
    for fam, grp in plasmid_hits.groupby('source_family'):
        n = len(grp)
        if n < 5:
            continue
        same_n = int((grp['same_family']).sum())
        fam_rates.append({'source_family': fam, 'n_hits': n,
                          'within_family': same_n,
                          'pct_within': round(100 * same_n / n, 1)})
    fr = pd.DataFrame(fam_rates).sort_values('pct_within', ascending=False)
    print(fr.to_string(index=False))
    fr.to_csv(OUT_DIR / 'crispr_within_family.csv', index=False)

    # convergent targets — plasmids hit by spacers from > 1 unrelated source
    target_sources = defaultdict(set)
    for _, r in plasmid_hits.iterrows():
        target_sources[r['target_plasmid']].add(r['source_plasmid'])
    convergent = {t: s for t, s in target_sources.items() if len(s) > 1}
    print(f"\nTargets recognised by ≥ 2 unrelated sources: "
          f"{len(convergent)} (of {len(target_sources)} unique targets)")
    if convergent:
        max_t, max_s = max(convergent.items(), key=lambda kv: len(kv[1]))
        print(f"  Most-recognised: {max_t} ({len(max_s)} distinct sources)")
    pd.DataFrame([(t, len(s)) for t, s in target_sources.items()],
                 columns=['target_plasmid', 'n_unique_sources']).to_csv(
        OUT_DIR / 'convergent_targets.csv', index=False)


if __name__ == "__main__":
    main()
