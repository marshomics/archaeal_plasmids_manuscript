#!/usr/bin/env python3
"""Spacer / source-plasmid counts; per-array target exclusivity."""
from collections import Counter
import pandas as pd

from common import SPACER_FASTA, BLAST_TSV, OUT_DIR, header


def _parse_spacer_fasta(path):
    """Header format expected: >source_plasmid__spacer_id"""
    pairs = []
    with open(path) as f:
        header_line = None
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                header_line = line[1:]
            elif line and header_line:
                source = header_line.split('__')[0]
                pairs.append((source, line))
                header_line = None
    return pairs


def main():
    header("CRISPR SPACER INVENTORY")
    spacers = _parse_spacer_fasta(SPACER_FASTA)
    source_counts = Counter(src for src, _ in spacers)
    print(f"Total spacers:                   {len(spacers)}")
    print(f"Source CRISPR-bearing plasmids:  {len(source_counts)}")

    hits = pd.read_csv(BLAST_TSV, sep='\t')
    if 'source_plasmid' not in hits.columns or 'target_category' not in hits.columns:
        print("\nBLAST table missing source_plasmid / target_category — skipping.")
        return
    per_source = hits.groupby('source_plasmid')['target_category'].apply(
        lambda x: set(x.dropna())).reset_index(name='cats')
    only_plasmid = (per_source['cats'] == {'plasmid'}).sum()
    only_virus   = (per_source['cats'] == {'virus'}).sum()
    both         = (per_source['cats'] == {'plasmid', 'virus'}).sum()
    n_arrays_with_hits = len(per_source)

    print("\nArray-level target exclusivity:")
    print(f"  Plasmid-only:  {only_plasmid}/{n_arrays_with_hits}")
    print(f"  Virus-only:    {only_virus}/{n_arrays_with_hits}")
    print(f"  Both:          {both}/{n_arrays_with_hits}")

    per_source['exclusivity'] = per_source['cats'].apply(
        lambda s: 'plasmid_only' if s == {'plasmid'} else
                  'virus_only'   if s == {'virus'}   else
                  'both' if s == {'plasmid', 'virus'} else 'none')
    per_source[['source_plasmid', 'exclusivity']].to_csv(
        OUT_DIR / 'array_target_exclusivity.csv', index=False)


if __name__ == "__main__":
    main()
