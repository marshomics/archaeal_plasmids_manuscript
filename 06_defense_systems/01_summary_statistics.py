#!/usr/bin/env python3
"""Defence carriage stats and per-(sub)type prevalence."""
import json
import pandas as pd

from common import load_defense_tables, OUT_DIR, header


def main():
    header("DEFENCE PREVALENCE & DOMINANT SUBTYPES")
    type_df, binary_type, sub_df, type_cols, sub_cols = load_defense_tables()
    N = len(type_df)

    n0 = int((type_df['n_instances'] == 0).sum())
    n1 = int((type_df['n_instances'] == 1).sum())
    n2p = int((type_df['n_instances'] >= 2).sum())
    print(f"Plasmids:        {N}")
    print(f"No defence:      {n0} ({n0/N*100:.1f}%)")
    print(f"≥ 1 defence:     {N-n0} ({(N-n0)/N*100:.1f}%)")
    print(f"≥ 2 defences:    {n2p} ({n2p/N*100:.1f}%)")

    type_prev = pd.DataFrame({
        'type': type_cols,
        'count': [int((binary_type[c] > 0).sum()) for c in type_cols],
    })
    type_prev['pct'] = 100 * type_prev['count'] / N
    type_prev = type_prev.sort_values('count', ascending=False)
    type_prev.to_csv(OUT_DIR / 'type_prevalence.csv', index=False)
    print("\nTop 10 defence types:")
    for _, row in type_prev.head(10).iterrows():
        print(f"  {row['type']:<20} {int(row['count']):>4} ({row['pct']:.1f}%)")

    sub_prev = pd.DataFrame({
        'subtype': sub_cols,
        'count': [int((sub_df[c] > 0).sum()) for c in sub_cols],
        'instances': [int(sub_df[c].sum()) for c in sub_cols],
    }).sort_values('count', ascending=False)
    sub_prev.to_csv(OUT_DIR / 'subtype_prevalence.csv', index=False)

    # roll up by prefix so we see the dominant subtype within each family
    print("\nDominant RM subtypes:")
    for _, row in sub_prev[sub_prev['subtype'].str.startswith('RM')].head(5).iterrows():
        print(f"  {row['subtype']:<20} {row['count']} ({row['instances']} inst.)")
    print("\nDominant CRISPR-Cas subtypes:")
    for _, row in sub_prev[sub_prev['subtype'].str.startswith('CAS')].head(5).iterrows():
        print(f"  {row['subtype']:<20} {row['count']} ({row['instances']} inst.)")

    with open(OUT_DIR / 'summary_statistics.json', 'w') as f:
        json.dump({
            'n_plasmids': N, 'pct_no_defense': round(n0/N*100, 1),
            'pct_one_plus': round((N-n0)/N*100, 1),
            'pct_two_plus': round(n2p/N*100, 1),
        }, f, indent=2)


if __name__ == "__main__":
    main()
