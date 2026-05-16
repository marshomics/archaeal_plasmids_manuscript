#!/usr/bin/env python3
"""Sulfolobaceae: phage-replication × conjugative-status Fisher with 95% CI."""
import pandas as pd
from common import load_data, header, fisher_exact_with_ci, OUT_DIR


def main():
    header("SULFOLOBACEAE PHAGE-REPLICATION × CONJUGATIVE STATUS")
    df, mob, _, conj_list, *_ = load_data()

    sulfo_ids = set(mob.loc[mob['gtdb_family'] == 'f__Sulfolobaceae', 'sample_id'])
    sulfo_conj    = sulfo_ids & conj_list
    sulfo_nonconj = sulfo_ids - conj_list
    sulfo_viral_df = df[df['replicon'].isin(sulfo_ids)]
    sulfo_viral_ids = set(sulfo_viral_df['replicon'])
    print(f"Sulfolobaceae plasmids:      {len(sulfo_ids)}")
    print(f"  Conjugative:               {len(sulfo_conj)}")
    print(f"  Non-conjugative:           {len(sulfo_nonconj)}")
    print(f"  With viral proteins:       {len(sulfo_viral_ids)}")

    conj_v    = sulfo_viral_ids & sulfo_conj
    nonconj_v = sulfo_viral_ids & sulfo_nonconj
    phage_ids = set(sulfo_viral_df.loc[
        sulfo_viral_df['new_category'] == 'Phage replication', 'replicon'])

    a = len(conj_v & phage_ids)
    b = len(conj_v - phage_ids)
    c = len(nonconj_v & phage_ids)
    d = len(nonconj_v - phage_ids)

    print("\nViral-carrying Sulfolobaceae 2×2 (Phage_rep × conjugative):")
    print(f"                   Phage_rep+   Phage_rep-")
    print(f"  Conjugative      {a:>9d}    {b:>9d}")
    print(f"  Non-conjugative  {c:>9d}    {d:>9d}")

    OR, p, lo, hi = fisher_exact_with_ci([[a, b], [c, d]])
    print(f"\nFisher: OR = {OR:.2f}  95% CI [{lo:.2f}, {hi:.2f}]  p = {p:.4f}")

    conj_cats = sorted(sulfo_viral_df.loc[
        sulfo_viral_df['replicon'].isin(sulfo_conj), 'new_category'].unique())
    nonconj_cats = sorted(sulfo_viral_df.loc[
        sulfo_viral_df['replicon'].isin(sulfo_nonconj), 'new_category'].unique())
    print(f"\nConjugative carry:     {conj_cats}")
    print(f"Non-conjugative carry: {nonconj_cats}")

    pd.DataFrame([
        {'group': 'sulfolobaceae_total',      'n': len(sulfo_ids)},
        {'group': 'sulfolobaceae_conj',       'n': len(sulfo_conj)},
        {'group': 'sulfolobaceae_nonconj',    'n': len(sulfo_nonconj)},
        {'group': 'sulfolobaceae_with_viral', 'n': len(sulfo_viral_ids)},
    ]).to_csv(OUT_DIR / "09_sulfolobaceae_counts.csv", index=False)

    pd.DataFrame([
        {'row': 'Conjugative',     'phage_rep_pos': a, 'phage_rep_neg': b},
        {'row': 'Non-conjugative', 'phage_rep_pos': c, 'phage_rep_neg': d},
    ]).to_csv(OUT_DIR / "09_sulfolobaceae_contingency.csv", index=False)

    pd.DataFrame([{
        'test': 'fisher_phage_rep_vs_conj_in_sulfolobaceae',
        'OR': OR, 'CI_low': lo, 'CI_high': hi, 'p_value': p,
        'conj_categories': "; ".join(conj_cats),
        'nonconj_categories': "; ".join(nonconj_cats),
    }]).to_csv(OUT_DIR / "09_sulfolobaceae_fisher.csv", index=False)


if __name__ == "__main__":
    main()
