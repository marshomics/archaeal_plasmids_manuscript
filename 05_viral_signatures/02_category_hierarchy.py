#!/usr/bin/env python3
"""Per-category prevalence and protein counts among viral-carrying plasmids."""
import pandas as pd
from common import load_data, header, OUT_DIR


def main():
    header("VIRAL CATEGORY HIERARCHY")
    df, *_ = load_data()
    n_viral = df['replicon'].nunique()
    all_cats = sorted(df['new_category'].unique())

    rows = []
    print(f"Among {n_viral} viral-carrying plasmids:")
    for cat in all_cats:
        n = df[df['new_category'] == cat]['replicon'].nunique()
        n_prot = int((df['new_category'] == cat).sum())
        rows.append({
            'category': cat,
            'n_plasmids': n,
            'pct_plasmids': n / n_viral * 100,
            'n_proteins': n_prot,
        })
        print(f"  {cat:<24} {n:>4} plasmids ({n/n_viral*100:.1f}%)")

    print(f"\nProtein-level counts:")
    print(f"  total: {len(df)}")
    for r in rows:
        print(f"  {r['category']:<24} {r['n_proteins']:>5} proteins")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "02_category_hierarchy.csv", index=False)


if __name__ == "__main__":
    main()
