#!/usr/bin/env python3
"""Per-category prevalence and protein counts among viral-carrying plasmids."""
from common import load_data, header


def main():
    header("VIRAL CATEGORY HIERARCHY")
    df, *_ = load_data()
    n_viral = df['replicon'].nunique()
    all_cats = sorted(df['new_category'].unique())

    print(f"Among {n_viral} viral-carrying plasmids:")
    for cat in all_cats:
        n = df[df['new_category'] == cat]['replicon'].nunique()
        print(f"  {cat:<24} {n:>4} plasmids ({n/n_viral*100:.1f}%)")

    print(f"\nProtein-level counts:")
    print(f"  total: {len(df)}")
    for cat in all_cats:
        n_prot = (df['new_category'] == cat).sum()
        print(f"  {cat:<24} {n_prot:>5} proteins")


if __name__ == "__main__":
    main()
