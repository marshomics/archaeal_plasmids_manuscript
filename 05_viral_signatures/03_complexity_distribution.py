#!/usr/bin/env python3
"""Distribution of distinct viral functional categories per plasmid.

Also reports the Spearman correlation between distinct category count and
total viral protein count — a check that 'complexity' isn't just a proxy
for protein count.
"""
from scipy import stats

from common import load_data, header


def main():
    header("COMPLEXITY DISTRIBUTION")
    df, _, _, _, complexity, _ = load_data()
    n_viral = complexity.size

    print("Distinct viral categories per plasmid:")
    for level in range(1, complexity.max() + 1):
        n = (complexity == level).sum()
        if n > 0:
            print(f"  {level:>2}: {n:>4} plasmids ({n/n_viral*100:.1f}%)")
    c1 = (complexity == 1).sum()
    c5 = (complexity >= 5).sum()
    print(f"\nComplexity = 1: {c1} ({c1/n_viral*100:.1f}%)")
    print(f"Complexity ≥ 5: {c5} ({c5/n_viral*100:.1f}%)")

    c1_ids = complexity[complexity == 1].index
    c1_data = df[df['replicon'].isin(c1_ids)]
    c1_cats = c1_data.groupby('replicon')['new_category'].first().value_counts()
    print(f"\nComplexity-1 plasmids by sole category:")
    for cat, n in c1_cats.items():
        print(f"  {cat:<24} {n:>4} ({n/len(c1_ids)*100:.1f}%)")

    # is category count just tracking protein count?
    proteins_per = df.groupby('replicon')['protein'].nunique()
    paired = complexity.to_frame('cats').join(
        proteins_per.rename('proteins'), how='inner')
    rho, p = stats.spearmanr(paired['cats'], paired['proteins'])
    print(f"\nSpearman (categories vs protein count): "
          f"ρ = {rho:.2f}, p = {p:.2e}, n = {len(paired)}")


if __name__ == "__main__":
    main()
