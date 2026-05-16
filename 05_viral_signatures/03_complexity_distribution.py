#!/usr/bin/env python3
"""Distribution of distinct viral functional categories per plasmid.

Also reports the Spearman correlation between distinct category count and
total viral protein count — a check that 'complexity' isn't just a proxy
for protein count.
"""
import pandas as pd
from scipy import stats

from common import load_data, header, OUT_DIR


def main():
    header("COMPLEXITY DISTRIBUTION")
    df, _, _, _, complexity, _ = load_data()
    n_viral = complexity.size

    print("Distinct viral categories per plasmid:")
    dist_rows = []
    for level in range(1, complexity.max() + 1):
        n = int((complexity == level).sum())
        if n > 0:
            print(f"  {level:>2}: {n:>4} plasmids ({n/n_viral*100:.1f}%)")
        dist_rows.append({'complexity_level': level, 'n_plasmids': n,
                          'pct_plasmids': n / n_viral * 100 if n_viral else 0.0})
    pd.DataFrame(dist_rows).to_csv(
        OUT_DIR / "03_complexity_distribution.csv", index=False)

    c1 = (complexity == 1).sum()
    c5 = (complexity >= 5).sum()
    print(f"\nComplexity = 1: {c1} ({c1/n_viral*100:.1f}%)")
    print(f"Complexity ≥ 5: {c5} ({c5/n_viral*100:.1f}%)")

    c1_ids = complexity[complexity == 1].index
    c1_data = df[df['replicon'].isin(c1_ids)]
    c1_cats = c1_data.groupby('replicon')['new_category'].first().value_counts()
    print(f"\nComplexity-1 plasmids by sole category:")
    c1_rows = []
    for cat, n in c1_cats.items():
        print(f"  {cat:<24} {n:>4} ({n/len(c1_ids)*100:.1f}%)")
        c1_rows.append({'sole_category': cat, 'n_plasmids': int(n),
                        'pct_of_complexity1': n / len(c1_ids) * 100})
    pd.DataFrame(c1_rows).to_csv(
        OUT_DIR / "03_complexity1_sole_category.csv", index=False)

    # per-plasmid complexity + protein count
    proteins_per = df.groupby('replicon')['protein'].nunique()
    paired = complexity.to_frame('cats').join(
        proteins_per.rename('proteins'), how='inner')
    paired.reset_index().rename(columns={'index': 'replicon'}).to_csv(
        OUT_DIR / "03_complexity_per_plasmid.csv", index=False)

    rho, p = stats.spearmanr(paired['cats'], paired['proteins'])
    print(f"\nSpearman (categories vs protein count): "
          f"ρ = {rho:.2f}, p = {p:.2e}, n = {len(paired)}")

    pd.DataFrame([{
        'test': 'spearman_categories_vs_protein_count',
        'rho': rho, 'p_value': p, 'n': len(paired),
    }]).to_csv(OUT_DIR / "03_complexity_protein_correlation.csv", index=False)


if __name__ == "__main__":
    main()
