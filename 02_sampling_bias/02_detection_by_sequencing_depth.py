#!/usr/bin/env python3
"""Detection vs depth bin; sampling asymmetry between carriers and non-carriers."""
from scipy import stats

from common import load_data, header, DEPTH_BINS


def main():
    reps, _, _ = load_data()
    header("DETECTION vs SAMPLING DEPTH")

    n_total = len(reps)
    n_carriers = int(reps['is_carrier'].sum())
    print(f"Reps:     {n_total}")
    print(f"Carriers: {n_carriers} ({n_carriers/n_total*100:.1f}%)")

    print("\nCarrier rate by depth bin:")
    for low, high, label in DEPTH_BINS:
        sub = reps[(reps['n_genomes'] >= low) & (reps['n_genomes'] <= high)]
        nc = int(sub['is_carrier'].sum())
        n = len(sub)
        print(f"  {label:>5}: {nc}/{n} ({nc/n*100:.1f}%)")

    carriers = reps[reps['is_carrier'] == 1]
    noncarriers = reps[reps['is_carrier'] == 0]

    # Fisher on the ≥ 2-genome cut-off (sampling asymmetry)
    c_multi = int((carriers['n_genomes'] >= 2).sum())
    n_multi = int((noncarriers['n_genomes'] >= 2).sum())
    OR, p = stats.fisher_exact([
        [c_multi, len(carriers) - c_multi],
        [n_multi, len(noncarriers) - n_multi],
    ])
    print("\nSpecies with ≥ 2 genomes:")
    print(f"  Carriers:     {c_multi}/{len(carriers)} ({c_multi/len(carriers)*100:.1f}%)")
    print(f"  Non-carriers: {n_multi}/{len(noncarriers)} "
          f"({n_multi/len(noncarriers)*100:.1f}%)")
    print(f"  Fisher OR = {OR:.2f}, p = {p:.2e}")

    # Mann-Whitney on the raw depth distributions — same question, no bin cut
    U, p_mw = stats.mannwhitneyu(
        carriers['n_genomes'], noncarriers['n_genomes'],
        alternative='two-sided')
    print("\nMann-Whitney U on n_genomes (carriers vs non-carriers):")
    print(f"  Carrier median:     {carriers['n_genomes'].median():.0f}")
    print(f"  Non-carrier median: {noncarriers['n_genomes'].median():.0f}")
    print(f"  U = {U:.0f}, p = {p_mw:.2e}")


if __name__ == "__main__":
    main()
