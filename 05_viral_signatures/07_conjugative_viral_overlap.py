#!/usr/bin/env python3
"""Conjugative vs non-conjugative: viral presence (Fisher) + depth (MW).

Presence test answers "do VirB4-T4CP+ plasmids carry viral genes more often";
depth test answers "do they carry more categories given they carry any?"
Run globally and within Halobacteriota.
"""
import pandas as pd
from scipy import stats

from common import load_data, header, fisher_exact_with_ci


def _summarise(comp, label):
    print(f"  {label}: n = {len(comp)}, mean = {comp.mean():.2f}, "
          f"median = {comp.median():.1f}, range = {comp.min()}-{comp.max()}")


def main():
    header("VirB4-T4CP × VIRAL CONTENT")
    _, mob, _, conj_list, complexity, full_complexity = load_data()

    cs = pd.DataFrame({
        'complexity': full_complexity,
        'conjugative': [pid in conj_list for pid in full_complexity.index],
    })

    print("\n>>> Global")
    cc = cs.loc[cs['conjugative'], 'complexity']
    nc = cs.loc[~cs['conjugative'], 'complexity']
    _summarise(cc, "Conjugative")
    _summarise(nc, "Non-conjugative")

    cc_any = (cc > 0).sum()
    nc_any = (nc > 0).sum()
    OR, p, lo, hi = fisher_exact_with_ci(
        [[cc_any, len(cc) - cc_any], [nc_any, len(nc) - nc_any]])
    print(f"\nFisher viral presence:")
    print(f"  Conjugative:     {cc_any}/{len(cc)} ({cc_any/len(cc)*100:.1f}%)")
    print(f"  Non-conjugative: {nc_any}/{len(nc)} ({nc_any/len(nc)*100:.1f}%)")
    print(f"  OR = {OR:.2f}  95% CI [{lo:.2f}, {hi:.2f}]  p = {p:.2e}")

    cc_v = cs[(cs['conjugative']) & (cs['complexity'] >= 1)]['complexity']
    nc_v = cs[(~cs['conjugative']) & (cs['complexity'] >= 1)]['complexity']
    U, p_mw = stats.mannwhitneyu(cc_v, nc_v, alternative='two-sided')
    print(f"\nMann-Whitney on complexity (viral-carrying): "
          f"U = {U:.0f}, p = {p_mw:.2f}")

    print("\n>>> Within Halobacteriota")
    halo_ids = set(mob.loc[mob['gtdb_phylum'] == 'p__Halobacteriota', 'sample_id'])
    halo_cs = cs[cs.index.isin(halo_ids)]
    cc_h = halo_cs.loc[halo_cs['conjugative'], 'complexity']
    nc_h = halo_cs.loc[~halo_cs['conjugative'], 'complexity']
    _summarise(cc_h, "Conjugative")
    _summarise(nc_h, "Non-conjugative")

    cc_h_any = (cc_h > 0).sum()
    nc_h_any = (nc_h > 0).sum()
    OR_h, p_h, lo_h, hi_h = fisher_exact_with_ci(
        [[cc_h_any, len(cc_h) - cc_h_any], [nc_h_any, len(nc_h) - nc_h_any]])
    print(f"\nFisher viral presence (Halo):")
    print(f"  Conjugative:     {cc_h_any}/{len(cc_h)} ({cc_h_any/len(cc_h)*100:.1f}%)")
    print(f"  Non-conjugative: {nc_h_any}/{len(nc_h)} ({nc_h_any/len(nc_h)*100:.1f}%)")
    print(f"  OR = {OR_h:.2f}  95% CI [{lo_h:.2f}, {hi_h:.2f}]  p = {p_h:.2e}")

    cc_hv = halo_cs[(halo_cs['conjugative']) & (halo_cs['complexity'] >= 1)]['complexity']
    nc_hv = halo_cs[(~halo_cs['conjugative']) & (halo_cs['complexity'] >= 1)]['complexity']
    U_h, p_mw_h = stats.mannwhitneyu(cc_hv, nc_hv, alternative='two-sided')
    print(f"\nMann-Whitney (Halo, viral-carrying): "
          f"U = {U_h:.0f}, p = {p_mw_h:.2f}")


if __name__ == "__main__":
    main()
