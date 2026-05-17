#!/usr/bin/env python3
"""Conjugative vs non-conjugative: viral presence (Fisher) + depth (MW).

Presence test answers "do VirB4-T4CP+ plasmids carry viral genes more often";
depth test answers "do they carry more categories given they carry any?"
Run globally and within Halobacteriota. Both Fisher and MW are two-sided.
BH-FDR is applied across the four p-values to control multiplicity over the
global × halo × {presence, depth} test family.
"""
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from common import load_data, header, fisher_exact_with_ci, OUT_DIR


def _summarise(comp, label):
    print(f"  {label}: n = {len(comp)}, mean = {comp.mean():.2f}, "
          f"median = {comp.median():.1f}, range = {comp.min()}-{comp.max()}")
    return {
        'group': label, 'n': int(len(comp)),
        'mean_complexity': float(comp.mean()) if len(comp) else 0.0,
        'median_complexity': float(comp.median()) if len(comp) else 0.0,
        'min_complexity': int(comp.min()) if len(comp) else 0,
        'max_complexity': int(comp.max()) if len(comp) else 0,
    }


def main():
    header("VirB4-T4CP × VIRAL CONTENT (BH-FDR across 4 tests)")
    _, mob, _, conj_list, complexity, full_complexity = load_data()

    cs = pd.DataFrame({
        'complexity': full_complexity,
        'conjugative': [pid in conj_list for pid in full_complexity.index],
    })

    print("\n>>> Global")
    cc = cs.loc[cs['conjugative'], 'complexity']
    nc = cs.loc[~cs['conjugative'], 'complexity']
    grp_rows = [
        {'scope': 'global', **_summarise(cc, "Conjugative")},
        {'scope': 'global', **_summarise(nc, "Non-conjugative")},
    ]

    cc_any = (cc > 0).sum()
    nc_any = (nc > 0).sum()
    OR, p, lo, hi = fisher_exact_with_ci(
        [[cc_any, len(cc) - cc_any], [nc_any, len(nc) - nc_any]])
    print(f"\nFisher viral presence (two-sided):")
    print(f"  Conjugative:     {cc_any}/{len(cc)} ({cc_any/len(cc)*100:.1f}%)")
    print(f"  Non-conjugative: {nc_any}/{len(nc)} ({nc_any/len(nc)*100:.1f}%)")
    print(f"  OR = {OR:.2f}  95% CI [{lo:.2f}, {hi:.2f}]  p = {p:.2e}")

    cc_v = cs[(cs['conjugative']) & (cs['complexity'] >= 1)]['complexity']
    nc_v = cs[(~cs['conjugative']) & (cs['complexity'] >= 1)]['complexity']
    U, p_mw = stats.mannwhitneyu(cc_v, nc_v, alternative='two-sided')
    print(f"\nMann-Whitney on complexity (viral-carrying, two-sided): "
          f"U = {U:.0f}, p = {p_mw:.2f}")

    print("\n>>> Within Halobacteriota")
    halo_ids = set(mob.loc[mob['gtdb_phylum'] == 'p__Halobacteriota', 'sample_id'])
    halo_cs = cs[cs.index.isin(halo_ids)]
    cc_h = halo_cs.loc[halo_cs['conjugative'], 'complexity']
    nc_h = halo_cs.loc[~halo_cs['conjugative'], 'complexity']
    grp_rows += [
        {'scope': 'halobacteriota', **_summarise(cc_h, "Conjugative")},
        {'scope': 'halobacteriota', **_summarise(nc_h, "Non-conjugative")},
    ]

    cc_h_any = (cc_h > 0).sum()
    nc_h_any = (nc_h > 0).sum()
    OR_h, p_h, lo_h, hi_h = fisher_exact_with_ci(
        [[cc_h_any, len(cc_h) - cc_h_any], [nc_h_any, len(nc_h) - nc_h_any]])
    print(f"\nFisher viral presence (Halo, two-sided):")
    print(f"  Conjugative:     {cc_h_any}/{len(cc_h)} ({cc_h_any/len(cc_h)*100:.1f}%)")
    print(f"  Non-conjugative: {nc_h_any}/{len(nc_h)} ({nc_h_any/len(nc_h)*100:.1f}%)")
    print(f"  OR = {OR_h:.2f}  95% CI [{lo_h:.2f}, {hi_h:.2f}]  p = {p_h:.2e}")

    cc_hv = halo_cs[(halo_cs['conjugative']) & (halo_cs['complexity'] >= 1)]['complexity']
    nc_hv = halo_cs[(~halo_cs['conjugative']) & (halo_cs['complexity'] >= 1)]['complexity']
    U_h, p_mw_h = stats.mannwhitneyu(cc_hv, nc_hv, alternative='two-sided')
    print(f"\nMann-Whitney (Halo, viral-carrying, two-sided): "
          f"U = {U_h:.0f}, p = {p_mw_h:.2f}")

    pd.DataFrame(grp_rows).to_csv(
        OUT_DIR / "07_conj_complexity_by_group.csv", index=False)

    # BH-FDR across the four tests
    tests = pd.DataFrame([
        {'scope': 'global', 'test': 'fisher_viral_presence',
         'conj_any': int(cc_any), 'conj_n': int(len(cc)),
         'nonconj_any': int(nc_any), 'nonconj_n': int(len(nc)),
         'OR': OR, 'CI_low': lo, 'CI_high': hi, 'p_value': p, 'U': float('nan')},
        {'scope': 'global', 'test': 'mannwhitney_complexity_among_viral',
         'conj_any': int(len(cc_v)), 'conj_n': int(len(cc_v)),
         'nonconj_any': int(len(nc_v)), 'nonconj_n': int(len(nc_v)),
         'OR': float('nan'), 'CI_low': float('nan'), 'CI_high': float('nan'),
         'p_value': p_mw, 'U': U},
        {'scope': 'halobacteriota', 'test': 'fisher_viral_presence',
         'conj_any': int(cc_h_any), 'conj_n': int(len(cc_h)),
         'nonconj_any': int(nc_h_any), 'nonconj_n': int(len(nc_h)),
         'OR': OR_h, 'CI_low': lo_h, 'CI_high': hi_h, 'p_value': p_h, 'U': float('nan')},
        {'scope': 'halobacteriota', 'test': 'mannwhitney_complexity_among_viral',
         'conj_any': int(len(cc_hv)), 'conj_n': int(len(cc_hv)),
         'nonconj_any': int(len(nc_hv)), 'nonconj_n': int(len(nc_hv)),
         'OR': float('nan'), 'CI_low': float('nan'), 'CI_high': float('nan'),
         'p_value': p_mw_h, 'U': U_h},
    ])
    tests['p_BH'] = multipletests(tests['p_value'], method='fdr_bh')[1]
    tests.to_csv(OUT_DIR / "07_conj_viral_tests.csv", index=False)

    print("\nBH-FDR across the four tests:")
    for _, r in tests.iterrows():
        marker = " *" if r['p_BH'] < 0.05 else ""
        print(f"  [{r['scope']:<15}] {r['test']:<35} "
              f"raw p = {r['p_value']:.2e}, BH q = {r['p_BH']:.2e}{marker}")


if __name__ == "__main__":
    main()
