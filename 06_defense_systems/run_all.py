#!/usr/bin/env python3
import importlib

for mod in (
    "01_summary_statistics",
    "02_phylum_distribution",
    "03_cooccurrence_weighted_perm",
    "04_defense_vs_size",
    "05_virb4_defense_associations",
    "06_crispr_spacer_summary",
    "07_crispr_targeting_enrichment",
    "08_crispr_within_family",
    "09_crispr_virb4_fisher",
):
    importlib.import_module(mod).main()
