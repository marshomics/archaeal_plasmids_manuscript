#!/usr/bin/env python3
import importlib

for mod in (
    "generate_fisher_enrichment",
    "01_viral_prevalence",
    "02_category_hierarchy",
    "03_complexity_distribution",
    "04_phylum_pairwise",
    "05_family_enrichment",
    "06_non_halobacteriota",
    "07_conjugative_viral_overlap",
    "08_subtype_viral_composition",
    "09_sulfolobaceae_segregation",
):
    importlib.import_module(mod).main()
