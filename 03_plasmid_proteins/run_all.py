#!/usr/bin/env python3
import importlib

for mod in (
    "00_build_cluster_summary",
    "01_cross_domain_overview",
    "02_archaeal_sharing_per_phylum",
    "03_balanced_subsampling",
    "04_functional_enrichment_crossdomain",
    "05_partner_residuals",
    "06_archaea_only_clr_enrichment",
    "07_unannotated_protein_analysis",
):
    importlib.import_module(mod).main()
