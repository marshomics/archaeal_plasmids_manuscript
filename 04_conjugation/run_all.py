#!/usr/bin/env python3
import importlib

for mod in (
    "01_virb4_t4cp_cooccurrence",
    "02_proximity_analysis",
    "03_taxonomy_breakdown",
    "04_subtype_clustering",
    "05_core_gene_enrichment",
    "06_dual_metric_topology",
):
    importlib.import_module(mod).main()
