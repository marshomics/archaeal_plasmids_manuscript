#!/usr/bin/env python3
import importlib

for mod in (
    "00_database_assembly",
    "01_taxonomic_distribution",
    "02_abundance_distribution",
    "03_family_level",
    "04_model_organism_bias",
):
    importlib.import_module(mod).main()
