#!/usr/bin/env python3
import importlib

for mod in (
    "01_detection_by_assembly_quality",
    "02_detection_by_sequencing_depth",
    "03_phylum_depth_stratified",
    "04_multivariate_logistic",
    "05_depth_matched_permutation",
    "06_quality_subset_robustness",
    "07_undersampled_phyla_power",
):
    importlib.import_module(mod).main()
