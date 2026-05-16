# Archaeal plasmid analyses

Reproducible analysis pipelines for the six Results subsections of an archaeal-plasmid manuscript. Each subdirectory is a self-contained pipeline that reads input tables from its own `data/` folder, runs the cited statistical tests, and writes tables to `outputs/`. Plot rendering is out of scope.

## Pipelines

| Folder | Subsection |
|---|---|
| `01_prevalence_abundance/` | Database assembly, carrier prevalence by phylum and class, per-genome abundance, model-organism contribution |
| `02_sampling_bias/` | Detection vs assembly quality and sequencing depth; phylum enrichment after correction |
| `03_plasmid_proteins/` | 6.4M-protein archaea-vs-bacteria clustering; cross-domain protein sharing; intra-archaea functional repertoire |
| `04_conjugation/` | VirB4–T4CP co-occurrence, proximity, subtype clustering, core-gene enrichment |
| `05_viral_signatures/` | Viral protein prevalence, functional hierarchy, complexity, conjugation × viral overlap |
| `06_defense_systems/` | Defence-system carriage, co-occurrence, size scaling, VirB4–T4CP × defence, CRISPR-spacer targeting |

## Layout convention

Each pipeline follows the same shape:

```
streamlined_<name>/
├── common.py                shared loaders, paths, methodological constants
├── 01_<step>.py             one script per Results claim (or pair of claims)
├── 02_<step>.py
├── ...
├── run_all.py               runs every step in order
├── data/                    input files
└── outputs                  output files
```

Run a single step with `python 03_<step>.py` from inside its pipeline folder, or run the whole pipeline with `python run_all.py`. Outputs land in `outputs/` next to the scripts.

## Inputs

Every script expects input files under its pipeline's `data/` folder; the expected file names and column schemas are documented in each pipeline's `common.py` docstring. No input files are shipped with the code. Where a pipeline relies on a methodological list (e.g. reference-strain species, replicons to exclude, regex labels), that list is loaded from a small config file in `data/` rather than baked into source code, so users can substitute their own definitions without editing scripts.

## Dependencies

Common across pipelines: `pandas`, `numpy`, `scipy`, `statsmodels`. A few pipelines additionally require `scikit-posthocs` (viral signatures, defence-systems CRISPR step) and `scikit-learn` (sampling-bias logistic regression). Per-pipeline install commands appear in each `README.md`.

```bash
pip install pandas numpy scipy statsmodels scikit-learn scikit-posthocs
```

