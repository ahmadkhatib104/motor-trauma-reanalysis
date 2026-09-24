# MOTOR trial reanalysis: reproducible code

This repository contains the analysis and independent numerical checks for a secondary analysis of the publicly released MOTOR trial dataset. It contains **no participant-level data** and no unpublished author or review documents.

The source dataset and data dictionary are available from [Mendeley Data, version 1](https://doi.org/10.17632/bgpmkpcwdt.1) under CC BY 4.0. The acquisition script checks the downloaded files against the SHA-256 values in `data/manifest.json` and stops if the dataset differs. The raw dataset is downloaded to `data/raw/motor/` locally and is not tracked in this repository.

## Reproduce

Tested with Python 3.12. Install the pinned dependencies:

```sh
python -m venv .venv
python -m pip install -r requirements.txt
python src/acquire_motor.py
python src/motor_analysis.py
python tests/verify_motor_independent.py
python src/make_outputs.py
```

The scripts produce aggregate analysis results in `outputs/motor/`, tables in `tables/`, figures in `figures/`, and audit files in `audit/`. They also create a local participant-level intermediate file under `data/processed/`; **do not publish that file**. Both `data/raw/` and `data/processed/` are ignored by Git.

The analysis uses recorded end-of-follow-up vital status in the released cohort, not verified 90-day survival. Its six-hospital allocation reference is exact for the recorded hospital summaries, while the reported confidence intervals are working-model approximations. These analyses do not remove possible post-randomization selection bias or missing-outcome uncertainty.

The original investigators' dataset and code are credited through the Mendeley Data record. This repository is the code for the secondary analysis, not the original trial's analysis code.

No reuse license has yet been assigned to this repository's code. Public access does not itself grant a license to redistribute or adapt it.
