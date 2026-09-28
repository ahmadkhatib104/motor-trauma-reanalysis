# MOTOR trial reanalysis: reproducible code

This repository contains the analysis and independent numerical checks for a secondary analysis of the publicly released MOTOR trial dataset. No participant-level data are included.

This repository is a release snapshot. The analysis was completed before it was placed under version control, so the commit history dates from publication rather than from the work itself.

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

The scripts produce aggregate analysis results in `outputs/motor/`, tables in `tables/`, figures in `figures/`, and audit files in `audit/`. The downloaded source data under `data/raw/` are ignored by Git.

The analysis uses recorded end-of-follow-up vital status in the released cohort, not verified 90-day survival. Its six-hospital allocation reference is exact for the recorded hospital summaries, while the reported confidence intervals are working-model approximations. These analyses do not remove possible post-randomization selection bias or missing-outcome uncertainty.

The original investigators' dataset and code are credited through the Mendeley Data record.

## Citation

Ahmad Khatib, Fawzi Mualla, Mohammad Yaaseen, Saleh Awad, Zayan Syed, Moustafa Hazin.
*Recorded mortality after rural trauma-team training in MOTOR: a hospital-level reanalysis.*
Creighton University School of Medicine, Phoenix, Arizona, USA.

Machine-readable metadata is in `CITATION.cff`.

## License

Released under the MIT License. See `LICENSE`.
