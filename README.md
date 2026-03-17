# robStatTM (Python wrappers for RobStatTM)

This repo contains Python wrappers for the **[RobStatTM](https://cran.r-project.org/web/packages/RobStatTM/index.html)** CRAN package using **rpy2**.
The core idea is simple: call the original R implementations (RobStatTM::locScaleM / RobStatTM::scaleM) from Python, so numerical results match R.

## What’s included

- Python wrappers:
  - `robstattm_py.loc_scale_m` → wraps `RobStatTM::locScaleM`
  - `robstattm_py.scale_m` → wraps `RobStatTM::scaleM`
- Conversion utilities (list/NumPy/pandas/polars → R numeric vector)
- Validation scripts (R vs Python)
- Pytest suite that compares wrapper outputs to direct R calls

## Requirements

- R installed and on PATH
- R package RobStatTM installed:

```r
install.packages("RobStatTM")
```

- Python dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

## Pre-check tests

### 1) Validate locScaleM and scaleM match R results

Run the R examples (tests both locScaleM and scaleM):

```bash
Rscript validation/validate.R
```

Run the Python vs R comparison (prints diffs; they should be exactly 0 for matching implementations):

```bash
python validation/validate_r_vs_python.py
```

### 2) scaleM wrapper: match R results + relationship explanation

`locScaleM(x)` computes **both** a robust location estimate and a robust scale estimate (dispersion):
- output components include `mu` (location), `std.mu`, and `disper` (scale)

`scaleM(u)` computes **only** a robust scale.
Conceptually:

- `locScaleM = (location, scale)`
- `scaleM = scale-only`

In other words, `scaleM` corresponds to the “scale part” of what you typically want from a robust location+scale procedure. (Numerical equality between `locScaleM(...).disper` and `scaleM(...)` is not guaranteed unless tuning/family choices align; both estimate robust spread.)

## Run tests

```bash
python -m pytest -q
```

Tests automatically skip if rpy2/RobStatTM are not available.

## Notes for extending to other RobStatTM functions

To add another wrapper:

1. Convert Python inputs in `robstattm_py/converters.py` (usually to a 1D vector or 2D matrix).
2. Call the R function via `robstattm_py/r_bridge.py::call_robstat_function`.
3. Convert the R return value to Python-native outputs.
4. Add a pytest that compares to a direct R call (same inputs).
