"""Reproducible R-vs-Python validation for locScaleM/scaleM wrappers.
Prints component-wise differences (should be exactly 0 for the same R inputs).
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from robstattm_py import loc_scale_m, scale_m


def _print_loc_diff(label: str, py: dict[str, float], r: dict[str, float]) -> None:
    print(f"\n{label}")
    for k in ["mu", "std_mu", "disper"]:
        d = py[k] - r[k]
        print(f"  {k:7s}  py={py[k]: .12g}  r={r[k]: .12g}  diff={d: .3g}")


def main() -> None:
    # Same concrete vectors in both languages
    x1 = np.array([1, 2, 2, 3, 100.0], dtype=float)

    # Compute via wrappers
    py_loc_1 = loc_scale_m(x1)
    py_scale_1 = scale_m(x1)

    # Compute via direct R calls (for comparison)
    from robstattm_py.converters import to_r_vector, r_scalar_to_float
    from robstattm_py.r_bridge import call_robstat_function

    r_x1 = to_r_vector(x1)
    r_loc_1 = call_robstat_function("locScaleM", r_x1)
    r_loc_1_py = {
        "mu": r_scalar_to_float(r_loc_1.rx2("mu")),
        "std_mu": r_scalar_to_float(r_loc_1.rx2("std.mu")),
        "disper": r_scalar_to_float(r_loc_1.rx2("disper")),
    }
    r_scale_1 = r_scalar_to_float(call_robstat_function("scaleM", r_x1))

    print("locScaleM(x1):")
    _print_loc_diff("x1", py_loc_1, r_loc_1_py)

    print("\nscaleM(x1):")
    print(f"  py={py_scale_1:.12g}  r={r_scale_1:.12g}  diff={py_scale_1 - r_scale_1:.3g}")

    # Examples matching RobStatTM documentation: generate the exact same samples using R RNG.
    from robstattm_py.r_bridge import r as R

    R()("set.seed(123)")
    r_vec = np.asarray(list(R()("rnorm(150, sd=1.5)")), dtype=float)
    py_loc = loc_scale_m(r_vec)
    py_s = scale_m(r_vec)

    r_r = to_r_vector(r_vec)
    r_loc = call_robstat_function("locScaleM", r_r)
    r_loc_py = {
        "mu": r_scalar_to_float(r_loc.rx2("mu")),
        "std_mu": r_scalar_to_float(r_loc.rx2("std.mu")),
        "disper": r_scalar_to_float(r_loc.rx2("disper")),
    }
    r_s = r_scalar_to_float(call_robstat_function("scaleM", r_r))

    _print_loc_diff("N(0,1.5) n=150", py_loc, r_loc_py)
    print("\nscaleM(N(0,1.5)):")
    print(f"  py={py_s:.12g}  r={r_s:.12g}  diff={py_s - r_s:.3g}")

    R()("set.seed(123)")
    r2_vec = np.asarray(list(R()("c(rnorm(135, sd=1.5), rnorm(15, mean=-10, sd=.5))")), dtype=float)
    py_loc2 = loc_scale_m(r2_vec)
    r_r2 = to_r_vector(r2_vec)
    r_loc2 = call_robstat_function("locScaleM", r_r2)
    r_loc2_py = {
        "mu": r_scalar_to_float(r_loc2.rx2("mu")),
        "std_mu": r_scalar_to_float(r_loc2.rx2("std.mu")),
        "disper": r_scalar_to_float(r_loc2.rx2("disper")),
    }
    _print_loc_diff("10% outliers (mean=-10)", py_loc2, r_loc2_py)

    R()("set.seed(123)")
    r3_vec = np.asarray(list(R()("c(rnorm(135, sd=1.5), rnorm(15, mean=-5, sd=.5))")), dtype=float)
    py_s3 = scale_m(r3_vec, family="opt")
    r_r3 = to_r_vector(r3_vec)
    r_s3 = r_scalar_to_float(call_robstat_function("scaleM", r_r3, family="opt"))
    print("\nscaleM(10% outliers, family='opt'):")
    print(f"  py={py_s3:.12g}  r={r_s3:.12g}  diff={py_s3 - r_s3:.3g}")


if __name__ == "__main__":
    main()
