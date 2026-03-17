#Tests for loc_scale_m and scale_m that compare outputs directly to R.

from __future__ import annotations
import numpy as np
import pytest
from robstattm_py import loc_scale_m, scale_m
from robstattm_py.converters import r_scalar_to_float, to_r_vector
from robstattm_py.r_bridge import call_robstat_function

pytestmark = pytest.mark.usefixtures("require_r")

def test_loc_scale_m_simple_vector_matches_r() -> None:
    x = np.array([1, 2, 2, 3, 100.0], dtype=float)

    py = loc_scale_m(x)

    r_x = to_r_vector(x)
    r_res = call_robstat_function("locScaleM", r_x)
    r_py = {
        "mu": r_scalar_to_float(r_res.rx2("mu")),
        "std_mu": r_scalar_to_float(r_res.rx2("std.mu")),
        "disper": r_scalar_to_float(r_res.rx2("disper")),
    }

    assert set(py.keys()) == {"mu", "std_mu", "disper"}
    for k in ["mu", "std_mu", "disper"]:
        assert np.isfinite(py[k]) or np.isnan(py[k])
        assert np.allclose(py[k], r_py[k], rtol=0, atol=0)

def test_scale_m_simple_vector_matches_r() -> None:
    x = np.array([1, 2, 2, 3, 100.0], dtype=float)

    py = scale_m(x)

    r_x = to_r_vector(x)
    r_val = r_scalar_to_float(call_robstat_function("scaleM", r_x))

    assert np.allclose(py, r_val, rtol=0, atol=0)

def test_outliers_robustness_examples() -> None:
    rng = np.random.default_rng(2026)
    good = rng.normal(0.0, 1.5, size=135)
    out = rng.normal(-10.0, 0.5, size=15)
    x = np.concatenate([good, out])

    py = loc_scale_m(x, psi="mopt", eff=0.95)

    r_x = to_r_vector(x)
    r_res = call_robstat_function("locScaleM", r_x, psi="mopt", eff=0.95)
    r_py = {
        "mu": r_scalar_to_float(r_res.rx2("mu")),
        "std_mu": r_scalar_to_float(r_res.rx2("std.mu")),
        "disper": r_scalar_to_float(r_res.rx2("disper")),
    }

    for k in ["mu", "std_mu", "disper"]:
        assert np.allclose(py[k], r_py[k], rtol=0, atol=0)

def test_missing_values_na_rm() -> None:
    x = np.array([1.0, np.nan, 2.0, 3.0], dtype=float)

    py = loc_scale_m(x, na_rm=True)

    r_x = to_r_vector(x)
    r_res = call_robstat_function("locScaleM", r_x, **{"na.rm": True})
    r_py = {
        "mu": r_scalar_to_float(r_res.rx2("mu")),
        "std_mu": r_scalar_to_float(r_res.rx2("std.mu")),
        "disper": r_scalar_to_float(r_res.rx2("disper")),
    }

    for k in ["mu", "std_mu", "disper"]:
        assert np.allclose(py[k], r_py[k], rtol=0, atol=0)
