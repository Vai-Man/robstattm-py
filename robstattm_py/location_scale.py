"""Public wrappers for robust univariate location/scale estimators.
Implements thin rpy2 wrappers for RobStatTM::locScaleM and RobStatTM::scaleM.
"""

from __future__ import annotations
from typing import Any, Dict, Optional
import numpy as np
from .converters import as_1d_numeric, r_scalar_to_float, to_r_vector
from .r_bridge import call_robstat_function, to_r_kwargs


def loc_scale_m(
    x: Any,
    *,
    psi: str = "mopt",
    eff: float = 0.95,
    maxit: int = 50,
    tol: float = 1e-4,
    na_rm: bool = False,
) -> Dict[str, float]:
    """Robust univariate location and scale M-estimators (R: RobStatTM::locScaleM).

    Parameters
    ----------
    x : array-like
        Univariate observations. Accepts Python lists/tuples, NumPy 1D arrays,
        pandas Series/DataFrame (single column), or polars Series/DataFrame (single column).
    psi : {"bisquare", "huber", "opt", "mopt"}, default "mopt"
        Score function.
    eff : float, default 0.95
        Desired asymptotic efficiency (allowed values depend on `psi`).
    maxit : int, default 50
        Maximum number of iterations.
    tol : float, default 1e-4
        Convergence tolerance.
    na_rm : bool, default False
        Whether to remove missing values before computation.

    Returns
    -------
    result : dict
        Dictionary with keys:

        - ``mu``: robust location estimate
        - ``std_mu``: estimated standard deviation of ``mu``
        - ``disper``: robust M-scale (dispersion)

    Notes
    -----
    This is a thin wrapper around the R function ``RobStatTM::locScaleM``.
    Numerical results should match R (up to floating tolerance) because the
    computation is performed in R.

    Examples
    --------
    >>> import numpy as np
    >>> from robstattm_py import loc_scale_m
    >>> x = np.array([1, 2, 2, 3, 100.0])
    >>> loc_scale_m(x, psi="mopt")
    {'mu': ..., 'std_mu': ..., 'disper': ...}
    """
    x_arr = as_1d_numeric(x)
    r_x = to_r_vector(x_arr)

    r_kwargs = to_r_kwargs(
        {
            "psi": psi,
            "eff": eff,
            "maxit": maxit,
            "tol": tol,
            "na_rm": na_rm,
        },
        rename={"na_rm": "na.rm"},
    )

    res = call_robstat_function("locScaleM", r_x, **r_kwargs)

    return {
        "mu": r_scalar_to_float(res.rx2("mu")),
        "std_mu": r_scalar_to_float(res.rx2("std.mu")),
        "disper": r_scalar_to_float(res.rx2("disper")),
    }


def scale_m(
    u: Any,
    *,
    delta: float = 0.5,
    family: str = "bisquare",
    max_it: int = 100,
    tol: float = 1e-6,
    tolerancezero: Optional[float] = None,
    tuning_chi: Optional[Any] = None,
) -> float:
    """Robust M-scale estimator (R: RobStatTM::scaleM).

    Parameters
    ----------
    u : array-like
        Residuals / univariate observations.
    delta : float, default 0.5
        Right-hand side of the M-scale equation.
    family : {"bisquare", "opt", "mopt"}, default "bisquare"
        Loss function family.
    max_it : int, default 100
        Maximum iterations.
    tol : float, default 1e-6
        Relative tolerance.
    tolerancezero : float, optional
        Smallest non-zero scale accepted. If omitted, R's default (.Machine$double.eps) is used.
    tuning_chi : optional
        Tuning object; if omitted, R computes the default consistent tuning for the chosen
        `family` and `delta`.

    Returns
    -------
    scale : float
        The robust scale estimate.

    Notes
    -----
    Thin wrapper around ``RobStatTM::scaleM``.

    Examples
    --------
    >>> import numpy as np
    >>> from robstattm_py import scale_m
    >>> u = np.array([1, 2, 2, 3, 100.0])
    >>> scale_m(u)
    ...
    """
    u_arr = as_1d_numeric(u)
    r_u = to_r_vector(u_arr)

    r_kwargs = to_r_kwargs(
        {
            "delta": delta,
            "family": family,
            "max_it": max_it,
            "tol": tol,
            "tolerancezero": tolerancezero,
            "tuning_chi": tuning_chi,
        },
        rename={
            "max_it": "max.it",
            "tuning_chi": "tuning.chi",
        },
    )

    res = call_robstat_function("scaleM", r_u, **r_kwargs)
    return r_scalar_to_float(res)
