"""Conversion utilities between Python containers and R objects.
Normalizes inputs to 1D numeric vectors and converts common R scalars to Python.
"""

from __future__ import annotations
import math
from typing import Any, Iterable, Optional
import numpy as np

def _is_pandas_series(obj: Any) -> bool:
    return obj.__class__.__module__.startswith("pandas") and obj.__class__.__name__ == "Series"


def _is_pandas_dataframe(obj: Any) -> bool:
    return obj.__class__.__module__.startswith("pandas") and obj.__class__.__name__ == "DataFrame"


def _is_polars_series(obj: Any) -> bool:
    return obj.__class__.__module__.startswith("polars") and obj.__class__.__name__ == "Series"


def _is_polars_dataframe(obj: Any) -> bool:
    return obj.__class__.__module__.startswith("polars") and obj.__class__.__name__ == "DataFrame"


def as_1d_numeric(x: Any) -> np.ndarray:
    """Convert common Python container types to a 1D float64 NumPy array.

    Accepted inputs: list/tuple, numpy array, pandas Series/DataFrame (1 column),
    polars Series/DataFrame (1 column).

    Missing values: None -> NaN.
    """
    if _is_pandas_series(x):
        arr = x.to_numpy(dtype=float, copy=False)
        return np.asarray(arr, dtype=np.float64)

    if _is_pandas_dataframe(x):
        if x.shape[1] != 1:
            raise ValueError("pandas DataFrame input must have exactly 1 column")
        arr = x.iloc[:, 0].to_numpy(dtype=float, copy=False)
        return np.asarray(arr, dtype=np.float64)

    if _is_polars_series(x):
        arr = x.to_numpy()
        return np.asarray(arr, dtype=np.float64)

    if _is_polars_dataframe(x):
        if x.width != 1:
            raise ValueError("polars DataFrame input must have exactly 1 column")
        arr = x.to_numpy()
        arr = arr.reshape(-1)
        return np.asarray(arr, dtype=np.float64)

    if isinstance(x, np.ndarray):
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim == 0:
            return arr.reshape(1)
        if arr.ndim != 1:
            raise ValueError("NumPy array input must be 1D")
        return arr

    if isinstance(x, (list, tuple)):
        return np.asarray([np.nan if v is None else v for v in x], dtype=np.float64)

    raise TypeError(
        "Unsupported input type. Expected list, numpy array, pandas Series/DataFrame, or polars Series/DataFrame."
    )


def to_r_vector(x_1d: np.ndarray):
    """Convert a 1D float64 NumPy array to an R numeric vector."""
    from .r_bridge import robjects_module

    robjects = robjects_module()
    return robjects.FloatVector(x_1d.tolist())


def r_scalar_to_float(x: Any) -> float:
    """Convert a length-1 R vector/scalar to Python float, mapping NA to NaN."""
    try:
        from rpy2.rinterface_lib.sexp import NARealType
    except Exception:  # pragma: no cover
        NARealType = ()  # type: ignore

    if hasattr(x, "__len__") and len(x) == 1:
        v = x[0]
    else:
        v = x

    if isinstance(v, NARealType):
        return float("nan")

    try:
        fv = float(v)
    except Exception:
        return float("nan")

    if math.isnan(fv):
        return float("nan")

    return fv
