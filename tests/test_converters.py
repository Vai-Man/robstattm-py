import numpy as np
import pytest
from robstattm_py.converters import as_1d_numeric

def test_convert_pandas_series():
    import pandas as pd
    s = pd.Series([1.5, 2.5, np.nan])
    arr = as_1d_numeric(s)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (3,)
    assert arr[0] == 1.5

def test_convert_pandas_dataframe():
    import pandas as pd
    df = pd.DataFrame({"A": [1.5, 2.5, np.nan]})
    arr = as_1d_numeric(df)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (3,)
    assert arr[0] == 1.5

def test_convert_polars_series():
    import polars as pl
    s = pl.Series([1.5, 2.5, None])
    arr = as_1d_numeric(s)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (3,)
    assert arr[0] == 1.5

def test_convert_polars_dataframe():
    import polars as pl
    df = pl.DataFrame({"A": [1.5, 2.5, None]})
    arr = as_1d_numeric(df)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (3,)
    assert arr[0] == 1.5
