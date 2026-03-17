#Pytest configuration and environment checks for rpy2 + RobStatTM.

from __future__ import annotations
import pytest

def _r_available() -> bool:
    try:
        import rpy2  # noqa: F401
        from rpy2.robjects import packages

        return packages.isinstalled("RobStatTM")
    except Exception:
        return False


@pytest.fixture(scope="session")
def require_r() -> None:
    if not _r_available():
        pytest.skip("Requires rpy2 + R package RobStatTM")
