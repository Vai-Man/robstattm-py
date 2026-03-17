"""R interop bridge: imports RobStatTM and exposes a small, cached call surface.
Keeps rpy2/R package handling and R keyword naming in one place.
"""

from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping, Optional


class RPackageNotInstalledError(RuntimeError):
    pass
class Rpy2NotInstalledError(RuntimeError):
    pass

def _import_rpy2() -> Any:
    try:
        import rpy2  # noqa: F401
        from rpy2 import robjects
        from rpy2.robjects import packages

        return robjects, packages
    except Exception as exc:  # pragma: no cover
        raise Rpy2NotInstalledError(
            "rpy2 is required. Install with `pip install rpy2` and ensure R is on PATH."
        ) from exc


@lru_cache(maxsize=1)
def _robstat_pkg():
    robjects, packages = _import_rpy2()
    if not packages.isinstalled("RobStatTM"):
        raise RPackageNotInstalledError(
            "R package 'RobStatTM' is not installed. In R run: install.packages('RobStatTM')"
        )
    return packages.importr("RobStatTM")


@lru_cache(maxsize=1)
def r():
    robjects, _ = _import_rpy2()
    return robjects.r


@lru_cache(maxsize=1)
def robjects_module():
    robjects, _ = _import_rpy2()
    return robjects


def robstattm() -> Any:
    return _robstat_pkg()


@dataclass(frozen=True)
class RCallResult:
    value: Any


def call_robstat_function(name: str, *args: Any, **kwargs: Any) -> Any:
    pkg = robstattm()
    fn = getattr(pkg, name)
    return fn(*args, **kwargs)


def to_r_kwargs(mapping: Mapping[str, Any], *, rename: Optional[Mapping[str, str]] = None) -> dict[str, Any]:
    rename = rename or {}
    out: dict[str, Any] = {}
    for k, v in mapping.items():
        rk = rename.get(k, k)
        if v is None:
            continue
        out[rk] = v
    return out
