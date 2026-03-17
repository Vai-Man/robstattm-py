"""Python wrappers for the RobStatTM R package (via rpy2).
Public API surface: import loc_scale_m and scale_m from here.
"""

from .location_scale import loc_scale_m, scale_m

__all__ = ["loc_scale_m", "scale_m"]
