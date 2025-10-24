"""
Shared DataFrame backend for Octavian.

Attempts to load Modin as a drop-in replacement for pandas so large host
workloads can leverage parallel execution. Falls back to pandas when
Modin is unavailable or the user prefers the serial implementation.
"""

from __future__ import annotations

from typing import Tuple

import numpy as _np  # type: ignore
import pandas as _pandas  # type: ignore

# ---------------------------------------------------------------------------
# Numpy compatibility helpers
# ---------------------------------------------------------------------------
# Modin < 0.37 expects numpy to expose NAN/INF aliases that were removed in
# numpy 2.0. Provide shims so we stay forward-compatible without requiring a
# Modin upgrade on every deployment. This is safe because numpy exposes
# lowercase equivalents (nan, inf, etc.) that we simply mirror.
_aliases = (
  ("NAN", lambda: _np.nan),
  ("INF", lambda: _np.inf),
  ("PINF", lambda: _np.inf),
  ("NINF", lambda: -_np.inf),
  ("NZERO", lambda: -0.0),
  ("PZERO", lambda: 0.0),
)
for _name, _factory in _aliases:
  if not hasattr(_np, _name):
    setattr(_np, _name, _factory())  # type: ignore[arg-type]

try:  # pragma: no cover - modin optional
  import modin.pandas as _modin  # type: ignore

  HAS_MODIN = True
except Exception:  # pragma: no cover - fallback path
  _modin = None  # type: ignore
  HAS_MODIN = False

pd = _modin if HAS_MODIN else _pandas  # type: ignore
USING_MODIN: bool = HAS_MODIN


def activate_modin() -> None:
  """Switch the shared DataFrame backend to Modin."""
  if not HAS_MODIN:
    raise ImportError('Modin is not available; install modin to enable the parallel backend.')
  global pd, USING_MODIN
  pd = _modin  # type: ignore[assignment]
  USING_MODIN = True


def deactivate_modin() -> None:
  """Switch the shared DataFrame backend to pandas."""
  global pd, USING_MODIN
  pd = _pandas  # type: ignore[assignment]
  USING_MODIN = False


def backend_info() -> Tuple[str, bool]:
  """
  Return a tuple describing the active dataframe backend.

  Useful for logging or telemetry where we want to know if Modin is
  currently powering the dataframe operations.
  """
  name = "modin.pandas" if USING_MODIN else "pandas"
  return name, USING_MODIN
