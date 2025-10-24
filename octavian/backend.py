"""
Shared DataFrame backend for Octavian.

Attempts to load Modin as a drop-in replacement for pandas so large host
workloads can leverage parallel execution. Falls back to pandas when
Modin is unavailable or the user prefers the serial implementation.
"""

from __future__ import annotations

from typing import Tuple

import pandas as _pandas  # type: ignore

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
