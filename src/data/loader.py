"""CSV loading helpers for the TSL dispatch dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - optional dependency guard
    pd = None

DataFrame = Any



def _require_pandas() -> Any:
    if pd is None:
        raise ModuleNotFoundError(
            "pandas is required for CSV loading. Install pandas to use src.data.loader."
        )
    return pd



def _load_csv(path: str | Path) -> DataFrame:
    pandas = _require_pandas()
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {csv_path}")

    try:
        return pandas.read_csv(csv_path)
    except Exception as exc:  # pragma: no cover - defensive error wrapping
        raise ValueError(f"Failed to load CSV file: {csv_path}") from exc



def load_waybill_data(path: str) -> DataFrame:
    """Load the full waybill table."""
    return _load_csv(path)



def load_courier_wave_data(path: str) -> DataFrame:
    """Load courier wave information."""
    return _load_csv(path)



def load_dispatch_waybill(path: str) -> DataFrame:
    """Load dispatch-to-order checkpoint records."""
    return _load_csv(path)



def load_dispatch_rider(path: str) -> DataFrame:
    """Load dispatch-to-rider checkpoint records."""
    return _load_csv(path)
