"""Route stop representation for simplified route insertion."""

from __future__ import annotations

from dataclasses import dataclass

from src.state.order import normalize_coordinate

VALID_STOP_TYPES = {"pickup", "dropoff"}


@dataclass(slots=True)
class RouteStop:
    """Single stop on a courier route."""

    order_id: str
    stop_type: str
    lat: float
    lng: float
    earliest_time: int | None = None
    latest_time: int | None = None

    def __post_init__(self) -> None:
        if self.stop_type not in VALID_STOP_TYPES:
            raise ValueError(f"stop_type must be one of {sorted(VALID_STOP_TYPES)}, got {self.stop_type!r}")
        self.order_id = str(self.order_id)
        self.lat = normalize_coordinate(self.lat)
        self.lng = normalize_coordinate(self.lng)
