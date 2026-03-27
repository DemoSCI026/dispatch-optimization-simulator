"""Order data model used by dispatch policies and routing logic."""

from __future__ import annotations

from dataclasses import dataclass

COORDINATE_SCALE = 1_000_000.0


def normalize_coordinate(value: float | int) -> float:
    """Convert integer micro-degree coordinates to plain latitude/longitude."""
    number = float(value)
    if abs(number) > 180.0:
        return number / COORDINATE_SCALE
    return number


@dataclass(slots=True)
class Order:
    """Minimal order representation for checkpoint-level assignment."""

    order_id: str
    waybill_id: str | None
    dt: str
    area_id: str | None
    create_time: int | None
    push_time: int | None
    promise_time: int | None
    est_meal_ready_time: int | None
    pickup_lat: float
    pickup_lng: float
    dropoff_lat: float
    dropoff_lng: float
    is_prebook: bool = False
    is_weekend: bool = False

    def __post_init__(self) -> None:
        self.order_id = str(self.order_id)
        self.waybill_id = None if self.waybill_id is None else str(self.waybill_id)
        self.area_id = None if self.area_id is None else str(self.area_id)
        self.pickup_lat = normalize_coordinate(self.pickup_lat)
        self.pickup_lng = normalize_coordinate(self.pickup_lng)
        self.dropoff_lat = normalize_coordinate(self.dropoff_lat)
        self.dropoff_lng = normalize_coordinate(self.dropoff_lng)

    @property
    def pickup_point(self) -> tuple[float, float]:
        return (self.pickup_lat, self.pickup_lng)

    @property
    def dropoff_point(self) -> tuple[float, float]:
        return (self.dropoff_lat, self.dropoff_lng)
