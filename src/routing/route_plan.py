"""Simplified route plan container."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.routing.route_stop import RouteStop


@dataclass(slots=True)
class RoutePlan:
    """Ordered list of route stops for one courier."""

    courier_id: str
    checkpoint_time: int
    stops: list[RouteStop] = field(default_factory=list)
    total_distance: float | None = None
    total_eta: float | None = None

    def copy(self) -> "RoutePlan":
        return RoutePlan(
            courier_id=self.courier_id,
            checkpoint_time=self.checkpoint_time,
            stops=list(self.stops),
            total_distance=self.total_distance,
            total_eta=self.total_eta,
        )

    def with_inserted_order(
        self,
        pickup_index: int,
        dropoff_index: int,
        pickup_stop: RouteStop,
        dropoff_stop: RouteStop,
    ) -> "RoutePlan":
        new_stops = list(self.stops)
        new_stops.insert(pickup_index, pickup_stop)
        new_stops.insert(dropoff_index, dropoff_stop)
        return RoutePlan(
            courier_id=self.courier_id,
            checkpoint_time=self.checkpoint_time,
            stops=new_stops,
        )
