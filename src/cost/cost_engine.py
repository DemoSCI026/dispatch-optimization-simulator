"""Cost calculations shared by dispatch policies."""

from __future__ import annotations

from dataclasses import dataclass
from math import asin, cos, radians, sin, sqrt

from src.state.courier_state import CourierState
from src.state.order import Order


@dataclass(slots=True)
class CostEngine:
    """Weighted heuristic cost engine for order-courier matching."""

    alpha: float = 1.0
    beta: float = 0.5
    gamma: float = 0.2
    avg_speed_kmh: float = 20.0
    service_time_minutes: float = 4.0

    def travel_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """Compute great-circle distance in kilometers."""
        earth_radius_km = 6371.0
        delta_lat = radians(lat2 - lat1)
        delta_lng = radians(lng2 - lng1)
        a = (
            sin(delta_lat / 2) ** 2
            + cos(radians(lat1)) * cos(radians(lat2)) * sin(delta_lng / 2) ** 2
        )
        c = 2 * asin(sqrt(a))
        return earth_radius_km * c

    def estimate_detour(self, courier: CourierState, order: Order) -> float:
        """Approximate incremental travel distance using courier position only."""
        to_pickup = self.travel_distance(courier.lat, courier.lng, order.pickup_lat, order.pickup_lng)
        to_dropoff = self.travel_distance(order.pickup_lat, order.pickup_lng, order.dropoff_lat, order.dropoff_lng)
        return to_pickup + to_dropoff

    def estimate_lateness_risk(self, courier: CourierState, order: Order) -> float:
        """Simple overtime proxy measured in late minutes."""
        if order.promise_time is None:
            return 0.0

        travel_minutes = self.travel_minutes(self.estimate_detour(courier, order))
        queue_minutes = courier.current_load * self.service_time_minutes
        slack_minutes = max(0.0, (order.promise_time - courier.checkpoint_time) / 60.0)
        return max(0.0, travel_minutes + queue_minutes - slack_minutes)

    def workload_penalty(self, courier: CourierState) -> float:
        return float(courier.current_load)

    def pair_cost(self, courier: CourierState, order: Order) -> float:
        detour = self.estimate_detour(courier, order)
        lateness_risk = self.estimate_lateness_risk(courier, order)
        workload = self.workload_penalty(courier)
        return self.alpha * detour + self.beta * lateness_risk + self.gamma * workload

    def travel_minutes(self, distance_km: float) -> float:
        speed = max(self.avg_speed_kmh, 1e-6)
        return distance_km / speed * 60.0
