"""Simplified routing engine for sequential route insertion."""

from __future__ import annotations

from src.cost.cost_engine import CostEngine
from src.routing.route_plan import RoutePlan
from src.routing.route_stop import RouteStop
from src.state.courier_state import CourierState
from src.state.order import Order


class RoutingEngine:
    """Route builder and insertion heuristic based on straight-line distance."""

    def __init__(self, cost_engine: CostEngine | None = None) -> None:
        self.cost_engine = cost_engine or CostEngine()

    def build_initial_route(self, courier: CourierState) -> RoutePlan:
        """Create a naive pickup-then-dropoff route from current on-hand orders."""
        stops: list[RouteStop] = []
        sorted_orders = sorted(
            courier.on_hand_orders,
            key=lambda order: (order.promise_time is None, order.promise_time or float("inf"), order.order_id),
        )

        for order in sorted_orders:
            stops.append(
                RouteStop(
                    order_id=order.order_id,
                    stop_type="pickup",
                    lat=order.pickup_lat,
                    lng=order.pickup_lng,
                    latest_time=order.est_meal_ready_time,
                )
            )
            stops.append(
                RouteStop(
                    order_id=order.order_id,
                    stop_type="dropoff",
                    lat=order.dropoff_lat,
                    lng=order.dropoff_lng,
                    latest_time=order.promise_time,
                )
            )

        route_plan = RoutePlan(
            courier_id=courier.courier_id,
            checkpoint_time=courier.checkpoint_time,
            stops=stops,
        )
        return self._populate_route_stats(route_plan)

    def build_route_from_orders(self, courier: CourierState) -> RoutePlan:
        """Reconstruct a lightweight valid route from current on-hand orders.

        This is an approximate deterministic heuristic, not a route optimizer. Starting
        from the courier location, it repeatedly picks the nearest feasible next stop:
        pickup stops for not-yet-picked orders, or dropoff stops for already-picked ones.
        """
        remaining_pickups = {order.order_id: order for order in courier.on_hand_orders}
        available_dropoffs: dict[str, Order] = {}
        current_lat = courier.lat
        current_lng = courier.lng
        stops: list[RouteStop] = []

        while remaining_pickups or available_dropoffs:
            candidates: list[tuple[float, str, int, Order]] = []

            for order in remaining_pickups.values():
                distance = self.cost_engine.travel_distance(current_lat, current_lng, order.pickup_lat, order.pickup_lng)
                candidates.append((distance, order.order_id, 0, order))

            for order in available_dropoffs.values():
                distance = self.cost_engine.travel_distance(current_lat, current_lng, order.dropoff_lat, order.dropoff_lng)
                candidates.append((distance, order.order_id, 1, order))

            if not candidates:
                break

            _, _, stop_priority, next_order = min(candidates)

            if stop_priority == 0:
                next_stop = RouteStop(
                    order_id=next_order.order_id,
                    stop_type="pickup",
                    lat=next_order.pickup_lat,
                    lng=next_order.pickup_lng,
                    latest_time=next_order.est_meal_ready_time,
                )
                remaining_pickups.pop(next_order.order_id, None)
                available_dropoffs[next_order.order_id] = next_order
            else:
                next_stop = RouteStop(
                    order_id=next_order.order_id,
                    stop_type="dropoff",
                    lat=next_order.dropoff_lat,
                    lng=next_order.dropoff_lng,
                    latest_time=next_order.promise_time,
                )
                available_dropoffs.pop(next_order.order_id, None)

            stops.append(next_stop)
            current_lat = next_stop.lat
            current_lng = next_stop.lng

        route_plan = RoutePlan(
            courier_id=courier.courier_id,
            checkpoint_time=courier.checkpoint_time,
            stops=stops,
        )
        return self._populate_route_stats(route_plan)

    def route_cost(self, route_plan: RoutePlan) -> float:
        """Compute route distance as the sum over adjacent stops."""
        if len(route_plan.stops) < 2:
            return 0.0

        total_distance = 0.0
        for current_stop, next_stop in zip(route_plan.stops, route_plan.stops[1:]):
            total_distance += self.cost_engine.travel_distance(
                current_stop.lat,
                current_stop.lng,
                next_stop.lat,
                next_stop.lng,
            )
        return total_distance

    def try_insert_order(self, route_plan: RoutePlan, order: Order) -> tuple[RoutePlan, dict[str, float | bool]]:
        """Try all pickup-before-dropoff insertion positions and keep the best one."""
        pickup_stop = RouteStop(
            order_id=order.order_id,
            stop_type="pickup",
            lat=order.pickup_lat,
            lng=order.pickup_lng,
            latest_time=order.est_meal_ready_time,
        )
        dropoff_stop = RouteStop(
            order_id=order.order_id,
            stop_type="dropoff",
            lat=order.dropoff_lat,
            lng=order.dropoff_lng,
            latest_time=order.promise_time,
        )

        base_route = self._populate_route_stats(route_plan.copy())
        base_cost = base_route.total_distance or 0.0

        best_route: RoutePlan | None = None
        best_delta = float("inf")
        stop_count = len(base_route.stops)

        for pickup_index in range(stop_count + 1):
            for dropoff_index in range(pickup_index + 1, stop_count + 2):
                candidate_route = base_route.with_inserted_order(
                    pickup_index=pickup_index,
                    dropoff_index=dropoff_index,
                    pickup_stop=pickup_stop,
                    dropoff_stop=dropoff_stop,
                )
                candidate_route = self._populate_route_stats(candidate_route)
                delta_distance = (candidate_route.total_distance or 0.0) - base_cost

                if delta_distance < best_delta:
                    best_delta = delta_distance
                    best_route = candidate_route

        if best_route is None:
            return base_route, {"delta_distance": 0.0, "feasible": False}

        return best_route, {"delta_distance": best_delta, "feasible": True}

    def _populate_route_stats(self, route_plan: RoutePlan) -> RoutePlan:
        route_plan.total_distance = self.route_cost(route_plan)
        route_plan.total_eta = self.cost_engine.travel_minutes(route_plan.total_distance)
        return route_plan
