"""Pooling-aware route insertion with lightweight checkpoint-local delay gating."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import floor
from time import perf_counter

from src.cost.cost_engine import CostEngine
from src.policies.base_policy import Policy, build_assignment_result, iter_eligible_couriers
from src.routing.route_plan import RoutePlan
from src.routing.routing_engine import RoutingEngine
from src.state.assignment_result import AssignmentDecision, AssignmentResult
from src.state.checkpoint_state import CheckpointState
from src.state.courier_state import CourierState
from src.state.order import Order


@dataclass(slots=True)
class _BestInsertion:
    """Best feasible insertion summary for one order."""

    courier: CourierState | None
    route: RoutePlan | None
    delta_distance: float | None
    lateness_risk: float | None
    score: float | None
    feasible_courier_count: int


class PoolingAwareDelayInsertionPolicy(Policy):
    """Delay only when a weak current insertion also has strong pooling potential."""

    name = "pooling_aware_delay_insertion"
    _COORDINATE_CELL_SIZE = 0.01

    def __init__(
        self,
        detour_scale: float = 3.0,
        pickup_scale: float = 5.0,
        delivery_scale: float = 5.0,
        slack_scale: float = 1800.0,
        w_spatial: float = 0.6,
        w_temporal: float = 0.4,
        min_slack: int = 300,
        good_detour_threshold: float = 0.3,
        bad_detour_threshold: float = 0.6,
        pooling_threshold: float = 0.5,
        max_delay_ratio: float = 0.3,
    ) -> None:
        self.detour_scale = detour_scale
        self.pickup_scale = pickup_scale
        self.delivery_scale = delivery_scale
        self.slack_scale = slack_scale
        self.w_spatial = w_spatial
        self.w_temporal = w_temporal
        self.min_slack = min_slack
        self.good_detour_threshold = good_detour_threshold
        self.bad_detour_threshold = bad_detour_threshold
        self.pooling_threshold = pooling_threshold
        self.max_delay_ratio = max_delay_ratio

    def solve(
        self,
        state: CheckpointState,
        cost_engine: CostEngine,
        routing_engine: RoutingEngine | None = None,
    ) -> AssignmentResult:
        if routing_engine is None:
            raise ValueError("PoolingAwareDelayInsertionPolicy requires a RoutingEngine instance")

        start_time = perf_counter()
        feature_start = perf_counter()
        for courier in state.candidate_couriers:
            if courier.route_plan is None:
                courier.route_plan = routing_engine.build_initial_route(courier)

        pickup_neighbor_counts = self._build_pickup_neighbor_counts(state.candidate_orders)
        delivery_neighbor_counts = self._build_delivery_neighbor_counts(state.candidate_orders)
        feature_time = perf_counter() - feature_start
        insertion_time = 0.0
        decision_time = 0.0
        decisions: list[AssignmentDecision] = []
        total_orders = len(state.candidate_orders)
        delayed_orders = 0

        for order in state.candidate_orders:
            insert_start = perf_counter()
            best_insertion = self._compute_best_insertion(state, order, cost_engine, routing_engine)
            insertion_time += perf_counter() - insert_start

            decision_start = perf_counter()
            if best_insertion.courier is None or best_insertion.route is None:
                decisions.append(AssignmentDecision(order_id=order.order_id, courier_id=None, assigned=False))
                decision_time += perf_counter() - decision_start
                continue

            slack_seconds = self._slack_seconds(order, state.checkpoint_time)
            if slack_seconds is None or slack_seconds < self.min_slack:
                self._commit_assignment(best_insertion, order, decisions)
                decision_time += perf_counter() - decision_start
                continue

            detour_badness = self._normalize_ratio(best_insertion.delta_distance or 0.0, self.detour_scale)
            if detour_badness < self.good_detour_threshold:
                self._commit_assignment(best_insertion, order, decisions)
                decision_time += perf_counter() - decision_start
                continue

            spatial_pooling = self._compute_spatial_pooling(
                order,
                pickup_neighbor_counts,
                delivery_neighbor_counts,
            )
            temporal_pooling = self._normalize_ratio(float(slack_seconds), self.slack_scale)
            pooling_potential = (
                self.w_spatial * spatial_pooling
                + self.w_temporal * temporal_pooling
            )
            current_delay_ratio = (delayed_orders / total_orders) if total_orders else 0.0

            if (
                pooling_potential > self.pooling_threshold
                and detour_badness > self.bad_detour_threshold
                and current_delay_ratio < self.max_delay_ratio
            ):
                delayed_orders += 1
                decisions.append(
                    AssignmentDecision(
                        order_id=order.order_id,
                        courier_id=None,
                        assigned=False,
                        delayed_to_next_checkpoint=True,
                        estimated_detour=best_insertion.delta_distance,
                        estimated_lateness=best_insertion.lateness_risk,
                        score=pooling_potential,
                    )
                )
                decision_time += perf_counter() - decision_start
                continue

            self._commit_assignment(best_insertion, order, decisions)
            decision_time += perf_counter() - decision_start

        runtime_sec = perf_counter() - start_time
        result = build_assignment_result(state.checkpoint_time, self.name, decisions, runtime_sec)
        delay_ratio = (delayed_orders / total_orders) if total_orders else 0.0
        result.summary_metrics.update(
            {
                "total_orders": float(total_orders),
                "delayed_orders": float(delayed_orders),
                "delay_ratio": delay_ratio,
            }
        )
        print(f"[DEBUG] delay_ratio={delay_ratio:.3f} (budget={self.max_delay_ratio})")
        print(
            "[PROFILE] feature_time={feature_time:.3f}s insertion_time={insertion_time:.3f}s "
            "decision_time={decision_time:.3f}s total_time={total_time:.3f}s".format(
                feature_time=feature_time,
                insertion_time=insertion_time,
                decision_time=decision_time,
                total_time=runtime_sec,
            )
        )
        return result

    def _compute_best_insertion(
        self,
        state: CheckpointState,
        order: Order,
        cost_engine: CostEngine,
        routing_engine: RoutingEngine,
    ) -> _BestInsertion:
        best_courier: CourierState | None = None
        best_route: RoutePlan | None = None
        best_delta = float("inf")
        best_score = float("inf")
        best_lateness = 0.0
        feasible_courier_count = 0

        for courier in iter_eligible_couriers(state):
            route_plan = courier.route_plan or routing_engine.build_initial_route(courier)
            candidate_route, insertion_info = routing_engine.try_insert_order(route_plan, order)
            if not insertion_info.get("feasible", False):
                continue

            feasible_courier_count += 1
            delta_distance = float(insertion_info.get("delta_distance", 0.0))
            lateness_risk = cost_engine.estimate_lateness_risk(courier, order)
            workload_penalty = cost_engine.workload_penalty(courier)
            score = (
                cost_engine.alpha * delta_distance
                + cost_engine.beta * lateness_risk
                + cost_engine.gamma * workload_penalty
            )

            if score < best_score:
                best_courier = courier
                best_route = candidate_route
                best_delta = delta_distance
                best_score = score
                best_lateness = lateness_risk

        if best_courier is None or best_route is None:
            return _BestInsertion(
                courier=None,
                route=None,
                delta_distance=None,
                lateness_risk=None,
                score=None,
                feasible_courier_count=feasible_courier_count,
            )

        return _BestInsertion(
            courier=best_courier,
            route=best_route,
            delta_distance=best_delta,
            lateness_risk=best_lateness,
            score=best_score,
            feasible_courier_count=feasible_courier_count,
        )

    def _build_pickup_neighbor_counts(self, orders: list[Order]) -> dict[str, int]:
        area_counts = Counter(order.area_id for order in orders if order.area_id)
        fallback_counts = Counter(
            self._coordinate_cell(order.pickup_lat, order.pickup_lng)
            for order in orders
            if not order.area_id
        )

        neighbor_counts: dict[str, int] = {}
        for order in orders:
            if order.area_id:
                neighbor_counts[order.order_id] = max(area_counts[order.area_id] - 1, 0)
            else:
                cell = self._coordinate_cell(order.pickup_lat, order.pickup_lng)
                neighbor_counts[order.order_id] = max(fallback_counts[cell] - 1, 0)

        return neighbor_counts

    def _build_delivery_neighbor_counts(self, orders: list[Order]) -> dict[str, int]:
        delivery_counts = Counter(
            self._coordinate_cell(order.dropoff_lat, order.dropoff_lng)
            for order in orders
        )
        return {
            order.order_id: max(
                delivery_counts[self._coordinate_cell(order.dropoff_lat, order.dropoff_lng)] - 1,
                0,
            )
            for order in orders
        }

    def _compute_spatial_pooling(
        self,
        order: Order,
        pickup_neighbor_counts: dict[str, int],
        delivery_neighbor_counts: dict[str, int],
    ) -> float:
        pickup_neighbor_count = pickup_neighbor_counts.get(order.order_id, 0)
        delivery_neighbor_count = delivery_neighbor_counts.get(order.order_id, 0)
        pickup_score = self._normalize_ratio(float(pickup_neighbor_count), self.pickup_scale)
        delivery_score = self._normalize_ratio(float(delivery_neighbor_count), self.delivery_scale)
        return 0.5 * pickup_score + 0.5 * delivery_score

    def _commit_assignment(
        self,
        best_insertion: _BestInsertion,
        order: Order,
        decisions: list[AssignmentDecision],
    ) -> None:
        if best_insertion.courier is None or best_insertion.route is None:
            decisions.append(AssignmentDecision(order_id=order.order_id, courier_id=None, assigned=False))
            return

        best_insertion.courier.route_plan = best_insertion.route
        best_insertion.courier.add_order(order)
        decisions.append(
            AssignmentDecision(
                order_id=order.order_id,
                courier_id=best_insertion.courier.courier_id,
                assigned=True,
                estimated_detour=best_insertion.delta_distance,
                estimated_lateness=best_insertion.lateness_risk,
                score=best_insertion.score,
            )
        )

    @staticmethod
    def _slack_seconds(order: Order, checkpoint_time: int) -> int | None:
        if order.promise_time is None:
            return None
        return order.promise_time - checkpoint_time

    @classmethod
    def _coordinate_cell(cls, lat: float, lng: float) -> tuple[int, int]:
        cell_size = cls._COORDINATE_CELL_SIZE
        return (floor(lat / cell_size), floor(lng / cell_size))

    @staticmethod
    def _normalize_ratio(value: float, scale: float) -> float:
        if scale <= 0.0:
            return 0.0
        return max(0.0, min(value / scale, 1.0))
