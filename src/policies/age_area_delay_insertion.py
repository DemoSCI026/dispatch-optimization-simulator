"""Age- and area-aware route insertion with lightweight delay gating."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
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


class AgeAreaDelayInsertionPolicy(Policy):
    """Delay low-urgency orders when detour and local area signals favor waiting."""

    name = "age_area_delay_insertion"

    def __init__(
        self,
        delay_threshold: float = 0.45,
        min_slack_to_delay: int = 300,
        max_age_to_delay: int = 1200,
        max_backlog_to_delay: int = 450,
        backlog_scale: float = 400.0,
        detour_scale: float = 3.0,
        area_count_scale: float = 5.0,
        supply_count_scale: float = 5.0,
        slack_scale: float = 1800.0,
        w_detour: float = 0.50,
        w_area: float = 0.20,
        w_supply: float = 0.20,
        w_urgency: float = 0.40,
        w_backlog: float = 0.30,
    ) -> None:
        self.delay_threshold = delay_threshold
        self.min_slack_to_delay = min_slack_to_delay
        self.max_age_to_delay = max_age_to_delay
        # Retained for constructor compatibility with existing experiment wiring.
        self.max_backlog_to_delay = max_backlog_to_delay
        self.backlog_scale = backlog_scale
        self.detour_scale = detour_scale
        self.area_count_scale = area_count_scale
        self.supply_count_scale = supply_count_scale
        self.slack_scale = slack_scale
        self.w_detour = w_detour
        self.w_area = w_area
        self.w_supply = w_supply
        self.w_urgency = w_urgency
        self.w_backlog = w_backlog

    def solve(
        self,
        state: CheckpointState,
        cost_engine: CostEngine,
        routing_engine: RoutingEngine | None = None,
    ) -> AssignmentResult:
        if routing_engine is None:
            raise ValueError("AgeAreaDelayInsertionPolicy requires a RoutingEngine instance")

        start_time = perf_counter()
        for courier in state.candidate_couriers:
            if courier.route_plan is None:
                courier.route_plan = routing_engine.build_initial_route(courier)

        area_hotness = self._build_area_hotness(state)
        backlog_size = len(state.candidate_orders)
        backlog_pressure = self._normalize_ratio(float(backlog_size), self.backlog_scale)
        decisions: list[AssignmentDecision] = []
        total_orders = 0
        delayed_orders = 0
        forced_assign_orders = 0

        for order in state.candidate_orders:
            total_orders += 1
            best_insertion = self._compute_best_insertion(state, order, cost_engine, routing_engine)

            if best_insertion.courier is None or best_insertion.route is None:
                decisions.append(AssignmentDecision(order_id=order.order_id, courier_id=None, assigned=False))
                continue

            if self._should_force_assign(order, state.checkpoint_time):
                forced_assign_orders += 1
                self._commit_assignment(best_insertion, order, decisions)
                continue

            delayable_score = self._compute_delayable_score(
                order=order,
                checkpoint_time=state.checkpoint_time,
                best_insertion=best_insertion,
                area_hotness=area_hotness.get(order.area_id or "", 0.0),
                backlog_pressure=backlog_pressure,
            )
            if delayable_score >= self.delay_threshold:
                delayed_orders += 1
                decisions.append(
                    AssignmentDecision(
                        order_id=order.order_id,
                        courier_id=None,
                        assigned=False,
                        delayed_to_next_checkpoint=True,
                        estimated_detour=best_insertion.delta_distance,
                        estimated_lateness=best_insertion.lateness_risk,
                        score=delayable_score,
                    )
                )
                continue

            self._commit_assignment(best_insertion, order, decisions)

        runtime_sec = perf_counter() - start_time
        delay_ratio = (delayed_orders / total_orders) if total_orders else 0.0
        force_assign_ratio = (forced_assign_orders / total_orders) if total_orders else 0.0
        print(f"[DEBUG] delay_ratio={delay_ratio:.3f}, force_assign_ratio={force_assign_ratio:.3f}")
        return build_assignment_result(state.checkpoint_time, self.name, decisions, runtime_sec)

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

    def _build_area_hotness(self, state: CheckpointState) -> dict[str, float]:
        counts = Counter(order.area_id for order in state.candidate_orders if order.area_id)
        return {
            area_id: self._normalize_ratio(float(count), self.area_count_scale)
            for area_id, count in counts.items()
        }

    def _compute_delayable_score(
        self,
        order: Order,
        checkpoint_time: int,
        best_insertion: _BestInsertion,
        area_hotness: float,
        backlog_pressure: float,
    ) -> float:
        detour_badness = self._normalize_ratio(best_insertion.delta_distance or 0.0, self.detour_scale)
        supply_scarcity = 1.0 - self._normalize_ratio(
            float(best_insertion.feasible_courier_count),
            self.supply_count_scale,
        )

        slack_seconds = self._slack_seconds(order, checkpoint_time)
        urgency_penalty = 0.0
        if slack_seconds is not None:
            urgency_penalty = 1.0 - self._normalize_ratio(float(slack_seconds), self.slack_scale)

        return (
            self.w_detour * detour_badness
            + self.w_area * area_hotness
            + self.w_supply * supply_scarcity
            - self.w_urgency * urgency_penalty
            - self.w_backlog * backlog_pressure
        )

    def _should_force_assign(self, order: Order, checkpoint_time: int) -> bool:
        slack_seconds = self._slack_seconds(order, checkpoint_time)
        if slack_seconds is not None and slack_seconds < self.min_slack_to_delay:
            return True

        order_age_seconds = self._order_age_seconds(order, checkpoint_time)
        if order_age_seconds is not None and order_age_seconds > (self.max_age_to_delay * 1.5):
            return True

        return False

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

    @staticmethod
    def _order_age_seconds(order: Order, checkpoint_time: int) -> int | None:
        if order.create_time is None:
            return None
        return checkpoint_time - order.create_time

    @staticmethod
    def _normalize_ratio(value: float, scale: float) -> float:
        if scale <= 0.0:
            return 0.0
        return max(0.0, min(value / scale, 1.0))
