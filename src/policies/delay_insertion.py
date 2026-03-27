"""Route-aware insertion policy with optional delay thresholds."""

from __future__ import annotations

from time import perf_counter

from src.cost.cost_engine import CostEngine
from src.policies.base_policy import Policy, build_assignment_result, iter_eligible_couriers
from src.routing.route_plan import RoutePlan
from src.routing.routing_engine import RoutingEngine
from src.state.assignment_result import AssignmentDecision, AssignmentResult
from src.state.checkpoint_state import CheckpointState
from src.state.courier_state import CourierState


class DelayInsertionPolicy(Policy):
    """Insert into the best courier route unless thresholds say to delay."""

    name = "delay_insertion"

    def __init__(
        self,
        score_threshold: float | None = None,
        detour_threshold: float | None = None,
        lateness_threshold: float | None = None,
    ) -> None:
        self.score_threshold = score_threshold
        self.detour_threshold = detour_threshold
        self.lateness_threshold = lateness_threshold

    def solve(
        self,
        state: CheckpointState,
        cost_engine: CostEngine,
        routing_engine: RoutingEngine | None = None,
    ) -> AssignmentResult:
        if routing_engine is None:
            raise ValueError("DelayInsertionPolicy requires a RoutingEngine instance")

        start_time = perf_counter()
        for courier in state.candidate_couriers:
            if courier.route_plan is None:
                courier.route_plan = routing_engine.build_initial_route(courier)

        decisions: list[AssignmentDecision] = []

        for order in state.candidate_orders:
            best_courier: CourierState | None = None
            best_route: RoutePlan | None = None
            best_delta = float("inf")
            best_score = float("inf")
            best_lateness = 0.0

            for courier in iter_eligible_couriers(state):
                route_plan = courier.route_plan or routing_engine.build_initial_route(courier)
                candidate_route, insertion_info = routing_engine.try_insert_order(route_plan, order)
                if not insertion_info.get("feasible", False):
                    continue

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
                decisions.append(AssignmentDecision(order_id=order.order_id, courier_id=None, assigned=False))
                continue

            if self._should_delay(best_score, best_delta, best_lateness):
                decisions.append(
                    AssignmentDecision(
                        order_id=order.order_id,
                        courier_id=None,
                        assigned=False,
                        estimated_detour=best_delta,
                        estimated_lateness=best_lateness,
                        score=best_score,
                    )
                )
                continue

            best_courier.route_plan = best_route
            best_courier.add_order(order)
            decisions.append(
                AssignmentDecision(
                    order_id=order.order_id,
                    courier_id=best_courier.courier_id,
                    assigned=True,
                    estimated_detour=best_delta,
                    estimated_lateness=best_lateness,
                    score=best_score,
                )
            )

        runtime_sec = perf_counter() - start_time
        return build_assignment_result(state.checkpoint_time, self.name, decisions, runtime_sec)

    def _should_delay(self, score: float, delta_distance: float, lateness_risk: float) -> bool:
        if self.score_threshold is not None and score > self.score_threshold:
            return True
        if self.detour_threshold is not None and delta_distance > self.detour_threshold:
            return True
        if self.lateness_threshold is not None and lateness_risk > self.lateness_threshold:
            return True
        return False
