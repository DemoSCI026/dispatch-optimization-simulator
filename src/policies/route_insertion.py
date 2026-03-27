"""Route-aware greedy insertion policy."""

from __future__ import annotations

from time import perf_counter

from src.cost.cost_engine import CostEngine
from src.policies.base_policy import Policy, build_assignment_result, iter_eligible_couriers
from src.routing.routing_engine import RoutingEngine
from src.state.assignment_result import AssignmentDecision, AssignmentResult
from src.state.checkpoint_state import CheckpointState


class RouteInsertionPolicy(Policy):
    """Sequentially insert each order into the best courier route."""

    name = "route_insertion"

    def solve(
        self,
        state: CheckpointState,
        cost_engine: CostEngine,
        routing_engine: RoutingEngine | None = None,
    ) -> AssignmentResult:
        if routing_engine is None:
            raise ValueError("RouteInsertionPolicy requires a RoutingEngine instance")

        start_time = perf_counter()
        for courier in state.candidate_couriers:
            if courier.route_plan is None:
                courier.route_plan = routing_engine.build_initial_route(courier)

        decisions: list[AssignmentDecision] = []

        for order in state.candidate_orders:
            best_courier = None
            best_route = None
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
                score = (
                    cost_engine.alpha * delta_distance
                    + cost_engine.beta * lateness_risk
                    + cost_engine.gamma * cost_engine.workload_penalty(courier)
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
