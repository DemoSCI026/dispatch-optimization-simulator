"""Score-based greedy assignment policy."""

from __future__ import annotations

from time import perf_counter

from src.cost.cost_engine import CostEngine
from src.policies.base_policy import Policy, build_assignment_result, iter_eligible_couriers
from src.routing.routing_engine import RoutingEngine
from src.state.assignment_result import AssignmentDecision, AssignmentResult
from src.state.checkpoint_state import CheckpointState


class ScoreGreedyPolicy(Policy):
    """Assign each order using the weighted pair cost from CostEngine."""

    name = "score_greedy"

    def solve(
        self,
        state: CheckpointState,
        cost_engine: CostEngine,
        routing_engine: RoutingEngine | None = None,
    ) -> AssignmentResult:
        start_time = perf_counter()
        decisions: list[AssignmentDecision] = []
        updated_courier_ids: set[str] = set()
        route_builder = routing_engine or RoutingEngine(cost_engine)

        for order in state.candidate_orders:
            best_courier = None
            best_score = float("inf")

            for courier in iter_eligible_couriers(state):
                score = cost_engine.pair_cost(courier, order)
                if score < best_score:
                    best_score = score
                    best_courier = courier

            if best_courier is None:
                decisions.append(AssignmentDecision(order_id=order.order_id, courier_id=None, assigned=False))
                continue

            detour = cost_engine.estimate_detour(best_courier, order)
            lateness = cost_engine.estimate_lateness_risk(best_courier, order)
            decisions.append(
                AssignmentDecision(
                    order_id=order.order_id,
                    courier_id=best_courier.courier_id,
                    assigned=True,
                    estimated_detour=detour,
                    estimated_lateness=lateness,
                    score=best_score,
                )
            )
            best_courier.add_order(order)
            updated_courier_ids.add(best_courier.courier_id)

        for courier in state.candidate_couriers:
            if courier.courier_id in updated_courier_ids:
                courier.route_plan = route_builder.build_route_from_orders(courier)

        runtime_sec = perf_counter() - start_time
        return build_assignment_result(state.checkpoint_time, self.name, decisions, runtime_sec)
