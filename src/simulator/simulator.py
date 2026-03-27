"""Stateful checkpoint replay simulator for dispatch policies."""

from __future__ import annotations

from src.cost.cost_engine import CostEngine
from src.evaluation.metrics import (
    compute_assigned_ratio,
    compute_avg_backlog_size,
    compute_avg_courier_load,
    compute_avg_detour,
    compute_avg_runtime,
    compute_completion_ratio,
    compute_total_assigned_orders,
    compute_total_unassigned_orders,
)
from src.policies.base_policy import Policy
from src.routing.routing_engine import RoutingEngine
from src.state.assignment_result import AssignmentResult
from src.state.checkpoint_state import CheckpointState
from src.state.courier_state import CourierState
from src.state.order import Order


class Simulator:
    """Replay checkpoints while keeping courier and order state across time.

    The simulator persists couriers, active orders, backlog, and approximate route/
    ETA information across checkpoints. ETA is a lightweight estimation for replay
    comparison, not a production routing promise.
    """

    def __init__(self, max_on_hand_orders: int = 3, backlog_grace_period: int = 300) -> None:
        self.max_on_hand_orders = max_on_hand_orders
        self.backlog_grace_period = backlog_grace_period
        self.initialize_global_state()

    def initialize_global_state(self) -> None:
        self.current_time: int | None = None
        self.active_couriers: dict[str, CourierState] = {}
        self.active_orders: dict[str, Order] = {}
        self.backlog_order_ids: set[str] = set()
        self.assigned_order_to_courier: dict[str, str] = {}
        self.order_expected_completion_time: dict[str, int] = {}
        self.order_is_late: dict[str, bool] = {}
        self.completed_orders: dict[str, Order] = {}
        self.expired_order_ids: set[str] = set()
        self.seen_order_ids: set[str] = set()
        self.last_transition_summary: dict[str, int] = {
            "completed_now": 0,
            "expired_now": 0,
            "late_now": 0,
        }

    def run(
        self,
        checkpoints: list[CheckpointState],
        policy: Policy,
        cost_engine: CostEngine,
        routing_engine: RoutingEngine | None = None,
    ) -> list[AssignmentResult]:
        self.initialize_global_state()
        results: list[AssignmentResult] = []

        for raw_checkpoint in sorted(checkpoints, key=lambda item: item.checkpoint_time):
            self.advance_to_checkpoint(raw_checkpoint.checkpoint_time)
            policy_state = self.build_policy_state(raw_checkpoint)
            result = policy.solve(policy_state, cost_engine, routing_engine)
            self.apply_assignment_result(policy_state, result, cost_engine)
            self._attach_checkpoint_metrics(result, policy_state)
            results.append(result)

        return results

    def advance_to_checkpoint(self, checkpoint_time: int) -> None:
        """Advance the global clock and update completion/late/expiration state."""
        self.current_time = checkpoint_time
        self.last_transition_summary = self.finalize_finished_orders(checkpoint_time)

    def build_policy_state(self, raw_checkpoint: CheckpointState) -> CheckpointState:
        """Build the policy-facing checkpoint from global persistent state."""
        observed_couriers = self._merge_observed_couriers(raw_checkpoint)
        self._register_candidate_orders(raw_checkpoint.candidate_orders)
        candidate_orders = self._collect_backlog_orders()
        candidate_couriers = [courier.copy_for_policy() for courier in observed_couriers]

        return CheckpointState(
            checkpoint_time=raw_checkpoint.checkpoint_time,
            dt=raw_checkpoint.dt,
            candidate_orders=candidate_orders,
            candidate_couriers=candidate_couriers,
            max_on_hand_orders=self.max_on_hand_orders,
        )

    def apply_assignment_result(
        self,
        state: CheckpointState,
        result: AssignmentResult,
        cost_engine: CostEngine,
    ) -> None:
        """Apply policy decisions and reconstructed routes back into global state."""
        current_time = self.current_time or result.checkpoint_time
        policy_couriers = {courier.courier_id: courier for courier in state.candidate_couriers}
        updated_courier_ids: set[str] = set()
        self._sync_policy_couriers(policy_couriers)

        for decision in result.decisions:
            order = self.active_orders.get(decision.order_id)
            if order is None:
                continue

            if decision.assigned and decision.courier_id is not None:
                global_courier = self.active_couriers.get(decision.courier_id)
                policy_courier = policy_couriers.get(decision.courier_id)
                if global_courier is None or policy_courier is None:
                    continue

                global_courier.add_order(order)
                self.assigned_order_to_courier[order.order_id] = global_courier.courier_id
                self.backlog_order_ids.discard(order.order_id)
                self.order_is_late[order.order_id] = bool(
                    order.promise_time is not None and current_time > order.promise_time
                )
                updated_courier_ids.add(global_courier.courier_id)
            else:
                self.backlog_order_ids.add(order.order_id)

        for courier_id in updated_courier_ids:
            courier = self.active_couriers.get(courier_id)
            if courier is not None:
                self._refresh_courier_order_etas(courier, current_time, cost_engine)

    def finalize_finished_orders(self, checkpoint_time: int) -> dict[str, int]:
        """Advance order lifecycle using ETA completion, lateness flags, and backlog expiration.

        Completion is ETA-driven. Lateness is tracked independently from completion.
        Backlog orders expire only after promise_time plus a grace period.
        """
        completed_now = 0
        expired_now = 0
        late_now = 0

        for order_id, order in list(self.active_orders.items()):
            courier_id = self.assigned_order_to_courier.get(order_id)
            predicted_completion_time = self.order_expected_completion_time.get(order_id)

            if courier_id is not None:
                if predicted_completion_time is not None and predicted_completion_time <= checkpoint_time:
                    self._complete_order(order_id, order, courier_id)
                    completed_now += 1
                    continue

                if (
                    order.promise_time is not None
                    and checkpoint_time > order.promise_time
                    and not self.order_is_late.get(order_id, False)
                ):
                    self.order_is_late[order_id] = True
                    late_now += 1
                continue

            if (
                order.promise_time is not None
                and checkpoint_time > order.promise_time + self.backlog_grace_period
            ):
                self._expire_backlog_order(order_id)
                expired_now += 1

        return {"completed_now": completed_now, "expired_now": expired_now, "late_now": late_now}

    def summarize(self, results: list[AssignmentResult]) -> dict[str, float]:
        return {
            "avg_detour": compute_avg_detour(results),
            "assigned_ratio": compute_assigned_ratio(results),
            "avg_runtime": compute_avg_runtime(results),
            "avg_backlog_size": compute_avg_backlog_size(results),
            "avg_courier_load": compute_avg_courier_load(results),
            "total_assigned_orders": compute_total_assigned_orders(results),
            "total_unassigned_orders": compute_total_unassigned_orders(results),
            "completion_ratio": compute_completion_ratio(results),
            "completed_orders": float(len(self.completed_orders)),
            "expired_orders": float(len(self.expired_order_ids)),
            "late_orders": float(sum(1 for is_late in self.order_is_late.values() if is_late)),
            "remaining_backlog": float(len(self.backlog_order_ids)),
            "remaining_active_orders": float(len(self.active_orders)),
            "active_couriers": float(len(self.active_couriers)),
        }

    def _merge_observed_couriers(self, raw_checkpoint: CheckpointState) -> list[CourierState]:
        merged_couriers: list[CourierState] = []

        for observed_courier in raw_checkpoint.candidate_couriers:
            courier = self.active_couriers.get(observed_courier.courier_id)
            if courier is None:
                courier = observed_courier
                self.active_couriers[courier.courier_id] = courier
            else:
                courier.sync_observation(
                    dt=observed_courier.dt,
                    checkpoint_time=observed_courier.checkpoint_time,
                    lat=observed_courier.lat,
                    lng=observed_courier.lng,
                )
                if observed_courier.route_plan is not None:
                    courier.route_plan = observed_courier.route_plan.copy()

            for order in observed_courier.on_hand_orders:
                self._register_assigned_order(order, courier.courier_id)
                courier.add_order(order)

            merged_couriers.append(courier)

        return merged_couriers

    def _register_candidate_orders(self, orders: list[Order]) -> None:
        for order in orders:
            order_id = order.order_id
            self.seen_order_ids.add(order_id)
            if order_id in self.completed_orders or order_id in self.expired_order_ids:
                continue

            self.active_orders.setdefault(order_id, order)
            self.order_is_late.setdefault(order_id, False)
            if order_id not in self.assigned_order_to_courier:
                self.backlog_order_ids.add(order_id)

    def _register_assigned_order(self, order: Order, courier_id: str) -> None:
        order_id = order.order_id
        self.seen_order_ids.add(order_id)
        if order_id in self.completed_orders or order_id in self.expired_order_ids:
            return

        self.active_orders.setdefault(order_id, order)
        self.order_is_late.setdefault(order_id, False)
        self.assigned_order_to_courier[order_id] = courier_id
        self.backlog_order_ids.discard(order_id)

    def _collect_backlog_orders(self) -> list[Order]:
        backlog_orders = [
            self.active_orders[order_id]
            for order_id in self.backlog_order_ids
            if order_id in self.active_orders
        ]
        backlog_orders.sort(key=self._order_priority)
        return backlog_orders

    def _sync_policy_couriers(self, policy_couriers: dict[str, CourierState]) -> None:
        for courier_id, policy_courier in policy_couriers.items():
            global_courier = self.active_couriers.get(courier_id)
            if global_courier is None:
                continue
            if policy_courier.route_plan is not None:
                global_courier.route_plan = policy_courier.route_plan.copy()

    def _attach_checkpoint_metrics(self, result: AssignmentResult, state: CheckpointState) -> None:
        avg_courier_load = 0.0
        if state.candidate_couriers:
            avg_courier_load = sum(courier.current_load for courier in state.candidate_couriers) / len(state.candidate_couriers)

        result.summary_metrics.update(
            {
                "backlog_size": float(len(self.backlog_order_ids)),
                "avg_courier_load": avg_courier_load,
                "completed_orders": float(len(self.completed_orders)),
                "expired_orders": float(len(self.expired_order_ids)),
                "late_orders": float(sum(1 for is_late in self.order_is_late.values() if is_late)),
                "completed_now": float(self.last_transition_summary.get("completed_now", 0)),
                "expired_now": float(self.last_transition_summary.get("expired_now", 0)),
                "late_now": float(self.last_transition_summary.get("late_now", 0)),
                "total_seen_orders": float(len(self.seen_order_ids)),
                "active_orders": float(len(self.active_orders)),
                "completion_ratio": (
                    len(self.completed_orders) / len(self.seen_order_ids)
                    if self.seen_order_ids
                    else 0.0
                ),
            }
        )

    def _refresh_courier_order_etas(
        self,
        courier: CourierState,
        current_time: int,
        cost_engine: CostEngine,
    ) -> None:
        """Refresh ETA for every active on-hand order after a route update."""
        for order in courier.on_hand_orders:
            self.order_expected_completion_time[order.order_id] = self._estimate_completion_time(
                current_time=current_time,
                courier=courier,
                order=order,
                cost_engine=cost_engine,
                estimated_detour=None,
            )

    def _estimate_completion_time(
        self,
        current_time: int,
        courier: CourierState,
        order: Order,
        cost_engine: CostEngine,
        estimated_detour: float | None,
    ) -> int:
        route_eta = self._estimate_completion_time_from_route(current_time, courier, order, cost_engine)
        if route_eta is not None:
            return route_eta

        detour = estimated_detour if estimated_detour is not None else cost_engine.estimate_detour(courier, order)
        wait_seconds = 0.0
        if order.est_meal_ready_time is not None and order.est_meal_ready_time > current_time:
            wait_seconds = float(order.est_meal_ready_time - current_time)
        travel_seconds = max(60.0, cost_engine.travel_minutes(detour) * 60.0)
        load_penalty_seconds = max(courier.current_load - 1, 0) * 180.0
        return int(current_time + wait_seconds + travel_seconds + load_penalty_seconds)

    def _estimate_completion_time_from_route(
        self,
        current_time: int,
        courier: CourierState,
        order: Order,
        cost_engine: CostEngine,
    ) -> int | None:
        if courier.route_plan is None or not courier.route_plan.stops:
            return None

        current_lat = courier.lat
        current_lng = courier.lng
        elapsed_seconds = 0.0

        for stop in courier.route_plan.stops:
            distance = cost_engine.travel_distance(current_lat, current_lng, stop.lat, stop.lng)
            elapsed_seconds += cost_engine.travel_minutes(distance) * 60.0

            if stop.stop_type == "pickup" and stop.order_id == order.order_id:
                if order.est_meal_ready_time is not None:
                    arrival_time = current_time + elapsed_seconds
                    if arrival_time < order.est_meal_ready_time:
                        elapsed_seconds += order.est_meal_ready_time - arrival_time

            if stop.stop_type == "dropoff" and stop.order_id == order.order_id:
                return int(current_time + max(elapsed_seconds, 60.0))

            current_lat = stop.lat
            current_lng = stop.lng

        return None

    def _complete_order(self, order_id: str, order: Order, courier_id: str) -> None:
        self.assigned_order_to_courier.pop(order_id, None)
        self.backlog_order_ids.discard(order_id)
        self.order_expected_completion_time.pop(order_id, None)
        self.active_orders.pop(order_id, None)
        if courier_id in self.active_couriers:
            self.active_couriers[courier_id].mark_order_completed(order_id)
        self.completed_orders[order_id] = order

    def _expire_backlog_order(self, order_id: str) -> None:
        self.backlog_order_ids.discard(order_id)
        self.active_orders.pop(order_id, None)
        self.order_expected_completion_time.pop(order_id, None)
        self.assigned_order_to_courier.pop(order_id, None)
        self.expired_order_ids.add(order_id)

    @staticmethod
    def _order_priority(order: Order) -> tuple[int, int, str]:
        priority_time = order.create_time or order.push_time or order.promise_time or 0
        has_priority_time = 0 if priority_time else 1
        return (has_priority_time, priority_time, order.order_id)
