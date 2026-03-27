"""Policy interface and small helpers for assignment results."""

from __future__ import annotations

from abc import ABC, abstractmethod
from statistics import mean
from typing import Iterable

from src.cost.cost_engine import CostEngine
from src.routing.routing_engine import RoutingEngine
from src.state.assignment_result import AssignmentDecision, AssignmentResult
from src.state.checkpoint_state import CheckpointState
from src.state.courier_state import CourierState


class Policy(ABC):
    """Abstract interface shared by all checkpoint policies."""

    name: str = "base_policy"

    @abstractmethod
    def solve(
        self,
        state: CheckpointState,
        cost_engine: CostEngine,
        routing_engine: RoutingEngine | None = None,
    ) -> AssignmentResult:
        """Return assignment results for a single checkpoint."""



def iter_eligible_couriers(state: CheckpointState) -> Iterable[CourierState]:
    """Yield couriers that still satisfy the simulator capacity constraint."""
    for courier in state.candidate_couriers:
        if courier.can_accept_order(state.max_on_hand_orders):
            yield courier



def build_assignment_result(
    checkpoint_time: int,
    policy_name: str,
    decisions: list[AssignmentDecision],
    runtime_sec: float,
) -> AssignmentResult:
    """Create an AssignmentResult with lightweight summary metrics."""
    assigned_decisions = [decision for decision in decisions if decision.assigned]
    detours = [
        decision.estimated_detour
        for decision in assigned_decisions
        if decision.estimated_detour is not None
    ]
    lateness = [
        decision.estimated_lateness
        for decision in assigned_decisions
        if decision.estimated_lateness is not None
    ]

    summary_metrics = {
        "avg_detour": mean(detours) if detours else 0.0,
        "avg_lateness": mean(lateness) if lateness else 0.0,
        "assigned_ratio": (len(assigned_decisions) / len(decisions)) if decisions else 0.0,
        "assigned_count": float(len(assigned_decisions)),
        "unassigned_count": float(len(decisions) - len(assigned_decisions)),
    }

    return AssignmentResult(
        checkpoint_time=checkpoint_time,
        policy_name=policy_name,
        decisions=decisions,
        runtime_sec=runtime_sec,
        summary_metrics=summary_metrics,
    )
