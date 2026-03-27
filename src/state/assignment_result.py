"""Assignment output data models."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class AssignmentDecision:
    """Assignment decision for a single order."""

    order_id: str
    courier_id: str | None
    assigned: bool
    delayed_to_next_checkpoint: bool = False
    estimated_detour: float | None = None
    estimated_lateness: float | None = None
    score: float | None = None


@dataclass(slots=True)
class AssignmentResult:
    """Policy output for one checkpoint."""

    checkpoint_time: int
    policy_name: str
    decisions: list[AssignmentDecision]
    runtime_sec: float
    summary_metrics: dict[str, float] = field(default_factory=dict)

    @property
    def assigned_count(self) -> int:
        return sum(1 for decision in self.decisions if decision.assigned)
