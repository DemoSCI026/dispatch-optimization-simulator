"""Checkpoint-level state passed into dispatch policies."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.state.courier_state import CourierState
from src.state.order import Order


@dataclass(slots=True)
class CheckpointState:
    """Orders and couriers waiting for assignment at one checkpoint."""

    checkpoint_time: int
    dt: str
    candidate_orders: list[Order] = field(default_factory=list)
    candidate_couriers: list[CourierState] = field(default_factory=list)
    max_on_hand_orders: int | None = None

    @property
    def num_orders(self) -> int:
        return len(self.candidate_orders)

    @property
    def num_couriers(self) -> int:
        return len(self.candidate_couriers)
