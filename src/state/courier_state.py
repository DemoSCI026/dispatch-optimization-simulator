"""Courier state used as policy input at a dispatch checkpoint."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.state.order import Order, normalize_coordinate

if TYPE_CHECKING:
    from src.routing.route_plan import RoutePlan


@dataclass(slots=True)
class CourierState:
    """Snapshot of a courier at one checkpoint."""

    courier_id: str
    dt: str
    checkpoint_time: int
    lat: float
    lng: float
    on_hand_order_ids: list[str] = field(default_factory=list)
    on_hand_orders: list[Order] = field(default_factory=list)
    route_plan: RoutePlan | None = None
    completed_order_ids: list[str] = field(default_factory=list)
    last_updated_time: int | None = None

    def __post_init__(self) -> None:
        self.courier_id = str(self.courier_id)
        self.lat = normalize_coordinate(self.lat)
        self.lng = normalize_coordinate(self.lng)
        self.on_hand_order_ids = [str(order_id) for order_id in self.on_hand_order_ids]
        self.completed_order_ids = [str(order_id) for order_id in self.completed_order_ids]
        if self.last_updated_time is None:
            self.last_updated_time = self.checkpoint_time

    @property
    def current_load(self) -> int:
        return len(self.on_hand_orders)

    def can_accept_order(self, max_on_hand_orders: int | None) -> bool:
        return max_on_hand_orders is None or self.current_load < max_on_hand_orders

    def sync_observation(self, dt: str, checkpoint_time: int, lat: float, lng: float) -> None:
        self.dt = dt
        self.checkpoint_time = checkpoint_time
        self.lat = normalize_coordinate(lat)
        self.lng = normalize_coordinate(lng)
        self.last_updated_time = checkpoint_time

    def copy_for_policy(self) -> "CourierState":
        """Build a lightweight working copy so policies do not mutate simulator state."""
        return CourierState(
            courier_id=self.courier_id,
            dt=self.dt,
            checkpoint_time=self.checkpoint_time,
            lat=self.lat,
            lng=self.lng,
            on_hand_order_ids=list(self.on_hand_order_ids),
            on_hand_orders=list(self.on_hand_orders),
            route_plan=self.route_plan.copy() if self.route_plan is not None else None,
            completed_order_ids=list(self.completed_order_ids),
            last_updated_time=self.last_updated_time,
        )

    def add_order(self, order: Order) -> None:
        if order.order_id in self.on_hand_order_ids:
            return
        self.on_hand_order_ids.append(order.order_id)
        self.on_hand_orders.append(order)

    def remove_order(self, order_id: str) -> None:
        self.on_hand_order_ids = [existing_id for existing_id in self.on_hand_order_ids if existing_id != order_id]
        self.on_hand_orders = [order for order in self.on_hand_orders if order.order_id != order_id]

        if self.route_plan is not None:
            self.route_plan.stops = [stop for stop in self.route_plan.stops if stop.order_id != order_id]
            self.route_plan.total_distance = None
            self.route_plan.total_eta = None

    def mark_order_completed(self, order_id: str) -> None:
        self.remove_order(order_id)
        if order_id not in self.completed_order_ids:
            self.completed_order_ids.append(order_id)
