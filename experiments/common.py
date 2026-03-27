"""Shared experiment helpers for running the MVP pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.state.checkpoint_state import CheckpointState
from src.state.courier_state import CourierState
from src.state.order import Order



def load_checkpoints(
    data_dir: str | None = None,
    limit: int = 5,
    strict_real_data: bool = False,
) -> list[CheckpointState]:
    """Load real checkpoints when possible and make fallback behavior explicit."""
    real_data_requested = data_dir is not None or strict_real_data
    try:
        from src.data.preprocess import build_checkpoint_states, resolve_real_dataset_paths
    except ModuleNotFoundError as exc:
        return _fallback_to_mock(
            reason=f"real-data dependencies are unavailable ({exc})",
            strict_real_data=strict_real_data,
            real_data_requested=real_data_requested,
            limit=limit,
            error=exc,
        )

    try:
        resolved_paths = resolve_real_dataset_paths(data_dir)
        _print_real_data_paths(resolved_paths)
        checkpoints = build_checkpoint_states(
            waybill_path=resolved_paths["waybill"],
            dispatch_waybill_path=resolved_paths["dispatch_waybill"],
            dispatch_rider_path=resolved_paths["dispatch_rider"],
            limit=limit,
        )
        if not checkpoints:
            raise ValueError("real-data preprocessing returned 0 checkpoints.")

        _print_checkpoint_debug_summary(
            checkpoints,
            real_data_requested=real_data_requested,
            limit=limit,
        )
        return checkpoints
    except Exception as exc:
        return _fallback_to_mock(
            reason=str(exc),
            strict_real_data=strict_real_data,
            real_data_requested=real_data_requested,
            limit=limit,
            error=exc,
        )



def build_mock_checkpoints() -> list[CheckpointState]:
    """Create a tiny synthetic workload that exercises all policies."""
    dt = "20221017"
    checkpoint_time = 1_700_000_000

    existing_order = Order(
        order_id="existing-1",
        waybill_id="w-existing-1",
        dt=dt,
        area_id="A",
        create_time=checkpoint_time - 900,
        push_time=checkpoint_time - 840,
        promise_time=checkpoint_time + 1500,
        est_meal_ready_time=checkpoint_time - 300,
        pickup_lat=31.2295,
        pickup_lng=121.4750,
        dropoff_lat=31.2330,
        dropoff_lng=121.4850,
        is_weekend=False,
    )
    order_1 = Order(
        order_id="order-1",
        waybill_id="w-1",
        dt=dt,
        area_id="A",
        create_time=checkpoint_time - 120,
        push_time=checkpoint_time - 90,
        promise_time=checkpoint_time + 1800,
        est_meal_ready_time=checkpoint_time + 180,
        pickup_lat=31.2304,
        pickup_lng=121.4737,
        dropoff_lat=31.2253,
        dropoff_lng=121.4815,
    )
    order_2 = Order(
        order_id="order-2",
        waybill_id="w-2",
        dt=dt,
        area_id="B",
        create_time=checkpoint_time - 60,
        push_time=checkpoint_time - 30,
        promise_time=checkpoint_time + 2100,
        est_meal_ready_time=checkpoint_time + 120,
        pickup_lat=31.2360,
        pickup_lng=121.4680,
        dropoff_lat=31.2405,
        dropoff_lng=121.4798,
    )
    order_3 = Order(
        order_id="order-3",
        waybill_id="w-3",
        dt=dt,
        area_id="B",
        create_time=checkpoint_time + 420,
        push_time=checkpoint_time + 450,
        promise_time=checkpoint_time + 2600,
        est_meal_ready_time=checkpoint_time + 540,
        pickup_lat=31.2210,
        pickup_lng=121.4690,
        dropoff_lat=31.2185,
        dropoff_lng=121.4595,
    )

    courier_1 = CourierState(
        courier_id="courier-1",
        dt=dt,
        checkpoint_time=checkpoint_time,
        lat=31.2290,
        lng=121.4710,
        on_hand_order_ids=[],
        on_hand_orders=[],
    )
    courier_2 = CourierState(
        courier_id="courier-2",
        dt=dt,
        checkpoint_time=checkpoint_time,
        lat=31.2350,
        lng=121.4820,
        on_hand_order_ids=[existing_order.order_id],
        on_hand_orders=[existing_order],
    )

    checkpoint_1 = CheckpointState(
        checkpoint_time=checkpoint_time,
        dt=dt,
        candidate_orders=[order_1, order_2],
        candidate_couriers=[courier_1, courier_2],
    )

    checkpoint_2 = CheckpointState(
        checkpoint_time=checkpoint_time + 600,
        dt=dt,
        candidate_orders=[order_3],
        candidate_couriers=[
            CourierState(
                courier_id="courier-1",
                dt=dt,
                checkpoint_time=checkpoint_time + 600,
                lat=31.2280,
                lng=121.4705,
                on_hand_order_ids=[],
                on_hand_orders=[],
            ),
            CourierState(
                courier_id="courier-3",
                dt=dt,
                checkpoint_time=checkpoint_time + 600,
                lat=31.2200,
                lng=121.4630,
                on_hand_order_ids=[],
                on_hand_orders=[],
            ),
        ],
    )

    return [checkpoint_1, checkpoint_2]



def print_run_summary(policy_name: str, results, summary: dict[str, float]) -> None:
    """Print a compact console summary for experiments."""
    print(f"policy={policy_name}")
    print(f"checkpoints={len(results)}")
    print(
        "avg_detour={avg_detour:.3f}km assigned_ratio={assigned_ratio:.3f} avg_runtime={avg_runtime:.6f}s".format(
            **summary
        )
    )
    print(
        "avg_backlog_size={avg_backlog_size:.3f} avg_courier_load={avg_courier_load:.3f} completion_ratio={completion_ratio:.3f}".format(
            **summary
        )
    )
    print(
        "completed_orders={completed_orders:.0f} late_orders={late_orders:.0f} expired_orders={expired_orders:.0f} remaining_backlog={remaining_backlog:.0f}".format(
            **summary
        )
    )
    print(
        "total_assigned_orders={total_assigned_orders:.0f} total_unassigned_orders={total_unassigned_orders:.0f}".format(
            **summary
        )
    )

    if not results:
        return

    first_result = results[0]
    print(f"first_checkpoint={first_result.checkpoint_time} decisions={len(first_result.decisions)}")
    for decision in first_result.decisions[:5]:
        score = "NA" if decision.score is None else f"{decision.score:.3f}"
        print(
            f"  order={decision.order_id} courier={decision.courier_id} "
            f"assigned={decision.assigned} score={score}"
        )



def _fallback_to_mock(
    reason: str,
    *,
    strict_real_data: bool,
    real_data_requested: bool,
    limit: int,
    error: Exception | None = None,
) -> list[CheckpointState]:
    if strict_real_data:
        raise RuntimeError(f"Unable to load real checkpoints: {reason}") from error

    print(f"Falling back to mock checkpoints because {_format_reason(reason)}")
    checkpoints = build_mock_checkpoints()
    _print_checkpoint_debug_summary(
        checkpoints,
        real_data_requested=real_data_requested,
        limit=limit,
    )
    return checkpoints


def _print_real_data_paths(resolved_paths: dict[str, Path]) -> None:
    source_dirs = list(dict.fromkeys(str(path.parent) for path in resolved_paths.values()))
    print(f"Loading real checkpoints from: {', '.join(source_dirs)}")
    print("Found files:")
    print(f"  waybill={resolved_paths['waybill']}")
    print(f"  dispatch_waybill={resolved_paths['dispatch_waybill']}")
    print(f"  dispatch_rider={resolved_paths['dispatch_rider']}")


def _print_checkpoint_debug_summary(
    checkpoints: list[CheckpointState],
    *,
    real_data_requested: bool,
    limit: int | None,
) -> None:
    print(f"Loaded checkpoints: {len(checkpoints)}")
    if not checkpoints:
        return

    first_checkpoint = checkpoints[0]
    print(f"First checkpoint_time: {first_checkpoint.checkpoint_time}")
    print(
        "First checkpoint candidates: "
        f"orders={len(first_checkpoint.candidate_orders)} "
        f"couriers={len(first_checkpoint.candidate_couriers)}"
    )

    if real_data_requested and len(checkpoints) == 2 and (limit is None or limit > 2):
        print(
            "Warning: only 2 checkpoints were loaded after requesting real data. "
            "Mock data may have been used or real-data discovery/loading may have failed."
        )


def _format_reason(reason: str) -> str:
    return reason if reason.endswith((".", "!", "?")) else f"{reason}."
