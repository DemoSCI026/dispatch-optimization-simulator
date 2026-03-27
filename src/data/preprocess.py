"""Lightweight preprocessing helpers for checkpoint construction."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, TypedDict

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - optional dependency guard
    pd = None

from src.state.checkpoint_state import CheckpointState
from src.state.courier_state import CourierState
from src.state.order import Order, normalize_coordinate

DEFAULT_DATASET_DIR = Path(__file__).resolve().parent / "dataset"
DEFAULT_CHALLENGE_DIR = Path(__file__).resolve().parents[3] / "Meituan-INFORMS-TSL-Research-Challenge-main"
DEFAULT_WAYBILL_FILENAMES = (
    "all_waybill_info_meituan.csv",
    "all_waybill_info_meituan.csv.zip",
    "all_waybill_info_meituan_0322.csv",
    "all_waybill_info_meituan_0322.csv.zip",
)
DEFAULT_DISPATCH_WAYBILL_FILENAMES = ("dispatch_waybill_meituan.csv",)
DEFAULT_DISPATCH_RIDER_FILENAMES = ("dispatch_rider_meituan.csv",)


class RealDatasetPaths(TypedDict):
    """Resolved file paths for the required Meituan real-data inputs."""

    waybill: Path
    dispatch_waybill: Path
    dispatch_rider: Path


def normalize_time_columns(df: Any, columns: list[str]) -> Any:
    """Convert selected columns to nullable integer timestamps, treating 0 as missing."""
    pandas = _require_pandas()
    normalized = df.copy()
    for column in columns:
        if column in normalized.columns:
            numeric_values = pandas.to_numeric(normalized[column], errors="coerce")
            normalized[column] = numeric_values.where(numeric_values.ne(0)).astype("Int64")
    return normalized


def normalize_identifier_columns(df: Any, columns: list[str]) -> Any:
    """Normalize identifier-like columns to strings while preserving missing values."""
    normalized = df.copy()
    for column in columns:
        if column in normalized.columns:
            normalized[column] = normalized[column].apply(_normalize_identifier)
    return normalized


def normalize_coordinate_columns(df: Any, columns: list[str]) -> Any:
    """Normalize coordinate columns into plain latitude/longitude floats."""
    normalized = df.copy()
    for column in columns:
        if column in normalized.columns:
            normalized[column] = normalized[column].apply(_normalize_coordinate_value)
    return normalized


def parse_set_like_column(value: Any) -> list[str]:
    """Parse strings such as '(1, 2)' or '[1, 2]' into a list of ids."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, (tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    if not isinstance(value, (str, bytes, dict)):
        try:
            iterator = iter(value)
        except TypeError:
            iterator = None
        if iterator is not None:
            return [str(item).strip() for item in iterator if str(item).strip()]
    if _is_missing(value):
        return []

    text = str(value).strip()
    if not text or text in {"[]", "()"}:
        return []
    if (text.startswith("[") and text.endswith("]")) or (text.startswith("(") and text.endswith(")")):
        text = text[1:-1]
    if not text.strip():
        return []
    return [item.strip().strip("'\"") for item in text.split(",") if item.strip()]


def parse_order_set(value: Any) -> list[str]:
    """Parse a serialized order-id set such as '(1,2,3)' into a list[str]."""
    return parse_set_like_column(value)


def build_checkpoint_states(
    waybill_path: str | Path | None = None,
    dispatch_waybill_path: str | Path | None = None,
    dispatch_rider_path: str | Path | None = None,
    limit: int | None = None,
) -> list[CheckpointState]:
    """Load Meituan CSV files and convert them into CheckpointState objects."""
    from src.data.loader import load_dispatch_rider, load_dispatch_waybill, load_waybill_data

    if waybill_path is None and dispatch_waybill_path is None and dispatch_rider_path is None:
        resolved_paths = resolve_real_dataset_paths()
        resolved_waybill_path = resolved_paths["waybill"]
        resolved_dispatch_waybill_path = resolved_paths["dispatch_waybill"]
        resolved_dispatch_rider_path = resolved_paths["dispatch_rider"]
    else:
        resolved_waybill_path = _resolve_data_path(waybill_path, DEFAULT_WAYBILL_FILENAMES)
        resolved_dispatch_waybill_path = _resolve_data_path(
            dispatch_waybill_path,
            DEFAULT_DISPATCH_WAYBILL_FILENAMES,
        )
        resolved_dispatch_rider_path = _resolve_data_path(
            dispatch_rider_path,
            DEFAULT_DISPATCH_RIDER_FILENAMES,
        )

    waybill_df = load_waybill_data(str(resolved_waybill_path))
    dispatch_waybill_df = load_dispatch_waybill(str(resolved_dispatch_waybill_path))
    dispatch_rider_df = load_dispatch_rider(str(resolved_dispatch_rider_path))

    return build_checkpoint_states_from_frames(
        waybill_df=waybill_df,
        dispatch_waybill_df=dispatch_waybill_df,
        dispatch_rider_df=dispatch_rider_df,
        limit=limit,
    )


def resolve_real_dataset_paths(data_dir: str | Path | None = None) -> RealDatasetPaths:
    """Resolve the required real-data files from an explicit directory and bundled defaults."""
    search_dirs = _build_dataset_search_dirs(data_dir)

    resolved_waybill_path = _find_existing_dataset_path(search_dirs, DEFAULT_WAYBILL_FILENAMES)
    resolved_dispatch_waybill_path = _find_existing_dataset_path(
        search_dirs,
        DEFAULT_DISPATCH_WAYBILL_FILENAMES,
    )
    resolved_dispatch_rider_path = _find_existing_dataset_path(
        search_dirs,
        DEFAULT_DISPATCH_RIDER_FILENAMES,
    )

    missing: list[str] = []
    if resolved_waybill_path is None:
        missing.append(f"waybill ({', '.join(DEFAULT_WAYBILL_FILENAMES)})")
    if resolved_dispatch_waybill_path is None:
        missing.append(f"dispatch_waybill ({', '.join(DEFAULT_DISPATCH_WAYBILL_FILENAMES)})")
    if resolved_dispatch_rider_path is None:
        missing.append(f"dispatch_rider ({', '.join(DEFAULT_DISPATCH_RIDER_FILENAMES)})")

    if missing:
        searched_locations = ", ".join(str(path) for path in search_dirs)
        raise FileNotFoundError(
            "Missing required real-data files: "
            + "; ".join(missing)
            + f". Searched directories: {searched_locations}."
        )

    return {
        "waybill": resolved_waybill_path,
        "dispatch_waybill": resolved_dispatch_waybill_path,
        "dispatch_rider": resolved_dispatch_rider_path,
    }


def build_orders_master(waybill_df: Any) -> dict[str, Order]:
    """Build an order lookup table and keep the first valid record per order id."""
    prepared_waybills = _prepare_waybill_frame(waybill_df)
    orders_master: dict[str, Order] = {}

    for record in prepared_waybills.to_dict("records"):
        order_id = _optional_str(record.get("order_id"))
        if order_id is None or order_id in orders_master:
            continue
        if not _is_valid_order_record(record):
            continue
        orders_master[order_id] = _record_to_order(record)

    return orders_master


def build_checkpoint_index(
    dispatch_waybill_df: Any,
    dispatch_rider_df: Any,
) -> dict[tuple[str, int], dict[str, list[dict[str, Any]]]]:
    """Map (dt, dispatch_time) to order and rider records."""
    normalized_orders = _prepare_dispatch_waybill_frame(dispatch_waybill_df)
    normalized_riders = _prepare_dispatch_rider_frame(dispatch_rider_df)

    checkpoint_index: dict[tuple[str, int], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: {"order_records": [], "rider_records": []}
    )

    for record in normalized_orders.to_dict("records"):
        checkpoint_key = _build_checkpoint_key(record)
        if checkpoint_key is None:
            continue
        checkpoint_index[checkpoint_key]["order_records"].append(record)

    for record in normalized_riders.to_dict("records"):
        checkpoint_key = _build_checkpoint_key(record)
        if checkpoint_key is None:
            continue
        checkpoint_index[checkpoint_key]["rider_records"].append(record)

    return dict(sorted(checkpoint_index.items()))


def build_checkpoint_states_from_frames(
    waybill_df: Any,
    dispatch_waybill_df: Any,
    dispatch_rider_df: Any,
    limit: int | None = None,
) -> list[CheckpointState]:
    """Build checkpoint states from already loaded Meituan data frames."""
    prepared_waybills = _prepare_waybill_frame(waybill_df)
    if "order_id" not in prepared_waybills.columns:
        raise ValueError("waybill_df must contain an 'order_id' column")

    prepared_dispatch_waybills = _prepare_dispatch_waybill_frame(dispatch_waybill_df)
    prepared_dispatch_riders = _prepare_dispatch_rider_frame(dispatch_rider_df)
    orders_master = build_orders_master(prepared_waybills)

    def get_order(order_id: str) -> Order | None:
        return orders_master.get(str(order_id))

    checkpoints: list[CheckpointState] = []
    checkpoint_index = build_checkpoint_index(prepared_dispatch_waybills, prepared_dispatch_riders)

    for (dt, checkpoint_time), bucket in checkpoint_index.items():
        candidate_orders: list[Order] = []
        seen_orders: set[str] = set()
        for record in bucket["order_records"]:
            order_id = _optional_str(record.get("order_id"))
            if order_id is None or order_id in seen_orders:
                continue
            order = get_order(order_id)
            if order is None:
                continue
            candidate_orders.append(order)
            seen_orders.add(order_id)

        candidate_couriers: list[CourierState] = []
        for record in bucket["rider_records"]:
            courier = _record_to_courier(record, checkpoint_time, get_order)
            candidate_couriers.append(courier)

        if not candidate_orders and not candidate_couriers:
            continue

        checkpoints.append(
            CheckpointState(
                checkpoint_time=checkpoint_time,
                dt=dt,
                candidate_orders=candidate_orders,
                candidate_couriers=candidate_couriers,
            )
        )

        if limit is not None and len(checkpoints) >= limit:
            break

    return checkpoints


def _prepare_waybill_frame(waybill_df: Any) -> Any:
    prepared = normalize_identifier_columns(
        waybill_df,
        ["order_id", "waybill_id", "courier_id", "dt", "da_id", "area_id"],
    )
    prepared = normalize_time_columns(
        prepared,
        [
            "create_time",
            "push_time",
            "dispatch_time",
            "estimate_arrived_time",
            "estimate_meal_prepare_time",
            "order_push_time",
            "platform_order_time",
            "grab_time",
            "fetch_time",
            "arrive_time",
        ],
    )
    return normalize_coordinate_columns(
        prepared,
        [
            "pickup_lat",
            "pickup_lng",
            "dropoff_lat",
            "dropoff_lng",
            "sender_lat",
            "sender_lng",
            "recipient_lat",
            "recipient_lng",
            "grab_lat",
            "grab_lng",
        ],
    )


def _prepare_dispatch_waybill_frame(dispatch_waybill_df: Any) -> Any:
    prepared = normalize_identifier_columns(dispatch_waybill_df, ["order_id", "dt"])
    return normalize_time_columns(prepared, ["dispatch_time"])


def _prepare_dispatch_rider_frame(dispatch_rider_df: Any) -> Any:
    prepared = normalize_identifier_columns(dispatch_rider_df, ["courier_id", "dt"])
    prepared = normalize_time_columns(prepared, ["dispatch_time"])
    prepared = normalize_coordinate_columns(prepared, ["rider_lat", "rider_lng", "lat", "lng"])

    for column in ("courier_waybills", "order_ids"):
        if column in prepared.columns:
            prepared[column] = prepared[column].apply(parse_order_set)

    return prepared


def _record_to_order(record: dict[str, Any]) -> Order:
    return Order(
        order_id=str(record.get("order_id")),
        waybill_id=_optional_str(record.get("waybill_id")),
        dt=_coalesce(_optional_str(record.get("dt")), ""),
        area_id=_coalesce(_optional_str(record.get("area_id")), _optional_str(record.get("da_id"))),
        create_time=_coalesce(
            _optional_int(record.get("create_time")),
            _optional_int(record.get("platform_order_time")),
        ),
        push_time=_coalesce(_optional_int(record.get("push_time")), _optional_int(record.get("order_push_time"))),
        promise_time=_coalesce(
            _optional_int(record.get("promise_time")),
            _optional_int(record.get("estimate_arrived_time")),
        ),
        est_meal_ready_time=_coalesce(
            _optional_int(record.get("est_meal_ready_time")),
            _optional_int(record.get("estimate_meal_prepare_time")),
        ),
        pickup_lat=_coalesce(_optional_float(record.get("pickup_lat")), _optional_float(record.get("sender_lat")), 0.0),
        pickup_lng=_coalesce(_optional_float(record.get("pickup_lng")), _optional_float(record.get("sender_lng")), 0.0),
        dropoff_lat=_coalesce(
            _optional_float(record.get("dropoff_lat")),
            _optional_float(record.get("recipient_lat")),
            0.0,
        ),
        dropoff_lng=_coalesce(
            _optional_float(record.get("dropoff_lng")),
            _optional_float(record.get("recipient_lng")),
            0.0,
        ),
        is_prebook=_optional_bool(record.get("is_prebook")),
        is_weekend=_optional_bool(record.get("is_weekend")),
    )


def _record_to_courier(
    record: dict[str, Any],
    checkpoint_time: int,
    order_getter: Callable[[str], Order | None],
) -> CourierState:
    on_hand_order_ids = parse_set_like_column(
        _coalesce(record.get("courier_waybills"), record.get("order_ids"), [])
    )
    on_hand_orders = [order for order_id in on_hand_order_ids if (order := order_getter(order_id)) is not None]

    return CourierState(
        courier_id=str(record.get("courier_id")),
        dt=_coalesce(_optional_str(record.get("dt")), ""),
        checkpoint_time=checkpoint_time,
        lat=_coalesce(_optional_float(record.get("rider_lat")), _optional_float(record.get("lat")), 0.0),
        lng=_coalesce(_optional_float(record.get("rider_lng")), _optional_float(record.get("lng")), 0.0),
        on_hand_order_ids=on_hand_order_ids,
        on_hand_orders=on_hand_orders,
    )


def _resolve_data_path(
    path: str | Path | None,
    default_filenames: tuple[str, ...],
    search_dirs: tuple[Path, ...] | None = None,
) -> Path:
    if path is not None:
        resolved = Path(path).resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Input file does not exist: {resolved}")
        return resolved

    resolved = _find_existing_dataset_path(search_dirs or _build_dataset_search_dirs(), default_filenames)
    if resolved is not None:
        return resolved

    searched_locations = [str(path) for path in (search_dirs or _build_dataset_search_dirs())]
    raise FileNotFoundError(
        f"Could not resolve dataset file. Looked for {default_filenames} in {searched_locations}."
    )


def _build_dataset_search_dirs(data_dir: str | Path | None = None) -> tuple[Path, ...]:
    search_dirs: list[Path] = []
    if data_dir is not None:
        search_dirs.append(Path(data_dir).resolve())
    search_dirs.extend((DEFAULT_DATASET_DIR, DEFAULT_CHALLENGE_DIR))

    unique_dirs: list[Path] = []
    for path in search_dirs:
        if path not in unique_dirs:
            unique_dirs.append(path)
    return tuple(unique_dirs)


def _find_existing_dataset_path(search_dirs: tuple[Path, ...], candidates: tuple[str, ...]) -> Path | None:
    for base_dir in search_dirs:
        for filename in candidates:
            candidate = base_dir / filename
            if candidate.exists():
                return candidate
    return None


def _build_checkpoint_key(record: dict[str, Any]) -> tuple[str, int] | None:
    dispatch_time = _optional_int(record.get("dispatch_time"))
    dt = _coalesce(_optional_str(record.get("dt")), "")
    if dispatch_time is None or not dt:
        return None
    return (dt, dispatch_time)


def _is_valid_order_record(record: dict[str, Any]) -> bool:
    pickup_lat = _coalesce(_optional_float(record.get("pickup_lat")), _optional_float(record.get("sender_lat")))
    pickup_lng = _coalesce(_optional_float(record.get("pickup_lng")), _optional_float(record.get("sender_lng")))
    dropoff_lat = _coalesce(_optional_float(record.get("dropoff_lat")), _optional_float(record.get("recipient_lat")))
    dropoff_lng = _coalesce(_optional_float(record.get("dropoff_lng")), _optional_float(record.get("recipient_lng")))
    return all(value is not None for value in (pickup_lat, pickup_lng, dropoff_lat, dropoff_lng))


def _normalize_identifier(value: Any) -> str | None:
    if value is None or _is_missing(value):
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _normalize_coordinate_value(value: Any) -> float | None:
    if value is None or _is_missing(value):
        return None
    return normalize_coordinate(float(value))


def _optional_int(value: Any) -> int | None:
    if value is None or _is_missing(value):
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None or _is_missing(value):
        return None
    return float(value)


def _optional_str(value: Any) -> str | None:
    if value is None or _is_missing(value):
        return None
    return str(value)


def _optional_bool(value: Any) -> bool:
    if value is None or _is_missing(value):
        return False
    return bool(int(value)) if isinstance(value, (int, float, str)) and str(value).isdigit() else bool(value)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple, set, dict)):
        return False
    if not isinstance(value, (str, bytes)):
        try:
            if len(value) > 1:
                return False
        except TypeError:
            pass
    if pd is not None:
        if not pd.api.types.is_scalar(value):
            return False
        missing = pd.isna(value)
        if isinstance(missing, (list, tuple, set, dict)):
            return False
        if not isinstance(missing, (str, bytes)):
            try:
                if len(missing) > 1:
                    return False
            except TypeError:
                pass
        return bool(missing)
    return isinstance(value, float) and value != value


def _require_pandas() -> Any:
    if pd is None:
        raise ModuleNotFoundError(
            "pandas is required for real-data preprocessing. Install pandas to use src.data.preprocess."
        )
    return pd


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None
