"""Run the pooling-aware delay insertion dispatch heuristic."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
for path in (CURRENT_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import load_checkpoints, print_run_summary
from src.cost.cost_engine import CostEngine
from src.policies.pooling_aware_delay_insertion import PoolingAwareDelayInsertionPolicy
from src.routing.routing_engine import RoutingEngine
from src.simulator.simulator import Simulator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the pooling-aware delay insertion dispatch heuristic."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory to search first for Meituan CSV files before falling back to bundled defaults.",
    )
    parser.add_argument("--limit", type=int, default=5, help="Maximum number of checkpoints to replay.")
    parser.add_argument(
        "--strict-real-data",
        action="store_true",
        help="Raise instead of falling back to mock checkpoints when real-data loading fails.",
    )
    parser.add_argument(
        "--detour-scale",
        type=float,
        default=3.0,
        help="Normalize best insertion detour by this scale before badness thresholds are applied.",
    )
    parser.add_argument(
        "--pickup-scale",
        type=float,
        default=5.0,
        help="Normalize pickup neighbor counts by this scale.",
    )
    parser.add_argument(
        "--delivery-scale",
        type=float,
        default=5.0,
        help="Normalize delivery neighbor counts by this scale.",
    )
    parser.add_argument(
        "--slack-scale",
        type=float,
        default=1800.0,
        help="Normalize promise slack by this scale in seconds.",
    )
    parser.add_argument(
        "--w-spatial",
        type=float,
        default=0.6,
        help="Weight on spatial pooling potential.",
    )
    parser.add_argument(
        "--w-temporal",
        type=float,
        default=0.4,
        help="Weight on temporal pooling potential.",
    )
    parser.add_argument(
        "--min-slack",
        type=int,
        default=300,
        help="Orders with less remaining promise slack are assigned immediately.",
    )
    parser.add_argument(
        "--good-detour-threshold",
        type=float,
        default=0.3,
        help="Assign immediately when detour badness falls below this threshold.",
    )
    parser.add_argument(
        "--bad-detour-threshold",
        type=float,
        default=0.6,
        help="Only allow delay when detour badness exceeds this threshold.",
    )
    parser.add_argument(
        "--pooling-threshold",
        type=float,
        default=0.5,
        help="Delay only when pooling potential exceeds this threshold.",
    )
    parser.add_argument(
        "--max-delay-ratio",
        type=float,
        default=0.3,
        help="Maximum fraction of orders that can be delayed in a checkpoint.",
    )
    args = parser.parse_args()

    checkpoints = load_checkpoints(
        data_dir=args.data_dir,
        limit=args.limit,
        strict_real_data=args.strict_real_data,
    )
    policy = PoolingAwareDelayInsertionPolicy(
        detour_scale=args.detour_scale,
        pickup_scale=args.pickup_scale,
        delivery_scale=args.delivery_scale,
        slack_scale=args.slack_scale,
        w_spatial=args.w_spatial,
        w_temporal=args.w_temporal,
        min_slack=args.min_slack,
        good_detour_threshold=args.good_detour_threshold,
        bad_detour_threshold=args.bad_detour_threshold,
        pooling_threshold=args.pooling_threshold,
        max_delay_ratio=args.max_delay_ratio,
    )
    cost_engine = CostEngine(alpha=1.0, beta=0.5, gamma=0.2)
    routing_engine = RoutingEngine(cost_engine)
    simulator = Simulator()

    results = simulator.run(checkpoints, policy, cost_engine, routing_engine)
    summary = simulator.summarize(results)
    total_orders = sum(result.summary_metrics.get("total_orders", 0.0) for result in results)
    delayed_orders = sum(result.summary_metrics.get("delayed_orders", 0.0) for result in results)
    delay_ratio = (delayed_orders / total_orders) if total_orders else 0.0

    print(
        "detour_scale={detour_scale:.2f} pickup_scale={pickup_scale:.2f} "
        "delivery_scale={delivery_scale:.2f} slack_scale={slack_scale:.0f} "
        "w_spatial={w_spatial:.2f} w_temporal={w_temporal:.2f} "
        "min_slack={min_slack} good_detour_threshold={good_detour_threshold:.2f} "
        "bad_detour_threshold={bad_detour_threshold:.2f} pooling_threshold={pooling_threshold:.2f} "
        "max_delay_ratio={max_delay_ratio:.2f}".format(
            detour_scale=args.detour_scale,
            pickup_scale=args.pickup_scale,
            delivery_scale=args.delivery_scale,
            slack_scale=args.slack_scale,
            w_spatial=args.w_spatial,
            w_temporal=args.w_temporal,
            min_slack=args.min_slack,
            good_detour_threshold=args.good_detour_threshold,
            bad_detour_threshold=args.bad_detour_threshold,
            pooling_threshold=args.pooling_threshold,
            max_delay_ratio=args.max_delay_ratio,
        )
    )
    print(
        f"total_orders={int(total_orders)} delayed_orders={int(delayed_orders)} delay_ratio={delay_ratio:.3f}"
    )
    print_run_summary(policy.name, results, summary)


if __name__ == "__main__":
    main()
