"""Run the age- and area-aware delayable insertion heuristic."""

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
from src.policies.age_area_delay_insertion import AgeAreaDelayInsertionPolicy
from src.routing.routing_engine import RoutingEngine
from src.simulator.simulator import Simulator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the age- and area-aware delayable insertion dispatch heuristic."
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
        "--delay-threshold",
        type=float,
        default=0.45,
        help="Delay when the delayable score is at least this value.",
    )
    parser.add_argument(
        "--min-slack-to-delay",
        type=int,
        default=900,
        help="Orders with smaller remaining promise slack are assigned immediately.",
    )
    parser.add_argument(
        "--max-age-to-delay",
        type=int,
        default=600,
        help="Orders older than this many seconds are assigned immediately.",
    )
    parser.add_argument(
        "--max-backlog-to-delay",
        type=int,
        default=380,
        help="Disable additional delaying once backlog reaches this size.",
    )
    args = parser.parse_args()

    checkpoints = load_checkpoints(
        data_dir=args.data_dir,
        limit=args.limit,
        strict_real_data=args.strict_real_data,
    )
    policy = AgeAreaDelayInsertionPolicy(
        delay_threshold=args.delay_threshold,
        min_slack_to_delay=args.min_slack_to_delay,
        max_age_to_delay=args.max_age_to_delay,
        max_backlog_to_delay=args.max_backlog_to_delay,
    )
    cost_engine = CostEngine(alpha=1.0, beta=0.5, gamma=0.2)
    routing_engine = RoutingEngine(cost_engine)
    simulator = Simulator()

    results = simulator.run(checkpoints, policy, cost_engine, routing_engine)
    summary = simulator.summarize(results)
    print(
        "delay_threshold={delay_threshold:.2f} min_slack_to_delay={min_slack_to_delay} "
        "max_age_to_delay={max_age_to_delay} max_backlog_to_delay={max_backlog_to_delay}".format(
            delay_threshold=args.delay_threshold,
            min_slack_to_delay=args.min_slack_to_delay,
            max_age_to_delay=args.max_age_to_delay,
            max_backlog_to_delay=args.max_backlog_to_delay,
        )
    )
    print_run_summary(policy.name, results, summary)


if __name__ == "__main__":
    main()
