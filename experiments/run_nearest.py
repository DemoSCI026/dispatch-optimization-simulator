"""Run the nearest-greedy baseline."""

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
from src.policies.nearest_greedy import NearestGreedyPolicy
from src.simulator.simulator import Simulator



def main() -> None:
    parser = argparse.ArgumentParser(description="Run the nearest-greedy dispatch baseline.")
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
    args = parser.parse_args()

    checkpoints = load_checkpoints(
        data_dir=args.data_dir,
        limit=args.limit,
        strict_real_data=args.strict_real_data,
    )
    policy = NearestGreedyPolicy()
    cost_engine = CostEngine(alpha=1.0, beta=0.0, gamma=0.0)
    simulator = Simulator()

    results = simulator.run(checkpoints, policy, cost_engine)
    summary = simulator.summarize(results)
    print_run_summary(policy.name, results, summary)


if __name__ == "__main__":
    main()
