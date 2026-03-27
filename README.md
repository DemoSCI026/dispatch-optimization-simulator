# TSL Dispatch Optimization

This repository is a first-pass code skeleton for a checkpoint-based dispatch replay simulator built around the TSL-Meituan challenge data. The current goal is not full production fidelity, but a clean and extensible MVP that lets us compare assignment heuristics quickly.

## Structure

```text
src/
  data/          CSV loading and lightweight preprocessing
  state/         Core dataclasses for orders, couriers, checkpoints, and results
  routing/       Simplified route representation and insertion logic
  cost/          Shared pair-cost calculations
  policies/      Nearest, score-based, and route-insertion heuristics
  simulator/     Checkpoint replay loop
  evaluation/    Aggregate metrics
experiments/     Example entry points for running the pipeline
```

## Policies

- `NearestGreedyPolicy`: assigns each order to the courier nearest to the pickup point.
- `ScoreGreedyPolicy`: assigns with a weighted combination of detour, lateness risk, and workload.
- `RouteInsertionPolicy`: builds a simplified route per courier and greedily inserts new orders into the least-cost position.

## Running Experiments

Run with mock data:

```bash
python experiments/run_nearest.py
python experiments/run_score.py
python experiments/run_insertion.py
```

Run with real data:

```bash
python experiments/run_insertion.py --data-dir Meituan-INFORMS-TSL-Research-Challenge-main --limit 10
```

The scripts automatically fall back to mock checkpoints when required CSV files are missing or parsing fails.

## Current Simplifications

- Checkpoint states are built from raw dispatch records with a lightweight join to the waybill table.
- Distances use straight-line haversine distance instead of a road-network ETA model.
- Initial routes assume on-hand orders can still be represented as pickup-plus-dropoff stop pairs.
- The simulator replays checkpoints independently and records policy outputs; it does not yet propagate state transitions across future checkpoints.
