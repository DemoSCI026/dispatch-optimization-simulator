# Dispatch Optimization Simulator

## Overview

This project implements a checkpoint-based dispatch simulation framework for the order–courier assignment problem, based on the Meituan TSL Challenge dataset.

The objective is to evaluate different heuristic policies for dynamic dispatch under a many-to-many setting, with multiple competing objectives including routing efficiency, timeliness, and system throughput.

---

## Problem Setting

At each dispatch checkpoint, the system receives:

* A set of pending orders
* A set of available couriers
* Current courier states (location and ongoing routes)

The task is to assign orders to couriers.

This can be formulated as a dynamic combinatorial optimization problem with objectives such as:

* minimizing detour distance
* reducing lateness risk
* balancing courier workload
* maintaining system throughput

---

## System Structure

```text
src/
  data/          Data loading and preprocessing
  state/         Core data structures (orders, couriers, checkpoints)
  routing/       Route representation and insertion evaluation
  cost/          Cost computation utilities
  policies/      Dispatch policies
  simulator/     Checkpoint replay engine
  evaluation/    Metrics aggregation
experiments/     Entry points for running experiments
```

The simulator replays dispatch decisions across checkpoints and records policy-level performance metrics.

---

## Implemented Policies

### Greedy Policies

* `NearestGreedyPolicy`
  Assigns each order to the geographically nearest courier.

* `ScoreGreedyPolicy`
  Uses a weighted combination of detour, lateness risk, and workload.

---

### Routing-Aware Policy

* `RouteInsertionPolicy`
  Maintains a route for each courier and inserts new orders at positions that minimize incremental cost.

---

### Delay-Based Policy

* Delay mechanisms are introduced to defer assignment when the current match is suboptimal, allowing potential improvement in future checkpoints.

---

### Pooling-Aware Extension

* A pooling score is defined based on:

  * spatial proximity (pickup and delivery clustering)
  * temporal slack (remaining time before deadline)

* Orders with high pooling potential may be delayed to enable more efficient routing in future assignments.

---

### Global Control

* A delay budget is applied to limit the proportion of delayed orders, preventing excessive backlog accumulation and ensuring system stability.

---

## Key Observations

* Routing-aware insertion significantly reduces detour compared to greedy baselines.
* Delay strategies improve routing quality but introduce a tradeoff with system throughput.
* Pooling-aware heuristics further improve routing efficiency by enabling implicit order clustering.
* Without global constraints, delay decisions can lead to system instability.
* Profiling shows that the main computational cost lies in route insertion evaluation.

---

## Running Experiments

Run baseline experiments:

```bash
python -m experiments.run_nearest
python -m experiments.run_score
python -m experiments.run_insertion
```

Run with real data:

```bash
python -m experiments.run_insertion --data-dir <path_to_dataset> --limit 20
```

---

## Notes

* Distance is approximated using haversine distance.
* The routing model is simplified and does not incorporate road network constraints.
* The simulator focuses on policy comparison rather than production deployment.
