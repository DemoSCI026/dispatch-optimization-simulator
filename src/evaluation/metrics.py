"""Evaluation metrics for checkpoint assignment results."""

from __future__ import annotations

from statistics import mean

from src.state.assignment_result import AssignmentDecision, AssignmentResult



def compute_avg_detour(results: list[AssignmentResult]) -> float:
    detours = [
        decision.estimated_detour
        for decision in _iter_assigned_decisions(results)
        if decision.estimated_detour is not None
    ]
    return mean(detours) if detours else 0.0



def compute_assigned_ratio(results: list[AssignmentResult]) -> float:
    total_decisions = sum(len(result.decisions) for result in results)
    total_assigned = sum(1 for _ in _iter_assigned_decisions(results))
    return (total_assigned / total_decisions) if total_decisions else 0.0



def compute_avg_runtime(results: list[AssignmentResult]) -> float:
    runtimes = [result.runtime_sec for result in results]
    return mean(runtimes) if runtimes else 0.0



def compute_avg_backlog_size(results: list[AssignmentResult]) -> float:
    backlog_sizes = [float(result.summary_metrics.get("backlog_size", 0.0)) for result in results]
    return mean(backlog_sizes) if backlog_sizes else 0.0



def compute_avg_courier_load(results: list[AssignmentResult]) -> float:
    loads = [float(result.summary_metrics.get("avg_courier_load", 0.0)) for result in results]
    return mean(loads) if loads else 0.0



def compute_total_assigned_orders(results: list[AssignmentResult]) -> float:
    return float(
        sum(
            int(result.summary_metrics.get("assigned_count", result.assigned_count))
            for result in results
        )
    )



def compute_total_unassigned_orders(results: list[AssignmentResult]) -> float:
    return float(
        sum(
            int(result.summary_metrics.get("unassigned_count", len(result.decisions) - result.assigned_count))
            for result in results
        )
    )



def compute_completion_ratio(results: list[AssignmentResult]) -> float:
    if not results:
        return 0.0

    latest_metrics = results[-1].summary_metrics
    total_seen_orders = float(latest_metrics.get("total_seen_orders", 0.0))
    completed_orders = float(latest_metrics.get("completed_orders", 0.0))
    return (completed_orders / total_seen_orders) if total_seen_orders else 0.0



def _iter_assigned_decisions(results: list[AssignmentResult]) -> list[AssignmentDecision]:
    return [
        decision
        for result in results
        for decision in result.decisions
        if decision.assigned
    ]
