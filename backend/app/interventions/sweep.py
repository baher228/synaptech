"""Experiment sweep orchestration.

Runs a combinatorial grid of (fraction x trial), collecting
failure scores and per-run metrics into a structured result set.

Also provides :func:`run_replacement_sweep` for per-step metric capture
during one-by-one neuron replacement.
"""

from __future__ import annotations

import random as stdlib_random
from dataclasses import asdict, dataclass, field
from typing import Literal

import networkx as nx
import numpy as np

from app.connectome import B_CLASS_MOTOR_NEURONS
from app.interventions.fault_detection import FaultDetectionService
from app.interventions.replacement_service import ReplacementService
from app.metrics.metrics import (
    compute_baseline,
    failure_score,
    firing_rate_distribution,
    kuramoto_order_parameter,
    network_synchrony,
    pathway_fidelity,
    pca_attractor_deviation,
    voltage_state_entropy,
)
from app.simulation.factory import get_engine

Intervention = Literal["dropout", "replacement"]


@dataclass
class SweepResult:
    fraction: float
    target_selector: str
    intervention: str
    trial: int
    failure: float
    targeted_neurons: list[str]
    baseline: dict
    post: dict

    def to_dict(self) -> dict:
        return asdict(self)


def _neuron_groups(graph: nx.DiGraph) -> tuple[list[str], list[str]]:
    """Return (sensory_names, motor_names) from the graph."""
    sensory = [n for n, d in graph.nodes(data=True) if d.get("type") == "S"]
    motor = [n for n, d in graph.nodes(data=True) if d.get("type") == "M"]
    return sensory, motor


def _replace_targets_in_graph(
    graph: nx.DiGraph,
    targets: list[str],
    rng: stdlib_random.Random,
) -> nx.DiGraph:
    service = ReplacementService(graph)
    for target in targets:
        session = service.start_replacement(
            faulty_neuron=target,
            edge_order="random",
            seed=rng.randint(0, 10**9),
        )
        while session.status != "completed":
            session = service.step_replacement(
                session.session_id,
                edges_to_migrate=max(1, min(100, len(session.pending))),
            )
    return service.graph


def run_sweep(
    graph: nx.DiGraph,
    fractions: list[float],
    intervention: Intervention = "dropout",
    n_trials: int = 3,
    engine_name: str = "brian2",
    neuron_model: str = "lif",
    burn_in_ms: float = 2000.0,
    duration_ms: float = 5000.0,
    seed: int | None = None,
) -> list[SweepResult]:
    """Run sweep protocol for random-faulty target selection.

    For each (fraction, trial):
      1. Build a fresh simulation from the graph.
      2. Burn in, then capture baseline metrics.
      3. Apply intervention.
      4. Run post-intervention capture.
      5. Compute failure score.
    """
    rng = stdlib_random.Random(seed)
    detector = FaultDetectionService()
    sensory, motor = _neuron_groups(graph)
    results: list[SweepResult] = []

    for frac in fractions:
        for trial in range(n_trials):
            engine = get_engine(engine_name)
            engine.build(graph, neuron_model=neuron_model)

            # Baseline capture
            engine.run(burn_in_ms)
            engine.run(duration_ms)
            bl_trains = engine.get_spike_trains()
            baseline = compute_baseline(bl_trains, duration_ms, sensory, motor)

            targets = detector.select_targets_by_fraction(
                graph=graph,
                fraction=frac,
                seed=rng.randint(0, 10**9),
            )

            if intervention == "dropout":
                engine.silence_neurons(targets)
                engine.run(duration_ms)
                post_trains = engine.get_spike_trains()
            elif intervention == "replacement":
                replaced_graph = _replace_targets_in_graph(graph, targets, rng)
                post_engine = get_engine(engine_name)
                post_engine.build(replaced_graph, neuron_model=neuron_model)
                post_engine.run(burn_in_ms)
                post_engine.run(duration_ms)
                post_trains = post_engine.get_spike_trains()
            else:
                raise ValueError(f"Unknown intervention '{intervention}'")

            post = compute_baseline(post_trains, duration_ms, sensory, motor)
            score = failure_score(baseline, post)

            results.append(
                SweepResult(
                    fraction=frac,
                    target_selector="random_faulty",
                    intervention=intervention,
                    trial=trial,
                    failure=score,
                    targeted_neurons=targets,
                    baseline=baseline.to_dict(),
                    post=post.to_dict(),
                )
            )

    return results


# ------------------------------------------------------------------ #
#  Per-step replacement sweep
# ------------------------------------------------------------------ #

@dataclass
class StepMetrics:
    step_index: int
    neuron_being_replaced: str
    edges_migrated: int
    total_edges: int
    kuramoto_r: float
    pca_deviation: float
    pca_sigma: float
    voltage_entropy: float
    firing_rate_mean: float
    synchrony: float
    pathway_fidelity_val: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ReplacementTimeSeries:
    strategy: str
    neuron_model: str
    replacement_order: list[str]
    baseline: dict
    steps: list[StepMetrics] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "neuron_model": self.neuron_model,
            "replacement_order": self.replacement_order,
            "baseline": self.baseline,
            "steps": [s.to_dict() for s in self.steps],
        }


def run_replacement_sweep(
    graph: nx.DiGraph,
    target_neurons: list[str],
    engine_name: str = "brian2",
    neuron_model: str = "hh",
    burn_in_ms: float = 2000.0,
    baseline_ms: float = 5000.0,
    step_ms: float = 500.0,
    edges_per_step: int = 1,
    seed: int | None = None,
) -> ReplacementTimeSeries:
    """Run neuron-by-neuron replacement with per-step metric snapshots.

    1. Build engine, burn in, capture baseline (voltages + spikes).
    2. For each neuron in *target_neurons*:
       - start_replacement (allocates engine slot)
       - For each edge migration batch: mutate engine, run, measure
    3. Return time-series of metrics at every step.
    """
    rng = stdlib_random.Random(seed)
    engine = get_engine(engine_name)
    working_graph = graph.copy()
    engine.build(working_graph, neuron_model=neuron_model)

    sensory, motor = _neuron_groups(graph)
    b_class = list(B_CLASS_MOTOR_NEURONS)

    # Burn-in + baseline capture
    engine.run(burn_in_ms)
    engine.run(baseline_ms)
    bl_trains = engine.get_spike_trains()
    bl_voltages = engine.get_voltage_matrix(window_ms=baseline_ms)
    baseline = compute_baseline(
        bl_trains, baseline_ms, sensory, motor,
        voltages=bl_voltages, b_class_names=b_class,
    )

    # Replacement loop
    service = ReplacementService(graph=working_graph, engine=engine)
    steps: list[StepMetrics] = []
    global_step = 0

    for neuron_name in target_neurons:
        session = service.start_replacement(
            faulty_neuron=neuron_name,
            edge_order="random",
            seed=rng.randint(0, 10**9),
        )
        total_edges = len(session.pending) + len(session.completed)

        while session.status != "completed":
            session = service.step_replacement(
                session.session_id,
                edges_to_migrate=edges_per_step,
            )
            engine.run(step_ms)

            trains = engine.get_spike_trains()
            voltages = engine.get_voltage_matrix(window_ms=step_ms)

            rates = firing_rate_distribution(trains, step_ms)
            rate_vals = np.array(list(rates.values()))
            pca_d, pca_s = pca_attractor_deviation(bl_voltages, voltages)

            steps.append(StepMetrics(
                step_index=global_step,
                neuron_being_replaced=neuron_name,
                edges_migrated=len(session.completed),
                total_edges=total_edges,
                kuramoto_r=kuramoto_order_parameter(trains, b_class, step_ms),
                pca_deviation=pca_d,
                pca_sigma=pca_s,
                voltage_entropy=voltage_state_entropy(voltages, step_ms),
                firing_rate_mean=float(np.mean(rate_vals)) if len(rate_vals) else 0.0,
                synchrony=network_synchrony(trains, step_ms),
                pathway_fidelity_val=pathway_fidelity(trains, sensory, motor, step_ms),
            ))
            global_step += 1

    return ReplacementTimeSeries(
        strategy="one_by_one_replacement",
        neuron_model=neuron_model,
        replacement_order=target_neurons,
        baseline=baseline.to_dict(),
        steps=steps,
    )
