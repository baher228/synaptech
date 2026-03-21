"""Experiment sweep orchestration.

Runs a combinatorial grid of (fraction x trial), collecting
failure scores and per-run metrics into a structured result set.
"""

from __future__ import annotations

import random as stdlib_random
from dataclasses import asdict, dataclass
from typing import Literal

import networkx as nx

from app.interventions.fault_detection import FaultDetectionService
from app.interventions.replacement_service import ReplacementService
from app.metrics.metrics import compute_baseline, failure_score
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
