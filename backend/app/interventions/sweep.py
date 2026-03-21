"""Experiment sweep orchestration.

Runs a combinatorial grid of (fraction x strategy x trial), collecting
failure scores and per-run metrics into a structured result set.
"""

from __future__ import annotations

import random as stdlib_random
from dataclasses import dataclass, asdict
from typing import Literal

import networkx as nx

from app.simulation.factory import get_engine
from app.metrics.metrics import (
    compute_baseline,
    failure_score,
    BaselineFingerprint,
)
from app.interventions.strategies import (
    Strategy,
    select_neurons,
    apply_dropout,
    apply_replacement,
    apply_graceful_fade,
)

Intervention = Literal["dropout", "replacement", "graceful"]


@dataclass
class SweepResult:
    fraction: float
    strategy: str
    intervention: str
    trial: int
    failure: float
    baseline: dict
    post: dict

    def to_dict(self) -> dict:
        return asdict(self)


def _neuron_groups(graph: nx.DiGraph) -> tuple[list[str], list[str]]:
    """Return (sensory_names, motor_names) from the graph."""
    sensory = [n for n, d in graph.nodes(data=True) if d.get("type") == "S"]
    motor = [n for n, d in graph.nodes(data=True) if d.get("type") == "M"]
    return sensory, motor


def run_sweep(
    graph: nx.DiGraph,
    fractions: list[float],
    strategies: list[Strategy],
    intervention: Intervention = "dropout",
    n_trials: int = 3,
    engine_name: str = "brian2",
    neuron_model: str = "lif",
    burn_in_ms: float = 2000.0,
    duration_ms: float = 5000.0,
    seed: int | None = None,
) -> list[SweepResult]:
    """Run the full replacement-sweep experimental protocol.

    For each (fraction, strategy, trial):
      1. Build a fresh simulation from the graph.
      2. Burn in, then capture baseline metrics.
      3. Apply the intervention.
      4. Run post-intervention, capture metrics.
      5. Compute failure score.
    """
    rng = stdlib_random.Random(seed)
    sensory, motor = _neuron_groups(graph)
    results: list[SweepResult] = []

    for frac in fractions:
        for strat in strategies:
            for trial in range(n_trials):
                engine = get_engine(engine_name)
                engine.build(graph, neuron_model=neuron_model)

                # Burn-in
                engine.run(burn_in_ms)

                # Baseline capture
                engine.run(duration_ms)
                bl_trains = engine.get_spike_trains()
                bl = compute_baseline(bl_trains, duration_ms, sensory, motor)

                # Intervention
                targets = select_neurons(graph, frac, strat, rng=rng)

                if intervention == "dropout":
                    apply_dropout(engine, targets)
                elif intervention == "replacement":
                    apply_replacement(engine, graph, targets, rng=rng)
                elif intervention == "graceful":
                    apply_graceful_fade(engine, graph, targets)
                else:
                    raise ValueError(f"Unknown intervention '{intervention}'")

                # Post-intervention capture
                engine.run(duration_ms)
                post_trains = engine.get_spike_trains()
                post = compute_baseline(post_trains, duration_ms, sensory, motor)

                score = failure_score(bl, post)

                results.append(SweepResult(
                    fraction=frac,
                    strategy=strat,
                    intervention=intervention,
                    trial=trial,
                    failure=score,
                    baseline=bl.to_dict(),
                    post=post.to_dict(),
                ))

    return results
