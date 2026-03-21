"""Neuron selection strategies and intervention application.

All functions are engine-agnostic — they operate on the NetworkX graph to
choose targets, then call protocol methods on whichever engine is active.
"""

from __future__ import annotations

import random as stdlib_random
from typing import Literal

import networkx as nx
import numpy as np

from app.simulation.protocol import SimulationEngine

Strategy = Literal[
    "random",
    "hub_targeted",
    "hub_sparing",
    "type_sensory",
    "type_motor",
    "type_interneuron",
    "region_head",
    "region_body",
    "region_tail",
]

ALL_STRATEGIES: list[Strategy] = [
    "random",
    "hub_targeted",
    "hub_sparing",
    "type_sensory",
    "type_motor",
    "type_interneuron",
    "region_head",
    "region_body",
    "region_tail",
]


# ------------------------------------------------------------------ #
#  Selection
# ------------------------------------------------------------------ #

def select_neurons(
    graph: nx.DiGraph,
    fraction: float,
    strategy: Strategy,
    rng: stdlib_random.Random | None = None,
) -> list[str]:
    """Return neuron names to target for the given strategy and fraction."""
    rng = rng or stdlib_random.Random()
    names = list(graph.nodes)
    n_target = max(1, int(len(names) * fraction))

    if strategy == "random":
        return rng.sample(names, min(n_target, len(names)))

    if strategy == "hub_targeted":
        ranked = sorted(
            names,
            key=lambda n: graph.nodes[n].get("degree_centrality", 0),
            reverse=True,
        )
        return ranked[:n_target]

    if strategy == "hub_sparing":
        ranked = sorted(
            names,
            key=lambda n: graph.nodes[n].get("degree_centrality", 0),
        )
        return ranked[:n_target]

    # Type-targeted
    type_map = {"type_sensory": "S", "type_motor": "M", "type_interneuron": "I"}
    if strategy in type_map:
        pool = [n for n in names if graph.nodes[n].get("type") == type_map[strategy]]
        return rng.sample(pool, min(n_target, len(pool)))

    # Region-targeted
    region_map = {"region_head": "head", "region_body": "body", "region_tail": "tail"}
    if strategy in region_map:
        pool = [n for n in names if graph.nodes[n].get("region") == region_map[strategy]]
        return rng.sample(pool, min(n_target, len(pool)))

    raise ValueError(f"Unknown strategy '{strategy}'")


# ------------------------------------------------------------------ #
#  Intervention application
# ------------------------------------------------------------------ #

def apply_dropout(
    engine: SimulationEngine,
    neuron_names: list[str],
) -> None:
    """Silence neurons — simulates neurodegeneration / death."""
    engine.silence_neurons(neuron_names)


def apply_replacement(
    engine: SimulationEngine,
    graph: nx.DiGraph,
    neuron_names: list[str],
    connectivity_restore: float = 0.5,
    rng: stdlib_random.Random | None = None,
) -> None:
    """Remove neurons then re-insert with partially restored connectivity.

    *connectivity_restore* in [0, 1] controls what fraction of the
    original outgoing synapse weight is restored (simulating imperfect
    wiring of a replacement neuron).
    """
    rng = rng or stdlib_random.Random()

    engine.silence_neurons(neuron_names)

    edges_to_restore: list[tuple[str, str]] = []
    weights_to_set: list[float] = []
    name_set = set(neuron_names)

    for name in neuron_names:
        for _, tgt, data in graph.out_edges(name, data=True):
            cw = data.get("chemical_weight", 0)
            if cw <= 0 or tgt in name_set:
                continue
            if rng.random() < connectivity_restore:
                edges_to_restore.append((name, tgt))
                weights_to_set.append(cw * connectivity_restore)

    if edges_to_restore:
        engine.set_weights(edges_to_restore, weights_to_set)


def apply_graceful_fade(
    engine: SimulationEngine,
    graph: nx.DiGraph,
    neuron_names: list[str],
    fade_steps: int = 5,
    step_duration_ms: float = 500.0,
) -> None:
    """Gradually reduce then replace neuron connectivity over *fade_steps*.

    At each step the outgoing weights are decreased by 1/fade_steps of
    their original value, and the simulation is advanced by
    *step_duration_ms*.  After full fade-out, a replacement with 50%
    restored connectivity is applied.
    """
    name_set = set(neuron_names)

    original_edges: list[tuple[str, str]] = []
    original_weights: list[float] = []
    for name in neuron_names:
        for _, tgt, data in graph.out_edges(name, data=True):
            cw = data.get("chemical_weight", 0)
            if cw <= 0 or tgt in name_set:
                continue
            original_edges.append((name, tgt))
            original_weights.append(float(cw))

    for step in range(1, fade_steps + 1):
        scale = 1.0 - step / fade_steps
        faded = [w * scale for w in original_weights]
        engine.set_weights(original_edges, faded)
        engine.run(step_duration_ms)

    apply_replacement(engine, graph, neuron_names, connectivity_restore=0.5)
