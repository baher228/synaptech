from __future__ import annotations

from typing import Protocol, runtime_checkable

import networkx as nx
import numpy as np


@runtime_checkable
class SimulationEngine(Protocol):
    """Contract that every simulation backend must satisfy.

    Both the NumPy and Brian2 engines expose identical data shapes so
    callers (API routes, sweep orchestration) never need to know which
    engine is running.
    """

    name: str

    def build(self, graph: nx.DiGraph, neuron_model: str = "lif") -> None:
        """Initialise the simulation from a connectome graph.

        *neuron_model* selects the dynamics:
        ``"lif"`` (default), ``"izhikevich"``, or ``"hh"``.
        """
        ...

    def run(self, duration_ms: float) -> None:
        """Advance the simulation by *duration_ms* milliseconds.

        May be called repeatedly; internal state persists between calls.
        """
        ...

    def get_spike_trains(self) -> dict[str, list[float]]:
        """Return spike times (ms) keyed by neuron name."""
        ...

    def get_firing_rates(self) -> dict[str, float]:
        """Return mean firing rate (Hz) per neuron over the last run."""
        ...

    def get_voltages(self) -> dict[str, np.ndarray]:
        """Return membrane-potential traces keyed by neuron name."""
        ...

    def silence_neurons(self, neuron_names: list[str]) -> None:
        """Disable the listed neurons (dropout / kill)."""
        ...

    def set_weights(
        self,
        edges: list[tuple[str, str]],
        weights: list[float],
    ) -> None:
        """Overwrite chemical synapse weights for the given edges."""
        ...

    def reset(self) -> None:
        """Clear all state and monitors so the next ``run`` starts fresh."""
        ...
