from __future__ import annotations

from typing import Protocol, runtime_checkable

import networkx as nx
import numpy as np


@runtime_checkable
class SimulationEngine(Protocol):
    """Contract that simulation backends must satisfy."""

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

    def get_voltage_matrix(self, window_ms: float | None = None) -> np.ndarray:
        """Return (N, T) voltage matrix in mV.

        If *window_ms* is given, return only the last *window_ms* columns.
        """
        ...

    def silence_neurons(self, neuron_names: list[str]) -> None:
        """Disable the listed neurons (dropout / kill)."""
        ...

    def activate_neuron(self, name: str) -> None:
        """Re-enable a previously silenced neuron."""
        ...

    def add_neuron_from_slot(
        self,
        replacement_name: str,
        copy_params_from: str | None = None,
    ) -> int:
        """Allocate a pre-reserved slot for a replacement neuron."""
        ...

    def set_weights(
        self,
        edges: list[tuple[str, str]],
        weights: list[float],
    ) -> None:
        """Overwrite chemical synapse weights for the given edges."""
        ...

    def set_gap_weights(
        self,
        edges: list[tuple[str, str]],
        weights: list[float],
    ) -> None:
        """Overwrite gap junction weights for the given edges."""
        ...

    def reset(self) -> None:
        """Clear all state and monitors so the next ``run`` starts fresh."""
        ...

    def apply_drive_overrides(self, overrides_pA: dict[str, float]) -> None:
        """Apply additive drive overrides (pA) for selected neurons.

        Values are interpreted as offsets relative to each neuron's
        baseline tonic drive configured during ``build``.
        """
        ...

    def clear_drive_overrides(self) -> None:
        """Remove all drive overrides and restore baseline tonic drives."""
        ...
