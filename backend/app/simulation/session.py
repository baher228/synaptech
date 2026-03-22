"""Persistent simulation session that lives across HTTP requests.

The Brian2 engine is built once (with burn-in) and advanced incrementally
on each poll. Internal state (membrane potentials, synaptic conductances,
spike monitors) carries forward between calls.
"""

from __future__ import annotations

import threading

import networkx as nx
import numpy as np

from app.connectome import get_connectome_graph
from app.simulation.factory import get_engine


class PersistentSimulation:
    """Long-lived simulation backed by a Brian2 engine."""

    def __init__(
        self,
        graph: nx.DiGraph | None = None,
        neuron_model: str = "lif",
        burn_in_ms: float = 2000.0,
    ) -> None:
        self._graph = graph if graph is not None else get_connectome_graph()
        self._neuron_model = neuron_model

        # Classify neurons for summary stats
        self._node_names = list(self._graph.nodes)
        self._node_count = len(self._node_names)
        self._sensory_mask = np.array([
            self._graph.nodes[n].get("type") == "S" for n in self._node_names
        ])
        self._inter_mask = np.array([
            self._graph.nodes[n].get("type") == "I" for n in self._node_names
        ])
        self._motor_mask = np.array([
            self._graph.nodes[n].get("type") == "M" for n in self._node_names
        ])

        # Build and burn in
        self._engine = get_engine("brian2")
        self._engine.build(self._graph, neuron_model=neuron_model)
        self._engine.run(burn_in_ms)
        self._step_count = 0
        self._lock = threading.Lock()

    def step(self, duration_ms: float = 500.0) -> dict:
        """Advance the simulation and return current firing rates.

        Thread-safe: Brian2 is not reentrant, so we serialize access.
        """
        with self._lock:
            self._engine.run(duration_ms)
            self._step_count += 1

            rates = self._engine.get_firing_rates()

        rate_array = np.array([rates.get(n, 0.0) for n in self._node_names])

        def _masked_mean(mask: np.ndarray) -> float:
            if mask.sum() == 0:
                return 0.0
            return float(np.mean(rate_array[mask]))

        firing_summary = {
            "overall_mean_hz": float(np.mean(rate_array)),
            "sensory_mean_hz": _masked_mean(self._sensory_mask),
            "interneuron_mean_hz": _masked_mean(self._inter_mask),
            "motor_mean_hz": _masked_mean(self._motor_mask),
            "active_fraction": float(np.mean(rate_array > 0)),
        }

        ranked = np.argsort(rate_array)[::-1][:10]
        top_firing = [
            {"name": self._node_names[int(i)], "firing_rate_hz": round(float(rate_array[int(i)]), 4)}
            for i in ranked
        ]

        rates_by_node = {
            name: round(float(rate_array[i]), 6)
            for i, name in enumerate(self._node_names)
        }

        return {
            "node_count": self._node_count,
            "step": self._step_count,
            "population_spike_rate_hz": float(rate_array.sum() / max(self._node_count, 1)),
            "firing_summary_hz": firing_summary,
            "firing_rates_hz_by_node": rates_by_node,
            "top_firing_neurons": top_firing,
        }

    @property
    def engine(self):
        return self._engine

    @property
    def graph(self):
        return self._graph
