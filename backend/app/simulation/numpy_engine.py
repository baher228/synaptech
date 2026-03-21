"""Vectorised Leaky Integrate-and-Fire simulation on the C. elegans connectome.

All 302 neurons are updated in parallel each timestep using NumPy matrix
operations.  The engine satisfies :class:`SimulationEngine` so it can be
swapped transparently with the Brian2 backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np


@dataclass
class NumpyLIFEngine:
    """Pure-NumPy LIF simulator wired to a connectome DiGraph."""

    name: str = "numpy"

    # ----- Biophysical parameters (SI-ish: mV / ms) ----- #
    v_rest: float = -65.0     # mV
    v_thresh: float = -50.0   # mV
    v_reset: float = -65.0    # mV
    tau: float = 20.0         # membrane time constant (ms)
    dt: float = 0.5           # integration timestep (ms)
    refractory_ms: float = 2.0

    # Synaptic scaling applied to raw synapse counts
    w_chem_scale: float = 0.35   # mV per synapse per spike
    w_gap_scale: float = 0.005   # coupling coefficient per gap-junction count
    noise_std: float = 0.6       # mV per timestep, background noise
    sensory_drive: float = 1.5   # mV per timestep tonic input to sensory neurons

    # ----- Internal state (populated by build()) ----- #
    _neuron_names: list[str] = field(default_factory=list)
    _name_to_idx: dict[str, int] = field(default_factory=dict)
    _n: int = 0

    _V: np.ndarray = field(default_factory=lambda: np.array([]))
    _active: np.ndarray = field(default_factory=lambda: np.array([]))
    _refractory_remaining: np.ndarray = field(default_factory=lambda: np.array([]))
    _sensory_mask: np.ndarray = field(default_factory=lambda: np.array([]))

    _W_chem: np.ndarray = field(default_factory=lambda: np.array([]))
    _W_gap: np.ndarray = field(default_factory=lambda: np.array([]))

    # Excitatory (+1) or inhibitory (-1) per neuron — simplified: all +1 for now
    _sign: np.ndarray = field(default_factory=lambda: np.array([]))

    # Recording buffers (filled during run())
    _spike_trains: dict[str, list[float]] = field(default_factory=dict)
    _voltage_traces: dict[str, list[float]] = field(default_factory=dict)
    _sim_time: float = 0.0
    _run_start_time: float = 0.0

    _built: bool = False

    # ------------------------------------------------------------------ #
    #  Protocol methods
    # ------------------------------------------------------------------ #

    def build(self, graph: nx.DiGraph, neuron_model: str = "lif") -> None:
        if neuron_model != "lif":
            raise ValueError(f"NumpyLIFEngine only supports 'lif', got '{neuron_model}'")

        self._neuron_names = list(graph.nodes)
        self._name_to_idx = {name: i for i, name in enumerate(self._neuron_names)}
        self._n = len(self._neuron_names)

        self._V = np.full(self._n, self.v_rest)
        self._active = np.ones(self._n, dtype=bool)
        self._refractory_remaining = np.zeros(self._n)

        self._sensory_mask = np.array(
            [graph.nodes[n].get("type") == "S" for n in self._neuron_names],
            dtype=bool,
        )

        self._W_chem = np.zeros((self._n, self._n))
        self._W_gap = np.zeros((self._n, self._n))

        for src, tgt, data in graph.edges(data=True):
            i, j = self._name_to_idx[src], self._name_to_idx[tgt]
            cw = data.get("chemical_weight", 0)
            gw = data.get("gap_weight", 0)
            if cw:
                self._W_chem[i, j] += cw * self.w_chem_scale
            if gw:
                self._W_gap[i, j] += gw * self.w_gap_scale

        self._sign = np.ones(self._n)

        self._spike_trains = {name: [] for name in self._neuron_names}
        self._voltage_traces = {name: [] for name in self._neuron_names}
        self._sim_time = 0.0
        self._run_start_time = 0.0
        self._built = True

    def run(self, duration_ms: float) -> None:
        if not self._built:
            raise RuntimeError("Call build() before run()")

        self._run_start_time = self._sim_time
        steps = int(duration_ms / self.dt)

        for _ in range(steps):
            self._step()

    def get_spike_trains(self) -> dict[str, list[float]]:
        return {
            name: [t for t in times if t >= self._run_start_time]
            for name, times in self._spike_trains.items()
        }

    def get_firing_rates(self) -> dict[str, float]:
        run_duration_s = (self._sim_time - self._run_start_time) / 1000.0
        if run_duration_s <= 0:
            return {n: 0.0 for n in self._neuron_names}
        trains = self.get_spike_trains()
        return {name: len(times) / run_duration_s for name, times in trains.items()}

    def get_voltages(self) -> dict[str, np.ndarray]:
        return {name: np.array(trace) for name, trace in self._voltage_traces.items()}

    def silence_neurons(self, neuron_names: list[str]) -> None:
        for name in neuron_names:
            idx = self._name_to_idx.get(name)
            if idx is None:
                continue
            self._active[idx] = False
            self._V[idx] = self.v_rest
            self._W_chem[idx, :] = 0.0
            self._W_chem[:, idx] = 0.0
            self._W_gap[idx, :] = 0.0
            self._W_gap[:, idx] = 0.0

    def set_weights(
        self,
        edges: list[tuple[str, str]],
        weights: list[float],
    ) -> None:
        for (src, tgt), w in zip(edges, weights):
            i = self._name_to_idx.get(src)
            j = self._name_to_idx.get(tgt)
            if i is not None and j is not None:
                self._W_chem[i, j] = w

    def reset(self) -> None:
        self._V[:] = self.v_rest
        self._refractory_remaining[:] = 0.0
        self._spike_trains = {name: [] for name in self._neuron_names}
        self._voltage_traces = {name: [] for name in self._neuron_names}
        self._sim_time = 0.0
        self._run_start_time = 0.0

    # ------------------------------------------------------------------ #
    #  Internal
    # ------------------------------------------------------------------ #

    def _step(self) -> None:
        n = self._n
        V = self._V
        active = self._active

        # --- external input ---
        I_ext = np.random.normal(0.0, self.noise_std, n)
        I_ext[self._sensory_mask] += self.sensory_drive

        # --- gap junction current (bidirectional, voltage-dependent) ---
        # I_gap_i = sum_j W_gap[j,i] * (V[j] - V[i])
        I_gap = self._W_gap.T @ V - np.sum(self._W_gap, axis=0) * V

        # --- membrane integration ---
        not_refractory = self._refractory_remaining <= 0
        dV = (-(V - self.v_rest) / self.tau + I_ext + I_gap) * self.dt
        V += dV * active * not_refractory

        # --- spike detection ---
        spiked = (V >= self.v_thresh) & active & not_refractory

        # --- synaptic current from spikes (delivered next conceptual step) ---
        if np.any(spiked):
            spike_input = self._W_chem.T @ (spiked.astype(float) * self._sign)
            V += spike_input * active * not_refractory

        # --- reset spiked neurons ---
        V[spiked] = self.v_reset
        self._refractory_remaining[spiked] = self.refractory_ms
        self._refractory_remaining -= self.dt
        np.clip(self._refractory_remaining, 0.0, None, out=self._refractory_remaining)

        # --- record ---
        spike_indices = np.where(spiked)[0]
        for idx in spike_indices:
            self._spike_trains[self._neuron_names[idx]].append(self._sim_time)

        for idx in range(n):
            self._voltage_traces[self._neuron_names[idx]].append(float(V[idx]))

        self._V = V
        self._sim_time += self.dt
