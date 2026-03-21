"""Brian2-backed simulation engine for the C. elegans connectome.

Supports three neuron models (``lif``, ``izhikevich``, ``hh``) selected
at :meth:`build` time.  Chemical synapses use ``on_pre`` spike delivery;
gap junctions use continuous voltage-dependent current.

Parameters are cross-checked against c302 ``parameters_A.py`` Level A
(IAF: leak_reversal=-50mV, thresh=-30mV, reset=-50mV, C=3pF, g_leak=0.1nS,
syn gbase=0.01nS, exc_erev=0mV, inh_erev=-80mV, tau_rise=3ms, tau_decay=10ms).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from brian2 import (
    NeuronGroup,
    Synapses,
    SpikeMonitor,
    StateMonitor,
    Network,
    defaultclock,
    start_scope,
    mV,
    ms,
    nS,
    pF,
    pA,
    Hz,
    second,
    prefs,
)

prefs.codegen.target = "numpy"

# ------------------------------------------------------------------ #
#  Neuron model equations — all currents in amps, divided by Cm
# ------------------------------------------------------------------ #

_LIF_EQS = """
dv/dt = (g_leak*(E_leak - v) + I_gap + I_ext) / Cm : volt
I_gap : amp
I_ext = drive + noise_amp*randn() : amp (constant over dt)
drive : amp
is_alive : 1
Cm : farad
g_leak : siemens
E_leak : volt
noise_amp : amp
"""

_LIF_THRESHOLD = "v > V_thresh and is_alive > 0.5"
_LIF_RESET = "v = V_reset"

# Izhikevich with physical units (Izhikevich 2007 formulation).
# k_izh has units of siemens/volt so that k*(v-vr)*(v-vt) yields amps.
_IZH_EQS = """
dv/dt = (k_izh*(v - v_r)*(v - v_t) - u_izh + I_gap + I_ext) / Cm : volt
du_izh/dt = a_izh*(b_izh*(v - v_r) - u_izh) : amp
I_gap : amp
I_ext = drive + noise_amp*randn() : amp (constant over dt)
drive : amp
is_alive : 1
Cm : farad
k_izh : amp/volt**2
v_r : volt
v_t : volt
a_izh : Hz
b_izh : siemens
noise_amp : amp
"""

_IZH_THRESHOLD = "v > 30*mV and is_alive > 0.5"
_IZH_RESET = """
v = v_r
u_izh += 100*pA
"""

# Hodgkin-Huxley (shifted so v is absolute, rest ~ -65 mV).
# Rate functions use v_s = v + 65*mV (displacement from rest) to match
# the classic HH parameterisation where 0 = resting potential.
_HH_EQS = """
dv/dt = (g_Na*m**3*h*(E_Na - v) + g_K*n**4*(E_K - v) + g_L*(E_L - v) + I_gap + I_ext) / Cm : volt
dm/dt = alpha_m*(1.0 - m) - beta_m*m : 1
dn/dt = alpha_n*(1.0 - n) - beta_n*n : 1
dh/dt = alpha_h*(1.0 - h) - beta_h*h : 1
v_s = v + 65*mV : volt
alpha_m = (0.182/mV)*(v_s - 25*mV + 0.001*mV) / (1.0 - exp(-(v_s - 25*mV + 0.001*mV)/(9*mV))) / ms : Hz
beta_m  = (-0.124/mV)*(v_s - 25*mV + 0.001*mV) / (1.0 - exp((v_s - 25*mV + 0.001*mV)/(9*mV))) / ms : Hz
alpha_n = (0.02/mV)*(v_s - 10*mV + 0.001*mV) / (1.0 - exp(-(v_s - 10*mV + 0.001*mV)/(9*mV))) / ms : Hz
beta_n  = (-0.002/mV)*(v_s - 10*mV + 0.001*mV) / (1.0 - exp((v_s - 10*mV + 0.001*mV)/(9*mV))) / ms : Hz
alpha_h = 0.07*exp(-(v_s)/(20*mV))/ms : Hz
beta_h  = 1.0/(1.0 + exp(-(v_s - 30*mV)/(10*mV)))/ms : Hz
I_gap : amp
I_ext = drive + noise_amp*randn() : amp (constant over dt)
drive : amp
is_alive : 1
g_Na : siemens
g_K : siemens
g_L : siemens
E_Na : volt
E_K : volt
E_L : volt
Cm : farad
noise_amp : amp
"""

_HH_THRESHOLD = "v > -20*mV and is_alive > 0.5"
_HH_RESET = ""


@dataclass
class Brian2Engine:
    """Brian2 simulation engine satisfying the SimulationEngine protocol."""

    name: str = "brian2"

    _neuron_names: list[str] = field(default_factory=list)
    _name_to_idx: dict[str, int] = field(default_factory=dict)
    _sensory_indices: list[int] = field(default_factory=list)
    _n: int = 0
    _model: str = "lif"

    _net: Network | None = None
    _neurons: NeuronGroup | None = None
    _chem_syn: Synapses | None = None
    _gap_syn: Synapses | None = None
    _spike_mon: SpikeMonitor | None = None
    _state_mon: StateMonitor | None = None

    _graph: nx.DiGraph | None = None
    _sim_time_ms: float = 0.0
    _run_start_ms: float = 0.0
    _built: bool = False

    # ------------------------------------------------------------------ #
    #  Protocol methods
    # ------------------------------------------------------------------ #

    def build(self, graph: nx.DiGraph, neuron_model: str = "lif") -> None:
        start_scope()
        defaultclock.dt = 0.5 * ms

        self._graph = graph
        self._model = neuron_model
        self._neuron_names = list(graph.nodes)
        self._name_to_idx = {name: i for i, name in enumerate(self._neuron_names)}
        self._n = len(self._neuron_names)
        self._sensory_indices = [
            i for i, n in enumerate(self._neuron_names)
            if graph.nodes[n].get("type") == "S"
        ]

        self._build_neurons()
        self._build_chemical_synapses(graph)
        self._build_gap_junctions(graph)

        self._spike_mon = SpikeMonitor(self._neurons)
        self._state_mon = StateMonitor(self._neurons, "v", record=True, dt=1 * ms)

        self._net = Network(
            self._neurons,
            self._chem_syn,
            self._gap_syn,
            self._spike_mon,
            self._state_mon,
        )
        self._sim_time_ms = 0.0
        self._run_start_ms = 0.0
        self._built = True

    def run(self, duration_ms: float) -> None:
        if not self._built:
            raise RuntimeError("Call build() before run()")
        self._run_start_ms = self._sim_time_ms
        self._net.run(duration_ms * ms)
        self._sim_time_ms += duration_ms

    def get_spike_trains(self) -> dict[str, list[float]]:
        trains: dict[str, list[float]] = {n: [] for n in self._neuron_names}
        indices = np.array(self._spike_mon.i)
        times = np.array(self._spike_mon.t / ms)
        mask = times >= self._run_start_ms
        for idx, t in zip(indices[mask], times[mask]):
            trains[self._neuron_names[int(idx)]].append(float(t))
        return trains

    def get_firing_rates(self) -> dict[str, float]:
        dur_s = (self._sim_time_ms - self._run_start_ms) / 1000.0
        if dur_s <= 0:
            return {n: 0.0 for n in self._neuron_names}
        trains = self.get_spike_trains()
        return {name: len(ts) / dur_s for name, ts in trains.items()}

    def get_voltages(self) -> dict[str, np.ndarray]:
        vs = np.array(self._state_mon.v / mV)
        return {self._neuron_names[i]: vs[i] for i in range(self._n)}

    def silence_neurons(self, neuron_names: list[str]) -> None:
        for name in neuron_names:
            idx = self._name_to_idx.get(name)
            if idx is not None:
                self._neurons.is_alive[idx] = 0

    def set_weights(
        self,
        edges: list[tuple[str, str]],
        weights: list[float],
    ) -> None:
        for (src, tgt), w in zip(edges, weights):
            si = self._name_to_idx.get(src)
            ti = self._name_to_idx.get(tgt)
            if si is not None and ti is not None:
                mask = (self._chem_syn.i == si) & (self._chem_syn.j == ti)
                self._chem_syn.w[mask] = w * 0.3 * mV

    def reset(self) -> None:
        if self._built:
            self.build(self._graph, self._model)

    # ------------------------------------------------------------------ #
    #  Internal builders
    # ------------------------------------------------------------------ #

    def _build_neurons(self) -> None:
        model = self._model

        if model == "lif":
            self._neurons = NeuronGroup(
                self._n, _LIF_EQS,
                threshold=_LIF_THRESHOLD,
                reset=_LIF_RESET,
                refractory=2 * ms,
                method="euler",
            )
            ng = self._neurons
            ng.v = -50 * mV
            ng.Cm = 3 * pF
            ng.g_leak = 0.1 * nS
            ng.E_leak = -50 * mV
            ng.is_alive = 1
            ng.noise_amp = 0.8 * pA
            ng.drive = 0 * pA
            for idx in self._sensory_indices:
                ng.drive[idx] = 3.5 * pA
            ng.namespace["V_thresh"] = -30 * mV
            ng.namespace["V_reset"] = -50 * mV

        elif model == "izhikevich":
            self._neurons = NeuronGroup(
                self._n, _IZH_EQS,
                threshold=_IZH_THRESHOLD,
                reset=_IZH_RESET,
                method="euler",
            )
            ng = self._neurons
            ng.v = -65 * mV
            ng.u_izh = 0 * pA
            ng.Cm = 100 * pF
            ng.k_izh = 0.7 * nS / mV
            ng.v_r = -60 * mV
            ng.v_t = -40 * mV
            ng.a_izh = 30 * Hz
            ng.b_izh = -2.0 * nS
            ng.is_alive = 1
            ng.noise_amp = 20.0 * pA
            ng.drive = 0 * pA
            for idx in self._sensory_indices:
                ng.drive[idx] = 80.0 * pA

        elif model == "hh":
            defaultclock.dt = 0.05 * ms
            self._neurons = NeuronGroup(
                self._n, _HH_EQS,
                threshold=_HH_THRESHOLD,
                reset=_HH_RESET,
                refractory=3 * ms,
                method="exponential_euler",
            )
            ng = self._neurons
            ng.v = -65 * mV
            ng.m = 0.05
            ng.h = 0.6
            ng.n = 0.32
            ng.g_Na = 120 * nS
            ng.g_K = 36 * nS
            ng.g_L = 0.3 * nS
            ng.E_Na = 50 * mV
            ng.E_K = -77 * mV
            ng.E_L = -54.4 * mV
            ng.Cm = 3 * pF
            ng.is_alive = 1
            ng.noise_amp = 5.0 * pA
            ng.drive = 0 * pA
            for idx in self._sensory_indices:
                ng.drive[idx] = 25.0 * pA

        else:
            raise ValueError(
                f"Unknown neuron_model '{model}'. Use 'lif', 'izhikevich', or 'hh'."
            )

    def _build_chemical_synapses(self, graph: nx.DiGraph) -> None:
        src_list, tgt_list, weight_list = [], [], []
        for src, tgt, data in graph.edges(data=True):
            cw = data.get("chemical_weight", 0)
            if cw <= 0:
                continue
            src_list.append(self._name_to_idx[src])
            tgt_list.append(self._name_to_idx[tgt])
            weight_list.append(cw)

        self._chem_syn = Synapses(
            self._neurons, self._neurons,
            "w : volt",
            on_pre="v_post += w * int(is_alive_pre > 0.5)",
        )
        if src_list:
            self._chem_syn.connect(i=src_list, j=tgt_list)
            self._chem_syn.w = np.array(weight_list) * 0.3 * mV

    def _build_gap_junctions(self, graph: nx.DiGraph) -> None:
        src_list, tgt_list, weight_list = [], [], []
        for src, tgt, data in graph.edges(data=True):
            gw = data.get("gap_weight", 0)
            if gw <= 0:
                continue
            src_list.append(self._name_to_idx[src])
            tgt_list.append(self._name_to_idx[tgt])
            weight_list.append(gw)

        self._gap_syn = Synapses(
            self._neurons, self._neurons,
            """w_gap : siemens
               I_gap_post = w_gap * (v_pre - v_post) : amp (summed)""",
        )
        if src_list:
            self._gap_syn.connect(i=src_list, j=tgt_list)
            self._gap_syn.w_gap = np.array(weight_list) * 0.005 * nS
