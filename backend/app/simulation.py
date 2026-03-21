from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import networkx as nx

from app.connectome import get_connectome_graph


@dataclass(frozen=True)
class LIFConfig:
    dt_ms: float = 0.5
    tau_ms: float = 20.0
    v_rest_mv: float = -65.0
    v_reset_mv: float = -65.0
    v_threshold_mv: float = -50.0
    refractory_ms: float = 2.0
    chemical_scale: float = 1.0
    gap_scale: float = 0.03
    initial_voltage_jitter: float = 1.5


@dataclass(frozen=True)
class LiveDriveConfig:
    sensory_tonic_current: float = 0.95
    sensory_noise_std: float = 0.30
    sensory_oscillation_hz: float = 3.0
    sensory_oscillation_amplitude: float = 0.25
    background_tonic_current: float = 0.12
    background_noise_std: float = 0.06


def _gabaergic_neuron_set() -> set[str]:
    # Cook 2019 synapse CSV here does not include neurotransmitter labels.
    # We use a conservative, known GABAergic subset to sign chemical output.
    names = {
        "AVL",
        "DVB",
        "RMEV",
        "RMED",
        "RMEL",
        "RMER",
        "RIS",
    }
    names.update({f"DD{idx:02d}" for idx in range(1, 7)})
    names.update({f"VD{idx:02d}" for idx in range(1, 14)})
    return names


GABAERGIC_NEURONS = _gabaergic_neuron_set()


class LIFNetworkSimulator:
    """Vectorized LIF simulator on top of the connectome graph."""

    def __init__(
        self,
        graph: nx.DiGraph,
        config: LIFConfig | None = None,
        drive_config: LiveDriveConfig | None = None,
        seed: int | None = None,
    ) -> None:
        self.graph = graph
        self.config = config or LIFConfig()
        self.drive_config = drive_config or LiveDriveConfig()
        self.rng = np.random.default_rng(seed)

        self.node_names = sorted(self.graph.nodes())
        self.node_index = {name: idx for idx, name in enumerate(self.node_names)}
        self.node_count = len(self.node_names)

        self.chemical_matrix, self.gap_matrix = self._build_weight_matrices()
        self.signed_chemical_matrix = self._build_signed_chemical_matrix()
        self.gap_degree = self.gap_matrix.sum(axis=1)

        node_types = np.array(
            [str(self.graph.nodes[name]["type"]) for name in self.node_names],
            dtype=object,
        )
        self.sensory_mask = (node_types == "S").astype(np.float64)
        self.interneuron_mask = (node_types == "I").astype(np.float64)
        self.motor_mask = (node_types == "M").astype(np.float64)

        self.step_index = 0
        self.refractory_steps = max(0, int(round(self.config.refractory_ms / self.config.dt_ms)))
        self.voltage_mv = np.full(
            self.node_count,
            fill_value=self.config.v_rest_mv,
            dtype=np.float64,
        )
        self.previous_spikes = np.zeros(self.node_count, dtype=np.float64)
        self.refractory_countdown = np.zeros(self.node_count, dtype=np.int32)
        self.reset_state()

    def _build_weight_matrices(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        chemical = np.zeros((self.node_count, self.node_count), dtype=np.float64)
        gap = np.zeros((self.node_count, self.node_count), dtype=np.float64)

        for source, target, edge_data in self.graph.edges(data=True):
            pre = self.node_index[source]
            post = self.node_index[target]
            chemical[pre, post] += float(edge_data.get("chemical_weight", 0.0))
            gap[pre, post] += float(edge_data.get("gap_weight", 0.0))

        return chemical, gap

    def _build_signed_chemical_matrix(self) -> npt.NDArray[np.float64]:
        presynaptic_sign = np.ones(self.node_count, dtype=np.float64)
        for idx, name in enumerate(self.node_names):
            if name in GABAERGIC_NEURONS:
                presynaptic_sign[idx] = -1.0

        return self.chemical_matrix * presynaptic_sign[:, None] * self.config.chemical_scale

    def reset_state(self) -> None:
        self.step_index = 0
        self.previous_spikes.fill(0.0)
        self.refractory_countdown.fill(0)
        self.voltage_mv = np.full(
            self.node_count,
            self.config.v_rest_mv,
            dtype=np.float64,
        )
        if self.config.initial_voltage_jitter > 0:
            self.voltage_mv += self.rng.normal(
                loc=0.0,
                scale=self.config.initial_voltage_jitter,
                size=self.node_count,
            )

    def sample_live_external_current(self) -> npt.NDArray[np.float64]:
        drive = self.drive_config
        external_current = np.full(
            self.node_count,
            fill_value=drive.background_tonic_current,
            dtype=np.float64,
        )
        if drive.background_noise_std > 0:
            external_current += self.rng.normal(
                loc=0.0,
                scale=drive.background_noise_std,
                size=self.node_count,
            )

        sensory_component = np.full(
            self.node_count,
            fill_value=drive.sensory_tonic_current,
            dtype=np.float64,
        )
        if drive.sensory_noise_std > 0:
            sensory_component += self.rng.normal(
                loc=0.0,
                scale=drive.sensory_noise_std,
                size=self.node_count,
            )

        if drive.sensory_oscillation_hz > 0 and drive.sensory_oscillation_amplitude != 0:
            time_sec = (self.step_index * self.config.dt_ms) / 1000.0
            phase = 2.0 * np.pi * drive.sensory_oscillation_hz * time_sec
            sensory_component += drive.sensory_oscillation_amplitude * np.sin(phase)

        external_current += self.sensory_mask * sensory_component
        return external_current

    def step(self, external_current: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
        available = self.refractory_countdown <= 0
        chemical_current = self.previous_spikes @ self.signed_chemical_matrix
        gap_current = self.config.gap_scale * (
            (self.gap_matrix @ self.voltage_mv) - (self.gap_degree * self.voltage_mv)
        )

        d_v_dt = (
            -(self.voltage_mv - self.config.v_rest_mv) / self.config.tau_ms
            + chemical_current
            + gap_current
            + external_current
        )
        self.voltage_mv[available] += self.config.dt_ms * d_v_dt[available]
        self.voltage_mv[~available] = self.config.v_reset_mv
        self.refractory_countdown[~available] -= 1

        spikes = available & (self.voltage_mv >= self.config.v_threshold_mv)
        self.voltage_mv[spikes] = self.config.v_reset_mv
        self.refractory_countdown[spikes] = self.refractory_steps
        self.previous_spikes = spikes.astype(np.float64)
        self.step_index += 1
        return spikes


def _summarize_firing_rates(
    simulator: LIFNetworkSimulator, firing_rates_hz: npt.NDArray[np.float64]
) -> dict[str, float]:
    def masked_mean(mask: npt.NDArray[np.float64]) -> float:
        if mask.sum() == 0:
            return 0.0
        return float(np.mean(firing_rates_hz[mask.astype(bool)]))

    return {
        "overall_mean_hz": float(np.mean(firing_rates_hz)),
        "sensory_mean_hz": masked_mean(simulator.sensory_mask),
        "interneuron_mean_hz": masked_mean(simulator.interneuron_mask),
        "motor_mean_hz": masked_mean(simulator.motor_mask),
        "active_fraction": float(np.mean(firing_rates_hz > 0)),
    }


def run_live_activity(
    *,
    duration_ms: float = 5_000.0,
    burn_in_ms: float = 1_000.0,
    seed: int | None = 7,
    max_samples: int = 500,
    preview_neurons: int = 14,
    config: LIFConfig | None = None,
    drive_config: LiveDriveConfig | None = None,
) -> dict[str, Any]:
    graph = get_connectome_graph()
    simulator = LIFNetworkSimulator(
        graph=graph,
        config=config,
        drive_config=drive_config,
        seed=seed,
    )

    dt_ms = simulator.config.dt_ms
    burn_in_steps = max(0, int(round(burn_in_ms / dt_ms)))
    measure_steps = max(1, int(round(duration_ms / dt_ms)))

    for _ in range(burn_in_steps):
        simulator.step(simulator.sample_live_external_current())

    sample_stride = max(1, measure_steps // max(1, max_samples))
    spike_counts = np.zeros(simulator.node_count, dtype=np.int64)
    sample_times_ms: list[float] = []
    sample_population_spikes: list[int] = []
    sample_voltage_mv: list[list[float]] = []

    for step_idx in range(measure_steps):
        spikes = simulator.step(simulator.sample_live_external_current())
        spike_counts += spikes.astype(np.int64)

        if step_idx % sample_stride == 0:
            # Keep payload bounded while preserving the temporal profile.
            sample_times_ms.append(round(step_idx * dt_ms, 3))
            sample_population_spikes.append(int(spikes.sum()))
            sample_voltage_mv.append(
                np.round(simulator.voltage_mv[:preview_neurons], 3).tolist()
            )

    duration_sec = (measure_steps * dt_ms) / 1000.0
    firing_rates_hz = spike_counts / duration_sec
    firing_summary = _summarize_firing_rates(simulator, firing_rates_hz)

    ranked_indices = np.argsort(firing_rates_hz)[::-1][:10]
    top_firing_neurons = [
        {
            "name": simulator.node_names[int(idx)],
            "firing_rate_hz": float(np.round(firing_rates_hz[int(idx)], 4)),
        }
        for idx in ranked_indices
    ]

    return {
        "config": {
            "dt_ms": simulator.config.dt_ms,
            "tau_ms": simulator.config.tau_ms,
            "v_rest_mv": simulator.config.v_rest_mv,
            "v_reset_mv": simulator.config.v_reset_mv,
            "v_threshold_mv": simulator.config.v_threshold_mv,
            "refractory_ms": simulator.config.refractory_ms,
            "chemical_scale": simulator.config.chemical_scale,
            "gap_scale": simulator.config.gap_scale,
            "burn_in_ms": burn_in_ms,
            "duration_ms": duration_ms,
            "seed": seed,
        },
        "drive": {
            "sensory_tonic_current": simulator.drive_config.sensory_tonic_current,
            "sensory_noise_std": simulator.drive_config.sensory_noise_std,
            "sensory_oscillation_hz": simulator.drive_config.sensory_oscillation_hz,
            "sensory_oscillation_amplitude": simulator.drive_config.sensory_oscillation_amplitude,
            "background_tonic_current": simulator.drive_config.background_tonic_current,
            "background_noise_std": simulator.drive_config.background_noise_std,
        },
        "node_count": simulator.node_count,
        "steps": {
            "burn_in": burn_in_steps,
            "measurement": measure_steps,
            "sample_stride": sample_stride,
        },
        "firing_summary_hz": firing_summary,
        "population_spike_rate_hz": float(
            spike_counts.sum() / (simulator.node_count * duration_sec)
        ),
        "top_firing_neurons": top_firing_neurons,
        "samples": {
            "times_ms": sample_times_ms,
            "population_spike_count": sample_population_spikes,
            "preview_node_names": simulator.node_names[:preview_neurons],
            "preview_voltage_mv": sample_voltage_mv,
        },
    }
