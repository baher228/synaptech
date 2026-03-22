"""Behavior-specific performance assays for locomotion readouts."""

from __future__ import annotations

from dataclasses import dataclass
import math

import networkx as nx
import numpy as np

from app.simulation.factory import get_engine

_FORWARD_COMMAND_NEURONS = ("AVBL", "AVBR", "PVCL", "PVCR")


def _series(prefix: str, start: int, stop: int, graph: nx.DiGraph) -> list[str]:
    names: list[str] = []
    for idx in range(start, stop + 1):
        candidate = f"{prefix}{idx:02d}"
        if graph.has_node(candidate):
            names.append(candidate)
    return names


def _resolve_targets(graph: nx.DiGraph, targets: list[str]) -> list[str]:
    resolved: list[str] = []
    for raw_name in targets:
        name = raw_name.strip().upper()
        if graph.has_node(name):
            resolved.append(name)
            continue
        if len(name) >= 3 and name[:2] in {"DB", "VB", "DD", "VD"} and name[2:].isdigit():
            padded = f"{name[:2]}{int(name[2:]):02d}"
            if graph.has_node(padded):
                resolved.append(padded)
    return sorted(set(resolved))


def _bin_counts(
    spike_trains: dict[str, list[float]],
    names: list[str],
    duration_ms: float,
    bin_ms: float,
) -> np.ndarray:
    n_bins = max(1, int(math.ceil(duration_ms / max(bin_ms, 1e-6))))
    counts = np.zeros(n_bins, dtype=float)
    for name in names:
        for spike_ms in spike_trains.get(name, []):
            bin_idx = min(int(spike_ms / bin_ms), n_bins - 1)
            counts[bin_idx] += 1.0
    return counts


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size != y.size or x.size < 2:
        return 0.0
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    corr = float(np.corrcoef(x, y)[0, 1])
    return corr if np.isfinite(corr) else 0.0


def _exp_filter(signal: np.ndarray, tau_ms: float, dt_ms: float) -> np.ndarray:
    if signal.size == 0:
        return signal
    alpha = math.exp(-max(dt_ms, 1e-6) / max(tau_ms, 1e-6))
    out = np.zeros_like(signal, dtype=float)
    for i, val in enumerate(signal):
        prev = out[i - 1] if i > 0 else 0.0
        out[i] = alpha * prev + (1.0 - alpha) * float(val)
    return out


def _population_rate_stats(
    spike_trains: dict[str, list[float]],
    names: list[str],
    duration_ms: float,
) -> dict[str, object]:
    dur_s = max(duration_ms / 1000.0, 1e-9)
    per_neuron = {name: len(spike_trains.get(name, [])) / dur_s for name in names}
    values = np.array(list(per_neuron.values()), dtype=float)
    return {
        "mean_rate_hz": float(np.mean(values)) if values.size else 0.0,
        "median_rate_hz": float(np.median(values)) if values.size else 0.0,
        "active_fraction": float(np.mean(values > 0.0)) if values.size else 0.0,
        "per_neuron_rate_hz": {k: round(float(v), 6) for k, v in per_neuron.items()},
    }


def _head_to_tail_wave_metrics(
    spike_trains: dict[str, list[float]],
    dorsal_b: list[str],
    ventral_b: list[str],
) -> dict[str, float]:
    max_seg = 0
    for name in dorsal_b + ventral_b:
        if len(name) >= 4 and name[2:].isdigit():
            max_seg = max(max_seg, int(name[2:]))

    seg_ids: list[float] = []
    first_spikes: list[float] = []
    for seg in range(1, max_seg + 1):
        choices = [f"DB{seg:02d}", f"VB{seg:02d}"]
        seg_spikes = [min(spike_trains.get(n, [])) for n in choices if spike_trains.get(n)]
        if seg_spikes:
            seg_ids.append(float(seg))
            first_spikes.append(float(min(seg_spikes)))

    if len(seg_ids) < 3:
        return {
            "head_to_tail_delay_ms_per_segment": 0.0,
            "head_to_tail_fit_r2": 0.0,
            "segments_with_activity": float(len(seg_ids)),
        }

    x = np.array(seg_ids, dtype=float)
    y = np.array(first_spikes, dtype=float)
    slope, intercept = np.polyfit(x, y, deg=1)
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
    return {
        "head_to_tail_delay_ms_per_segment": float(slope),
        "head_to_tail_fit_r2": float(max(0.0, r2)),
        "segments_with_activity": float(len(seg_ids)),
    }


def _muscle_ca_wave_proxy(
    spike_trains: dict[str, list[float]],
    duration_ms: float,
    include_traces: bool,
    bin_ms: float = 20.0,
    tau_ms: float = 120.0,
) -> dict[str, object]:
    segments = list(range(1, 12))
    dorsal_segments: dict[str, np.ndarray] = {}
    ventral_segments: dict[str, np.ndarray] = {}

    for seg in segments:
        db_name = f"DB{seg:02d}"
        vb_name = f"VB{seg:02d}"
        d_counts = _bin_counts(spike_trains, [db_name], duration_ms, bin_ms)
        v_counts = _bin_counts(spike_trains, [vb_name], duration_ms, bin_ms)
        dorsal_segments[f"seg{seg:02d}"] = _exp_filter(d_counts, tau_ms=tau_ms, dt_ms=bin_ms)
        ventral_segments[f"seg{seg:02d}"] = _exp_filter(v_counts, tau_ms=tau_ms, dt_ms=bin_ms)

    combined = {
        seg: dorsal_segments[seg] + ventral_segments[seg]
        for seg in dorsal_segments
    }
    seg_numbers = np.array([int(seg[3:]) for seg in combined.keys()], dtype=float)
    peak_times: list[float] = []
    valid_segments: list[float] = []

    for seg_key in sorted(combined.keys()):
        arr = combined[seg_key]
        if np.max(arr) <= 1e-12:
            continue
        valid_segments.append(float(int(seg_key[3:])))
        peak_times.append(float(np.argmax(arr) * bin_ms))

    if len(valid_segments) >= 3:
        x = np.array(valid_segments, dtype=float)
        y = np.array(peak_times, dtype=float)
        slope, intercept = np.polyfit(x, y, deg=1)
        y_hat = slope * x + intercept
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
    else:
        slope = 0.0
        r2 = 0.0

    adjacent_corrs: list[float] = []
    for left, right in zip(segments[:-1], segments[1:]):
        left_key = f"seg{left:02d}"
        right_key = f"seg{right:02d}"
        adjacent_corrs.append(_safe_corr(combined[left_key], combined[right_key]))

    dorsal_stack = np.vstack([dorsal_segments[f"seg{s:02d}"] for s in segments])
    ventral_stack = np.vstack([ventral_segments[f"seg{s:02d}"] for s in segments])
    mean_dorsal = np.mean(dorsal_stack, axis=0)
    mean_ventral = np.mean(ventral_stack, axis=0)
    x = mean_dorsal - np.mean(mean_dorsal)
    y = mean_ventral - np.mean(mean_ventral)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        lag_ms = 0.0
    else:
        xcorr = np.correlate(x, y, mode="full")
        lags = np.arange(-len(x) + 1, len(x))
        lag_ms = float(lags[int(np.argmax(np.abs(xcorr)))] * bin_ms)

    amplitudes = [float(np.max(arr) - np.min(arr)) for arr in combined.values()]
    result: dict[str, object] = {
        "available": True,
        "is_proxy": True,
        "notes": (
            "Proxy muscle Ca2+ waves inferred from DB/VB motor spike activity; "
            "not a direct muscle-cell simulation."
        ),
        "bin_ms": bin_ms,
        "tau_ms": tau_ms,
        "segment_count": len(segments),
        "wave_travel_delay_ms_per_segment": float(slope),
        "wave_travel_fit_r2": float(max(0.0, r2)),
        "adjacent_segment_coherence": float(np.mean(adjacent_corrs)) if adjacent_corrs else 0.0,
        "dorsal_to_ventral_phase_lag_ms": lag_ms,
        "mean_wave_amplitude": float(np.mean(amplitudes)) if amplitudes else 0.0,
    }

    if include_traces:
        time_axis = (np.arange(dorsal_stack.shape[1]) * bin_ms).astype(float).tolist()
        result["time_ms"] = time_axis
        result["dorsal_ca_proxy"] = {
            seg: [round(float(v), 6) for v in arr]
            for seg, arr in dorsal_segments.items()
        }
        result["ventral_ca_proxy"] = {
            seg: [round(float(v), 6) for v in arr]
            for seg, arr in ventral_segments.items()
        }

    return result


def _body_kinematics_metrics(
    curvature: list[float] | None,
    speed_mm_s: list[float] | None,
) -> dict[str, object]:
    if curvature is None and speed_mm_s is None:
        return {
            "available": False,
            "reason": (
                "Body curvature/speed metrics require coupling to OpenWorm/Sibernetic "
                "or external kinematic samples in the request."
            ),
        }

    result: dict[str, object] = {"available": True}
    if curvature is not None:
        c = np.array(curvature, dtype=float)
        if c.size == 0:
            raise ValueError("body_curvature must contain at least one value when provided.")
        result["curvature_rms"] = float(np.sqrt(np.mean(c ** 2)))
        result["curvature_peak_to_peak"] = float(np.max(c) - np.min(c))

    if speed_mm_s is not None:
        s = np.array(speed_mm_s, dtype=float)
        if s.size == 0:
            raise ValueError("body_speed_mm_s must contain at least one value when provided.")
        result["speed_mean_mm_s"] = float(np.mean(s))
        result["speed_peak_mm_s"] = float(np.max(s))
        result["forward_fraction"] = float(np.mean(s > 0.0))

    return result


def forward_locomotion_behavior_spec(graph: nx.DiGraph) -> dict[str, object]:
    b_dorsal = _series("DB", 1, 7, graph)
    b_ventral = _series("VB", 1, 11, graph)
    d_dorsal = _series("DD", 1, 6, graph)
    d_ventral = _series("VD", 1, 13, graph)
    return {
        "behavior_id": "forward_locomotion",
        "description": (
            "Forward crawling assay centered on AVB/PVC command input and "
            "B- and D-type body-wall motor neurons."
        ),
        "canonical_circuit": {
            "command_interneurons": [n for n in _FORWARD_COMMAND_NEURONS if graph.has_node(n)],
            "b_type_motor_neurons": b_dorsal + b_ventral,
            "d_type_motor_neurons": d_dorsal + d_ventral,
            "stimulus_entry_targets": ["DB01", "VB01"],
        },
        "behavioral_readouts": [
            "motor_neuron_firing_patterns",
            "muscle_ca2_wave_proxy",
            "body_kinematics",
        ],
    }


@dataclass(frozen=True)
class ForwardLocomotionProtocol:
    targets: list[str]
    amplitude_pA: float
    period_ms: float
    duty_cycle: float
    start_ms: float
    stop_ms: float

    def pulse_width_ms(self) -> float:
        return self.period_ms * self.duty_cycle

    def is_active(self, t_ms: float) -> bool:
        if t_ms < self.start_ms or t_ms >= self.stop_ms:
            return False
        phase = (t_ms - self.start_ms) % self.period_ms
        return phase < self.pulse_width_ms()

    def to_dict(self) -> dict[str, object]:
        return {
            "type": "periodic_current",
            "targets": self.targets,
            "amplitude_pA": self.amplitude_pA,
            "period_ms": self.period_ms,
            "duty_cycle": self.duty_cycle,
            "pulse_width_ms": self.pulse_width_ms(),
            "start_ms": self.start_ms,
            "stop_ms": self.stop_ms,
        }


def run_forward_locomotion_assay(
    graph: nx.DiGraph,
    engine_name: str,
    neuron_model: str,
    burn_in_ms: float,
    duration_ms: float,
    protocol: ForwardLocomotionProtocol,
    integration_step_ms: float = 10.0,
    include_traces: bool = False,
    body_curvature: list[float] | None = None,
    body_speed_mm_s: list[float] | None = None,
) -> dict[str, object]:
    if duration_ms <= 0:
        raise ValueError("duration_ms must be > 0.")
    if integration_step_ms <= 0:
        raise ValueError("integration_step_ms must be > 0.")
    if protocol.period_ms <= 0:
        raise ValueError("protocol.period_ms must be > 0.")
    if not (0.0 < protocol.duty_cycle <= 1.0):
        raise ValueError("protocol.duty_cycle must be in (0, 1].")
    if protocol.stop_ms <= protocol.start_ms:
        raise ValueError("protocol.stop_ms must be greater than protocol.start_ms.")

    resolved_targets = _resolve_targets(graph, protocol.targets)
    if not resolved_targets:
        raise ValueError("No valid stimulus targets found in graph for the requested protocol.")

    engine = get_engine(engine_name)
    engine.build(graph.copy(), neuron_model=neuron_model)

    if not hasattr(engine, "apply_drive_overrides") or not hasattr(engine, "clear_drive_overrides"):
        raise ValueError(f"Engine '{engine_name}' does not support drive override protocols.")

    engine.run(burn_in_ms)
    is_stim_on = None
    elapsed_ms = 0.0
    while elapsed_ms < duration_ms - 1e-9:
        run_ms = min(integration_step_ms, duration_ms - elapsed_ms)
        now_ms = elapsed_ms + 0.5 * run_ms
        stim_on = protocol.is_active(now_ms)
        if stim_on != is_stim_on:
            if stim_on:
                engine.apply_drive_overrides({n: protocol.amplitude_pA for n in resolved_targets})
            else:
                engine.clear_drive_overrides()
            is_stim_on = stim_on
        engine.run(run_ms)
        elapsed_ms += run_ms

    engine.clear_drive_overrides()
    spike_trains = engine.get_spike_trains()

    b_dorsal = _series("DB", 1, 7, graph)
    b_ventral = _series("VB", 1, 11, graph)
    d_dorsal = _series("DD", 1, 6, graph)
    d_ventral = _series("VD", 1, 13, graph)

    b_stats = _population_rate_stats(spike_trains, b_dorsal + b_ventral, duration_ms)
    d_stats = _population_rate_stats(spike_trains, d_dorsal + d_ventral, duration_ms)
    dorsal_profile = _bin_counts(spike_trains, b_dorsal + d_dorsal, duration_ms, bin_ms=20.0)
    ventral_profile = _bin_counts(spike_trains, b_ventral + d_ventral, duration_ms, bin_ms=20.0)
    dorsal_ventral_corr = _safe_corr(dorsal_profile, ventral_profile)
    wave_metrics = _head_to_tail_wave_metrics(spike_trains, b_dorsal, b_ventral)

    return {
        "behavior": forward_locomotion_behavior_spec(graph),
        "input_protocol": ForwardLocomotionProtocol(
            targets=resolved_targets,
            amplitude_pA=protocol.amplitude_pA,
            period_ms=protocol.period_ms,
            duty_cycle=protocol.duty_cycle,
            start_ms=protocol.start_ms,
            stop_ms=protocol.stop_ms,
        ).to_dict(),
        "behavioral_readout": {
            "motor_neuron_firing_patterns": {
                "b_type": b_stats,
                "d_type": d_stats,
                "dorsal_ventral_correlation": dorsal_ventral_corr,
                "dorsal_ventral_anti_phase_index": max(0.0, -dorsal_ventral_corr),
                **wave_metrics,
            },
            "muscle_ca2_wave_proxy": _muscle_ca_wave_proxy(
                spike_trains=spike_trains,
                duration_ms=duration_ms,
                include_traces=include_traces,
            ),
            "body_kinematics": _body_kinematics_metrics(body_curvature, body_speed_mm_s),
        },
        "assay_context": {
            "engine": engine_name,
            "neuron_model": neuron_model,
            "burn_in_ms": burn_in_ms,
            "duration_ms": duration_ms,
            "integration_step_ms": integration_step_ms,
        },
    }
