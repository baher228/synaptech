"""Tests for the behavioral testing / locomotion assay pipeline.

Covers:
 - helper utilities (_series, _resolve_targets, _bin_counts, etc.)
 - ForwardLocomotionProtocol dataclass
 - forward_locomotion_behavior_spec
 - _population_rate_stats
 - _head_to_tail_wave_metrics
 - _muscle_ca_wave_proxy
 - _body_kinematics_metrics
 - Full run_forward_locomotion_assay integration test via Brian2
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest

from app.metrics.behavior_assays import (
    ForwardLocomotionProtocol,
    forward_locomotion_behavior_spec,
    run_forward_locomotion_assay,
    _series,
    _resolve_targets,
    _bin_counts,
    _safe_corr,
    _exp_filter,
    _population_rate_stats,
    _head_to_tail_wave_metrics,
    _muscle_ca_wave_proxy,
    _body_kinematics_metrics,
)


# ------------------------------------------------------------------ #
#  Fixtures
# ------------------------------------------------------------------ #


def _make_mini_graph() -> nx.DiGraph:
    """Build a tiny graph with enough locomotion-circuit neurons for testing."""
    g = nx.DiGraph()
    # Command interneurons
    for name in ("AVBL", "AVBR", "PVCL", "PVCR"):
        g.add_node(name, type="I")
    # B-type motor neurons (dorsal + ventral)
    for i in range(1, 4):
        g.add_node(f"DB{i:02d}", type="M")
        g.add_node(f"VB{i:02d}", type="M")
    # D-type motor neurons
    for i in range(1, 3):
        g.add_node(f"DD{i:02d}", type="M")
        g.add_node(f"VD{i:02d}", type="M")
    # A sensory neuron for connectivity
    g.add_node("ASEL", type="S")

    # Add some edges (chemical synapses)
    g.add_edge("AVBL", "DB01", chemical_weight=2, gap_weight=0)
    g.add_edge("AVBL", "DB02", chemical_weight=1, gap_weight=0)
    g.add_edge("AVBR", "VB01", chemical_weight=2, gap_weight=0)
    g.add_edge("AVBR", "VB02", chemical_weight=1, gap_weight=0)
    g.add_edge("DB01", "DD01", chemical_weight=1, gap_weight=0)
    g.add_edge("VB01", "VD01", chemical_weight=1, gap_weight=0)
    g.add_edge("ASEL", "AVBL", chemical_weight=1, gap_weight=0)
    # A gap junction
    g.add_edge("DB01", "DB02", chemical_weight=0, gap_weight=1)
    return g


@pytest.fixture
def mini_graph():
    return _make_mini_graph()


@pytest.fixture
def sample_spike_trains():
    """Spike trains with deterministic data for unit-level helpers."""
    return {
        "DB01": [10.0, 50.0, 90.0],
        "DB02": [20.0, 60.0],
        "DB03": [30.0, 70.0, 110.0],
        "VB01": [15.0, 55.0, 95.0],
        "VB02": [25.0, 65.0],
        "VB03": [35.0, 75.0, 115.0],
        "DD01": [40.0],
        "VD01": [45.0],
    }


# ------------------------------------------------------------------ #
#  Unit tests for helper functions
# ------------------------------------------------------------------ #


class TestSeries:
    def test_basic(self, mini_graph):
        result = _series("DB", 1, 3, mini_graph)
        assert result == ["DB01", "DB02", "DB03"]

    def test_missing_nodes(self, mini_graph):
        result = _series("DB", 1, 10, mini_graph)
        # Only DB01-03 exist in mini_graph
        assert result == ["DB01", "DB02", "DB03"]

    def test_empty(self, mini_graph):
        result = _series("XX", 1, 5, mini_graph)
        assert result == []


class TestResolveTargets:
    def test_basic(self, mini_graph):
        result = _resolve_targets(mini_graph, ["DB01", "VB01"])
        assert "DB01" in result
        assert "VB01" in result

    def test_unpadded_names(self, mini_graph):
        # "DB1" should resolve to "DB01"
        result = _resolve_targets(mini_graph, ["DB1"])
        assert "DB01" in result

    def test_unknown_ignored(self, mini_graph):
        result = _resolve_targets(mini_graph, ["UNKNOWN_NEURON"])
        assert result == []

    def test_case_insensitive(self, mini_graph):
        result = _resolve_targets(mini_graph, ["db01"])
        assert "DB01" in result


class TestBinCounts:
    def test_basic(self, sample_spike_trains):
        counts = _bin_counts(sample_spike_trains, ["DB01"], duration_ms=100.0, bin_ms=50.0)
        assert counts.shape == (2,)
        # DB01 spikes at 10, 50, 90. bin_ms=50, n_bins=2.
        # 10→bin0, 50→int(50/50)=1→bin1, 90→min(int(90/50),1)=1→bin1
        assert counts[0] == 1.0
        assert counts[1] == 2.0

    def test_multiple_names(self, sample_spike_trains):
        counts = _bin_counts(
            sample_spike_trains, ["DB01", "VB01"], duration_ms=100.0, bin_ms=50.0
        )
        # DB01: 10,50,90; VB01: 15,55,95
        # bin0: 10,15 → 2; bin1: 50,55,90,95 → 4
        assert counts[0] == 2.0
        assert counts[1] == 4.0

    def test_empty_trains(self):
        counts = _bin_counts({}, ["DB01"], duration_ms=100.0, bin_ms=50.0)
        np.testing.assert_array_equal(counts, [0.0, 0.0])


class TestSafeCorr:
    def test_identical_arrays(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        assert _safe_corr(x, x) == pytest.approx(1.0)

    def test_constant_array(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([5.0, 5.0, 5.0])
        assert _safe_corr(x, y) == 0.0

    def test_too_short(self):
        assert _safe_corr(np.array([1.0]), np.array([2.0])) == 0.0


class TestExpFilter:
    def test_smooths(self):
        signal = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
        filtered = _exp_filter(signal, tau_ms=10.0, dt_ms=5.0)
        # The filter should smooth the impulse
        assert filtered[2] > 0.0
        assert filtered[3] > 0.0  # tail from exponential

    def test_empty(self):
        result = _exp_filter(np.array([]), tau_ms=10.0, dt_ms=5.0)
        assert result.size == 0


# ------------------------------------------------------------------ #
#  Unit tests for population/wave metrics
# ------------------------------------------------------------------ #


class TestPopulationRateStats:
    def test_rates(self, sample_spike_trains):
        stats = _population_rate_stats(
            sample_spike_trains, ["DB01", "DB02", "DB03"], duration_ms=200.0
        )
        assert "mean_rate_hz" in stats
        assert "median_rate_hz" in stats
        assert "active_fraction" in stats
        assert "per_neuron_rate_hz" in stats
        # All three neurons fired, so active_fraction should be 1.0
        assert stats["active_fraction"] == 1.0
        # DB01: 3 spikes / 0.2s = 15 Hz
        assert stats["per_neuron_rate_hz"]["DB01"] == pytest.approx(15.0)

    def test_empty_population(self):
        stats = _population_rate_stats({}, [], duration_ms=100.0)
        assert stats["mean_rate_hz"] == 0.0


class TestHeadToTailWaveMetrics:
    def test_basic(self, sample_spike_trains):
        dorsal = ["DB01", "DB02", "DB03"]
        ventral = ["VB01", "VB02", "VB03"]
        result = _head_to_tail_wave_metrics(sample_spike_trains, dorsal, ventral)
        assert "head_to_tail_delay_ms_per_segment" in result
        assert "head_to_tail_fit_r2" in result
        assert "segments_with_activity" in result
        assert result["segments_with_activity"] == 3.0

    def test_insufficient_segments(self):
        trains = {"DB01": [10.0]}
        result = _head_to_tail_wave_metrics(trains, ["DB01"], ["VB01"])
        assert result["segments_with_activity"] == 1.0
        assert result["head_to_tail_delay_ms_per_segment"] == 0.0


class TestMuscleCaWaveProxy:
    def test_structure(self, sample_spike_trains):
        result = _muscle_ca_wave_proxy(
            sample_spike_trains, duration_ms=200.0, include_traces=False
        )
        assert result["available"] is True
        assert result["is_proxy"] is True
        assert "wave_travel_delay_ms_per_segment" in result
        assert "adjacent_segment_coherence" in result
        assert "dorsal_to_ventral_phase_lag_ms" in result
        assert "mean_wave_amplitude" in result

    def test_with_traces(self, sample_spike_trains):
        result = _muscle_ca_wave_proxy(
            sample_spike_trains, duration_ms=200.0, include_traces=True
        )
        assert "time_ms" in result
        assert "dorsal_ca_proxy" in result
        assert "ventral_ca_proxy" in result


class TestBodyKinematicsMetrics:
    def test_no_data(self):
        result = _body_kinematics_metrics(None, None)
        assert result["available"] is False

    def test_with_curvature(self):
        result = _body_kinematics_metrics([0.1, -0.2, 0.15], None)
        assert result["available"] is True
        assert "curvature_rms" in result
        assert "curvature_peak_to_peak" in result

    def test_with_speed(self):
        result = _body_kinematics_metrics(None, [0.5, 1.0, -0.2])
        assert result["available"] is True
        assert "speed_mean_mm_s" in result
        assert "speed_peak_mm_s" in result
        assert "forward_fraction" in result

    def test_empty_curvature_raises(self):
        with pytest.raises(ValueError):
            _body_kinematics_metrics([], None)


# ------------------------------------------------------------------ #
#  ForwardLocomotionProtocol tests
# ------------------------------------------------------------------ #


class TestForwardLocomotionProtocol:
    def test_pulse_width(self):
        proto = ForwardLocomotionProtocol(
            targets=["DB01"], amplitude_pA=8.0,
            period_ms=200.0, duty_cycle=0.5,
            start_ms=0.0, stop_ms=1000.0,
        )
        assert proto.pulse_width_ms() == 100.0

    def test_is_active(self):
        proto = ForwardLocomotionProtocol(
            targets=["DB01"], amplitude_pA=8.0,
            period_ms=100.0, duty_cycle=0.5,
            start_ms=100.0, stop_ms=500.0,
        )
        assert proto.is_active(50.0) is False      # before start
        assert proto.is_active(100.0) is True       # at start, in duty
        assert proto.is_active(149.0) is True       # still in duty
        assert proto.is_active(150.0) is False      # past duty cycle
        assert proto.is_active(200.0) is True       # next period
        assert proto.is_active(500.0) is False      # at stop

    def test_to_dict(self):
        proto = ForwardLocomotionProtocol(
            targets=["DB01", "VB01"], amplitude_pA=8.0,
            period_ms=200.0, duty_cycle=0.5,
            start_ms=0.0, stop_ms=1000.0,
        )
        d = proto.to_dict()
        assert d["type"] == "periodic_current"
        assert d["amplitude_pA"] == 8.0
        assert d["pulse_width_ms"] == 100.0


# ------------------------------------------------------------------ #
#  Behavior spec test
# ------------------------------------------------------------------ #


class TestForwardLocomotionBehaviorSpec:
    def test_structure(self, mini_graph):
        spec = forward_locomotion_behavior_spec(mini_graph)
        assert spec["behavior_id"] == "forward_locomotion"
        assert "canonical_circuit" in spec
        circuit = spec["canonical_circuit"]
        assert isinstance(circuit["command_interneurons"], list)
        assert isinstance(circuit["b_type_motor_neurons"], list)
        assert isinstance(circuit["d_type_motor_neurons"], list)
        assert "behavioral_readouts" in spec


# ------------------------------------------------------------------ #
#  Integration test: full assay through Brian2
# ------------------------------------------------------------------ #


class TestRunForwardLocomotionAssay:
    """Integration test that runs a short behavioral assay via Brian2.

    Uses the mini graph and a short simulation to keep runtime reasonable.
    """

    @pytest.fixture
    def assay_result(self, mini_graph):
        protocol = ForwardLocomotionProtocol(
            targets=["DB01", "VB01"],
            amplitude_pA=8.0,
            period_ms=200.0,
            duty_cycle=0.5,
            start_ms=0.0,
            stop_ms=500.0,
        )
        return run_forward_locomotion_assay(
            graph=mini_graph,
            engine_name="brian2",
            neuron_model="lif",
            burn_in_ms=200.0,
            duration_ms=500.0,
            protocol=protocol,
            integration_step_ms=10.0,
            include_traces=True,
        )

    def test_top_level_keys(self, assay_result):
        assert "behavior" in assay_result
        assert "input_protocol" in assay_result
        assert "behavioral_readout" in assay_result
        assert "assay_context" in assay_result

    def test_behavioral_readout_sections(self, assay_result):
        readout = assay_result["behavioral_readout"]
        assert "motor_neuron_firing_patterns" in readout
        assert "muscle_ca2_wave_proxy" in readout
        assert "body_kinematics" in readout

    def test_motor_neuron_firing_patterns(self, assay_result):
        motor = assay_result["behavioral_readout"]["motor_neuron_firing_patterns"]
        assert "b_type" in motor
        assert "d_type" in motor
        assert "dorsal_ventral_correlation" in motor
        assert "dorsal_ventral_anti_phase_index" in motor
        assert "head_to_tail_delay_ms_per_segment" in motor

    def test_muscle_ca2_proxy_available(self, assay_result):
        proxy = assay_result["behavioral_readout"]["muscle_ca2_wave_proxy"]
        assert proxy["available"] is True
        assert proxy["is_proxy"] is True
        # Since include_traces=True, traces should be present
        assert "time_ms" in proxy
        assert "dorsal_ca_proxy" in proxy
        assert "ventral_ca_proxy" in proxy

    def test_body_kinematics_unavailable(self, assay_result):
        # No curvature/speed data was provided
        kinematics = assay_result["behavioral_readout"]["body_kinematics"]
        assert kinematics["available"] is False

    def test_assay_context(self, assay_result):
        ctx = assay_result["assay_context"]
        assert ctx["engine"] == "brian2"
        assert ctx["neuron_model"] == "lif"
        assert ctx["duration_ms"] == 500.0

    def test_input_protocol_recorded(self, assay_result):
        proto = assay_result["input_protocol"]
        assert proto["type"] == "periodic_current"
        assert proto["amplitude_pA"] == 8.0

    def test_validation_errors(self, mini_graph):
        base_proto = ForwardLocomotionProtocol(
            targets=["DB01"], amplitude_pA=8.0,
            period_ms=200.0, duty_cycle=0.5,
            start_ms=0.0, stop_ms=500.0,
        )
        # duration_ms <= 0
        with pytest.raises(ValueError):
            run_forward_locomotion_assay(
                mini_graph, "brian2", "lif", 100.0, 0.0, base_proto,
            )
        # bad duty cycle
        bad_duty = ForwardLocomotionProtocol(
            targets=["DB01"], amplitude_pA=8.0,
            period_ms=200.0, duty_cycle=0.0,
            start_ms=0.0, stop_ms=500.0,
        )
        with pytest.raises(ValueError):
            run_forward_locomotion_assay(
                mini_graph, "brian2", "lif", 100.0, 500.0, bad_duty,
            )
        # stop <= start
        bad_stop = ForwardLocomotionProtocol(
            targets=["DB01"], amplitude_pA=8.0,
            period_ms=200.0, duty_cycle=0.5,
            start_ms=500.0, stop_ms=100.0,
        )
        with pytest.raises(ValueError):
            run_forward_locomotion_assay(
                mini_graph, "brian2", "lif", 100.0, 500.0, bad_stop,
            )
