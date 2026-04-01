from __future__ import annotations

import networkx as nx
import pytest

from app.interventions.fault_detection import FaultDetectionService
from app.simulation.session import PersistentSimulation


class DummyEngine:
    name = "brian2"

    def __init__(self) -> None:
        self.build_calls: list[tuple[nx.DiGraph, str]] = []
        self.run_calls: list[float] = []

    def build(self, graph: nx.DiGraph, neuron_model: str = "lif") -> None:
        self.build_calls.append((graph.copy(), neuron_model))

    def run(self, duration_ms: float) -> None:
        self.run_calls.append(duration_ms)

    def get_firing_rates(self) -> dict[str, float]:
        return {"AVBL": 5.0, "DB01": 8.0, "VB01": 3.0}


@pytest.fixture
def mini_graph() -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_node("AVBL", type="I")
    graph.add_node("DB01", type="M")
    graph.add_node("VB01", type="M")
    graph.add_edge("AVBL", "DB01", weight=2.0, chemical_weight=2.0, gap_weight=0.0)
    graph.add_edge("AVBL", "VB01", weight=1.0, chemical_weight=1.0, gap_weight=0.0)
    return graph


def test_persistent_simulation_reset_rebuilds_with_requested_neuron_model(monkeypatch, mini_graph):
    engines: list[DummyEngine] = []

    def fake_get_engine(name: str = "brian2") -> DummyEngine:
        engine = DummyEngine()
        engines.append(engine)
        return engine

    monkeypatch.setattr("app.simulation.session.get_engine", fake_get_engine)

    sim = PersistentSimulation(graph=mini_graph, neuron_model="lif", burn_in_ms=25.0)
    assert sim.neuron_model == "lif"
    assert len(engines) == 1
    assert engines[0].build_calls[0][1] == "lif"
    assert engines[0].run_calls == [25.0]

    sim.reset(neuron_model="hh", burn_in_ms=40.0)
    assert sim.neuron_model == "hh"
    assert len(engines) == 2
    assert engines[1].build_calls[0][1] == "hh"
    assert engines[1].run_calls == [40.0]


def test_persistent_simulation_step_returns_summary(monkeypatch, mini_graph):
    engine = DummyEngine()
    monkeypatch.setattr("app.simulation.session.get_engine", lambda name="brian2": engine)

    sim = PersistentSimulation(graph=mini_graph, neuron_model="lif", burn_in_ms=10.0)
    payload = sim.step(50.0)

    assert payload["node_count"] == 3
    assert payload["step"] == 1
    assert payload["top_firing_neurons"][0]["name"] == "DB01"
    assert payload["firing_summary_hz"]["motor_mean_hz"] == pytest.approx(5.5)
    assert engine.run_calls == [10.0, 50.0]


def test_fault_detection_strategy_alias_normalization() -> None:
    assert FaultDetectionService._normalise_strategy("periphery_first") == "peripheral_first"


def test_fault_detection_unknown_strategy_raises() -> None:
    with pytest.raises(ValueError, match="Unknown strategy"):
        FaultDetectionService._normalise_strategy("definitely_not_real")


def test_candidate_neurons_excludes_replacement_and_ghosted() -> None:
    graph = nx.DiGraph()
    graph.add_node("A", is_replacement=False, is_ghosted=False)
    graph.add_node("B", is_replacement=True, is_ghosted=False)
    graph.add_node("C", is_replacement=False, is_ghosted=True)

    candidates = FaultDetectionService.candidate_neurons(graph)
    assert candidates == ["A"]
