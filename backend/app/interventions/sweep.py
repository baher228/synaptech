"""Experiment sweep orchestration.

Runs a combinatorial grid of (fraction x trial), collecting
failure scores and per-run metrics into a structured result set.

Also provides :func:`run_replacement_sweep` for per-step metric capture
during neuron replacement (supports batch replacement).
"""

from __future__ import annotations

import math
import random as stdlib_random
from dataclasses import asdict, dataclass, field
from typing import Generator, Literal

import networkx as nx
import numpy as np

from app.connectome import B_CLASS_MOTOR_NEURONS
from app.interventions.fault_detection import FaultDetectionService
from app.interventions.replacement_service import ReplacementService
from app.metrics.metrics import (
    compute_baseline,
    failure_score,
    firing_rate_distribution,
    kuramoto_order_parameter,
    network_synchrony,
    pathway_fidelity,
    pca_attractor_deviation,
    voltage_state_entropy,
)
from app.simulation.factory import get_engine

Intervention = Literal["dropout", "replacement"]


@dataclass
class SweepResult:
    fraction: float
    target_selector: str
    intervention: str
    trial: int
    failure: float
    targeted_neurons: list[str]
    baseline: dict
    post: dict

    def to_dict(self) -> dict:
        return asdict(self)


def _neuron_groups(graph: nx.DiGraph) -> tuple[list[str], list[str]]:
    """Return (sensory_names, motor_names) from the graph."""
    sensory = [n for n, d in graph.nodes(data=True) if d.get("type") == "S"]
    motor = [n for n, d in graph.nodes(data=True) if d.get("type") == "M"]
    return sensory, motor


def _canonical_neuron_name(name: str) -> str:
    """Map replacement neuron names back to canonical neuron ids."""
    if "__rep_" in name:
        return name.split("__rep_", 1)[0]
    return name


def _active_b_class_population(
    graph: nx.DiGraph,
    spike_trains: dict[str, list[float]] | None = None,
) -> list[str]:
    """Resolve current B-class population, including live replacement neurons."""
    names: list[str] = []
    for node_name, attrs in graph.nodes(data=True):
        if attrs.get("is_ghosted"):
            continue
        if _canonical_neuron_name(node_name) in B_CLASS_MOTOR_NEURONS:
            names.append(node_name)

    if spike_trains is not None:
        for node_name in spike_trains.keys():
            if _canonical_neuron_name(node_name) in B_CLASS_MOTOR_NEURONS:
                names.append(node_name)

    return sorted(set(names))


def _replace_targets_in_graph(
    graph: nx.DiGraph,
    targets: list[str],
    rng: stdlib_random.Random,
) -> nx.DiGraph:
    service = ReplacementService(graph)
    for target in targets:
        session = service.start_replacement(
            faulty_neuron=target,
            edge_order="random",
            seed=rng.randint(0, 10**9),
        )
        while session.status != "completed":
            session = service.step_replacement(
                session.session_id,
                edges_to_migrate=max(1, min(100, len(session.pending))),
            )
    return service.graph


def run_sweep(
    graph: nx.DiGraph,
    fractions: list[float],
    intervention: Intervention = "dropout",
    n_trials: int = 3,
    engine_name: str = "brian2",
    neuron_model: str = "lif",
    burn_in_ms: float = 2000.0,
    duration_ms: float = 5000.0,
    seed: int | None = None,
) -> list[SweepResult]:
    """Run sweep protocol for random-faulty target selection.

    For each (fraction, trial):
      1. Build a fresh simulation from the graph.
      2. Burn in, then capture baseline metrics.
      3. Apply intervention.
      4. Run post-intervention capture.
      5. Compute failure score.
    """
    rng = stdlib_random.Random(seed)
    detector = FaultDetectionService()
    sensory, motor = _neuron_groups(graph)
    results: list[SweepResult] = []

    for frac in fractions:
        for trial in range(n_trials):
            engine = get_engine(engine_name)
            engine.build(graph, neuron_model=neuron_model)

            # Baseline capture
            engine.run(burn_in_ms)
            engine.run(duration_ms)
            bl_trains = engine.get_spike_trains()
            baseline = compute_baseline(bl_trains, duration_ms, sensory, motor)

            targets = detector.select_targets_by_fraction(
                graph=graph,
                fraction=frac,
                seed=rng.randint(0, 10**9),
            )

            if intervention == "dropout":
                engine.silence_neurons(targets)
                engine.run(duration_ms)
                post_trains = engine.get_spike_trains()
            elif intervention == "replacement":
                replaced_graph = _replace_targets_in_graph(graph, targets, rng)
                post_engine = get_engine(engine_name)
                post_engine.build(replaced_graph, neuron_model=neuron_model)
                post_engine.run(burn_in_ms)
                post_engine.run(duration_ms)
                post_trains = post_engine.get_spike_trains()
            else:
                raise ValueError(f"Unknown intervention '{intervention}'")

            post = compute_baseline(post_trains, duration_ms, sensory, motor)
            score = failure_score(baseline, post)

            results.append(
                SweepResult(
                    fraction=frac,
                    target_selector="random_faulty",
                    intervention=intervention,
                    trial=trial,
                    failure=score,
                    targeted_neurons=targets,
                    baseline=baseline.to_dict(),
                    post=post.to_dict(),
                )
            )

    return results


# ------------------------------------------------------------------ #
#  Per-step replacement sweep
# ------------------------------------------------------------------ #

@dataclass
class StepMetrics:
    step_index: int
    neuron_being_replaced: str
    edges_migrated: int
    total_edges: int
    kuramoto_r: float
    pca_deviation: float
    pca_sigma: float
    voltage_entropy: float
    firing_rate_mean: float
    synchrony: float
    pathway_fidelity_val: float
    ou_convergence: float | None = None
    neurons_in_batch: list[str] = field(default_factory=list)
    batch_index: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ReplacementTimeSeries:
    strategy: str
    neuron_model: str
    replacement_order: list[str]
    baseline: dict
    integration: str = "mirror"
    steps: list[StepMetrics] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "neuron_model": self.neuron_model,
            "replacement_order": self.replacement_order,
            "baseline": self.baseline,
            "integration": self.integration,
            "steps": [s.to_dict() for s in self.steps],
        }


# ------------------------------------------------------------------ #
#  Helpers for batch replacement
# ------------------------------------------------------------------ #

def _collect_step_metrics(
    *,
    engine: object,
    bl_voltages: object,
    b_class: list[str],
    graph: nx.DiGraph | None,
    sensory: list[str],
    motor: list[str],
    step_ms: float,
    global_step: int,
    sessions: list,
    batch: list[str],
    batch_index: int,
) -> StepMetrics:
    """Measure all metrics after one tick of replacement progress."""
    trains = engine.get_spike_trains()  # type: ignore[attr-defined]
    voltages = engine.get_voltage_matrix(window_ms=step_ms)  # type: ignore[attr-defined]

    rates = firing_rate_distribution(trains, step_ms)
    rate_vals = np.array(list(rates.values()))
    pca_d, pca_s = pca_attractor_deviation(bl_voltages, voltages)

    edges_migrated = sum(len(s.completed) for s in sessions)
    total_edges = sum(len(s.pending) + len(s.completed) for s in sessions)

    ou_conv = None
    ou_managers = [s.ou_manager for s in sessions if s.ou_manager is not None]
    if ou_managers:
        convs = [m.convergence_fraction() for m in ou_managers]
        ou_conv = sum(convs) / len(convs)

    b_class_names = (
        _active_b_class_population(graph, spike_trains=trains)
        if graph is not None
        else b_class
    )

    return StepMetrics(
        step_index=global_step,
        neuron_being_replaced=", ".join(batch),
        edges_migrated=edges_migrated,
        total_edges=total_edges,
        kuramoto_r=kuramoto_order_parameter(trains, b_class_names, step_ms),
        pca_deviation=pca_d,
        pca_sigma=pca_s,
        voltage_entropy=voltage_state_entropy(voltages, step_ms),
        firing_rate_mean=float(np.mean(rate_vals)) if len(rate_vals) else 0.0,
        synchrony=network_synchrony(trains, step_ms),
        pathway_fidelity_val=pathway_fidelity(trains, sensory, motor, step_ms),
        ou_convergence=ou_conv,
        neurons_in_batch=list(batch),
        batch_index=batch_index,
    )


def _tick_sessions(
    service: ReplacementService,
    sessions: list,
    edges_per_step: int,
) -> None:
    """Advance all active sessions by one step."""
    for session in sessions:
        if session.status == "completed":
            continue
        if session.mode == "ou":
            service.tick_ou(session.session_id)
        else:
            service.step_replacement(
                session.session_id,
                edges_to_migrate=edges_per_step,
            )


def _start_batch(
    service: ReplacementService,
    batch: list[str],
    rng: stdlib_random.Random,
    replacement_mode: str,
    ou_theta: float | None,
    ou_sigma: float | None,
    integration: str,
    integration_params: dict | None,
) -> list:
    """Start replacement sessions for all neurons in a batch."""
    sessions = []
    for neuron_name in batch:
        session = service.start_replacement(
            faulty_neuron=neuron_name,
            edge_order="random",
            seed=rng.randint(0, 10**9),
            mode=replacement_mode,
            theta=ou_theta,
            sigma=ou_sigma,
            integration=integration,
            integration_params=integration_params,
        )
        sessions.append(session)
    return sessions


# ------------------------------------------------------------------ #
#  Main sweep functions
# ------------------------------------------------------------------ #

def run_replacement_sweep(
    graph: nx.DiGraph,
    target_neurons: list[str],
    engine_name: str = "brian2",
    neuron_model: str = "lif",
    burn_in_ms: float = 2000.0,
    baseline_ms: float = 5000.0,
    step_ms: float = 500.0,
    edges_per_step: int = 1,
    seed: int | None = None,
    replacement_mode: str = "instant",
    ou_theta: float | None = None,
    ou_sigma: float | None = None,
    integration: str = "mirror",
    integration_params: dict | None = None,
    batch_size: int = 1,
    settle_ms: float = 0.0,
) -> ReplacementTimeSeries:
    """Run neuron replacement with per-step metric snapshots.

    1. Build engine, burn in, capture baseline (voltages + spikes).
    2. For each batch of *batch_size* neurons in *target_neurons*:
       - start_replacement for all neurons in the batch
       - Drive all sessions to completion in lockstep
       - Optionally settle the network for *settle_ms*
    3. Return time-series of metrics at every step.

    Set *replacement_mode* to ``"ou"`` for gradual OU-based replacement.
    Set *integration* to vary how replacement neurons wire in.
    """
    rng = stdlib_random.Random(seed)
    engine = get_engine(engine_name)
    working_graph = graph.copy()
    engine.build(working_graph, neuron_model=neuron_model)

    sensory, motor = _neuron_groups(graph)
    b_class = _active_b_class_population(graph)

    # Burn-in + baseline capture
    engine.run(burn_in_ms)
    engine.run(baseline_ms)
    bl_trains = engine.get_spike_trains()
    bl_voltages = engine.get_voltage_matrix(window_ms=baseline_ms)
    baseline = compute_baseline(
        bl_trains, baseline_ms, sensory, motor,
        voltages=bl_voltages, b_class_names=b_class,
    )

    # Replacement loop — process neurons in batches
    service = ReplacementService(graph=working_graph, engine=engine)
    steps: list[StepMetrics] = []
    global_step = 0

    batches = [
        target_neurons[i : i + batch_size]
        for i in range(0, len(target_neurons), batch_size)
    ]

    for batch_index, batch in enumerate(batches):
        sessions = _start_batch(
            service, batch, rng,
            replacement_mode, ou_theta, ou_sigma,
            integration, integration_params,
        )

        while any(s.status != "completed" for s in sessions):
            _tick_sessions(service, sessions, edges_per_step)
            engine.run(step_ms)

            steps.append(_collect_step_metrics(
                engine=engine,
                bl_voltages=bl_voltages,
                b_class=b_class,
                graph=service.graph,
                sensory=sensory,
                motor=motor,
                step_ms=step_ms,
                global_step=global_step,
                sessions=sessions,
                batch=batch,
                batch_index=batch_index,
            ))
            global_step += 1

        if settle_ms > 0.0:
            engine.run(settle_ms)

    return ReplacementTimeSeries(
        strategy="one_by_one_replacement",
        neuron_model=neuron_model,
        replacement_order=target_neurons,
        baseline=baseline.to_dict(),
        integration=integration,
        steps=steps,
    )


def run_replacement_sweep_stream(
    graph: nx.DiGraph,
    target_neurons: list[str],
    engine_name: str = "brian2",
    neuron_model: str = "lif",
    burn_in_ms: float = 2000.0,
    baseline_ms: float = 5000.0,
    step_ms: float = 500.0,
    edges_per_step: int = 1,
    seed: int | None = None,
    replacement_mode: str = "instant",
    ou_theta: float | None = None,
    ou_sigma: float | None = None,
    integration: str = "mirror",
    integration_params: dict | None = None,
    batch_size: int = 1,
    settle_ms: float = 0.0,
) -> Generator[dict, None, None]:
    """Streaming version of :func:`run_replacement_sweep`.

    Yields SSE-ready dicts: baseline event, then one step event per
    edge-migration batch, then a done event.
    """
    rng = stdlib_random.Random(seed)
    engine = get_engine(engine_name)
    working_graph = graph.copy()
    engine.build(working_graph, neuron_model=neuron_model)

    sensory, motor = _neuron_groups(graph)
    b_class = _active_b_class_population(graph)

    # Estimate total steps for progress reporting
    total_steps = sum(
        math.ceil((graph.in_degree(n) + graph.out_degree(n)) / edges_per_step)
        for n in target_neurons
    )

    # Burn-in + baseline capture
    engine.run(burn_in_ms)
    engine.run(baseline_ms)
    bl_trains = engine.get_spike_trains()
    bl_voltages = engine.get_voltage_matrix(window_ms=baseline_ms)
    baseline = compute_baseline(
        bl_trains, baseline_ms, sensory, motor,
        voltages=bl_voltages, b_class_names=b_class,
    )

    yield {
        "type": "baseline",
        "data": baseline.to_dict(),
        "strategy": "one_by_one_replacement",
        "neuron_model": neuron_model,
        "replacement_order": target_neurons,
        "total_steps": total_steps,
        "replacement_mode": replacement_mode,
        "integration": integration,
        "batch_size": batch_size,
    }

    # Replacement loop
    service = ReplacementService(graph=working_graph, engine=engine)
    global_step = 0

    batches = [
        target_neurons[i : i + batch_size]
        for i in range(0, len(target_neurons), batch_size)
    ]

    for batch_index, batch in enumerate(batches):
        sessions = _start_batch(
            service, batch, rng,
            replacement_mode, ou_theta, ou_sigma,
            integration, integration_params,
        )

        while any(s.status != "completed" for s in sessions):
            _tick_sessions(service, sessions, edges_per_step)
            engine.run(step_ms)

            step = _collect_step_metrics(
                engine=engine,
                bl_voltages=bl_voltages,
                b_class=b_class,
                graph=service.graph,
                sensory=sensory,
                motor=motor,
                step_ms=step_ms,
                global_step=global_step,
                sessions=sessions,
                batch=batch,
                batch_index=batch_index,
            )
            yield {"type": "step", "data": step.to_dict()}
            global_step += 1

        if settle_ms > 0.0:
            engine.run(settle_ms)

    yield {"type": "done"}


def run_live_sweep_stream(
    sim: object,
    target_neurons: list[str],
    step_ms: float = 500.0,
    baseline_ms: float = 1000.0,
    edges_per_step: int = 5,
    seed: int | None = None,
    replacement_mode: str = "instant",
    ou_theta: float | None = None,
    ou_sigma: float | None = None,
    integration: str = "mirror",
    integration_params: dict | None = None,
    batch_size: int = 1,
    settle_ms: float = 0.0,
) -> Generator[dict, None, None]:
    """Run a replacement sweep on the persistent simulation.

    Uses the already-running engine — no build or burn-in needed.
    Captures a quick baseline from the current state, then replaces
    neurons in batches on the live engine.
    """
    rng = stdlib_random.Random(seed)
    engine = sim.engine  # type: ignore[attr-defined]
    graph = sim.graph  # type: ignore[attr-defined]
    # Alias for compatibility with older code paths that referenced this name.
    working_graph = graph
    sim_lock = getattr(sim, "lock", getattr(sim, "_lock", None))

    def _is_clock_exhaustion(exc: Exception) -> bool:
        if isinstance(exc, StopIteration):
            return True
        if isinstance(exc, RuntimeError):
            return "Clock has reached the end of its available times" in str(exc)
        return False

    # Only replace currently valid, non-ghosted neurons.
    valid_targets = [
        n for n in target_neurons
        if graph.has_node(n) and not graph.nodes[n].get("is_ghosted")
    ]

    # Persistent live simulation keeps replacement neurons allocated between calls.
    # If we are out of reserved slots, emit a structured SSE error instead of
    # throwing from inside StreamingResponse.
    next_free_slot = getattr(engine, "_next_free_slot", None)
    pool_size = getattr(engine, "_n", None)
    if isinstance(next_free_slot, int) and isinstance(pool_size, int):
        remaining_slots = max(0, pool_size - next_free_slot)
        required_slots = len(valid_targets)
        if required_slots > remaining_slots:
            yield {
                "type": "error",
                "error": "replacement_slot_exhausted",
                "message": (
                    "Not enough free neuron slots in persistent simulation for this live sweep. "
                    "Reset the session via POST /api/simulation/session/reset and retry."
                ),
                "required_slots": required_slots,
                "remaining_slots": remaining_slots,
                "pool_size": pool_size,
            }
            yield {"type": "done"}
            return

    sensory, motor = _neuron_groups(graph)
    b_class = _active_b_class_population(working_graph)

    total_steps = sum(
        math.ceil((graph.in_degree(n) + graph.out_degree(n)) / edges_per_step)
        for n in valid_targets
    )

    # Quick baseline from current state (no rebuild)
    try:
        if sim_lock is not None:
            with sim_lock:
                engine.run(baseline_ms)
                bl_trains = engine.get_spike_trains()
                bl_voltages = engine.get_voltage_matrix(window_ms=baseline_ms)
        else:
            engine.run(baseline_ms)
            bl_trains = engine.get_spike_trains()
            bl_voltages = engine.get_voltage_matrix(window_ms=baseline_ms)
    except Exception as exc:
        if _is_clock_exhaustion(exc):
            yield {
                "type": "error",
                "error": "clock_exhausted",
                "message": (
                    "Persistent simulation clock exhausted during live sweep baseline. "
                    "Reset via POST /api/simulation/session/reset and retry."
                ),
            }
            yield {"type": "done"}
            return
        raise
    baseline = compute_baseline(
        bl_trains, baseline_ms, sensory, motor,
        voltages=bl_voltages, b_class_names=b_class,
    )

    yield {
        "type": "baseline",
        "data": baseline.to_dict(),
        "strategy": "one_by_one_live",
        "neuron_model": "lif",
        "replacement_order": valid_targets,
        "total_steps": total_steps,
        "replacement_mode": replacement_mode,
        "integration": integration,
        "batch_size": batch_size,
    }

    # Replacement loop on the live engine
    service = ReplacementService(graph=graph, engine=engine)
    global_step = 0

    batches = [
        valid_targets[i : i + batch_size]
        for i in range(0, len(valid_targets), batch_size)
    ]

    for batch_index, batch in enumerate(batches):
        try:
            if sim_lock is not None:
                with sim_lock:
                    sessions = _start_batch(
                        service, batch, rng,
                        replacement_mode, ou_theta, ou_sigma,
                        integration, integration_params,
                    )
            else:
                sessions = _start_batch(
                    service, batch, rng,
                    replacement_mode, ou_theta, ou_sigma,
                    integration, integration_params,
                )
        except RuntimeError as exc:
            yield {
                "type": "error",
                "error": "replacement_start_failed",
                "batch": batch,
                "message": str(exc),
            }
            yield {"type": "done"}
            return

        while any(s.status != "completed" for s in sessions):
            try:
                if sim_lock is not None:
                    with sim_lock:
                        _tick_sessions(service, sessions, edges_per_step)
                        engine.run(step_ms)
                        step = _collect_step_metrics(
                            engine=engine,
                            bl_voltages=bl_voltages,
                            b_class=b_class,
                            graph=service.graph,
                            sensory=sensory,
                            motor=motor,
                            step_ms=step_ms,
                            global_step=global_step,
                            sessions=sessions,
                            batch=batch,
                            batch_index=batch_index,
                        )
                else:
                    _tick_sessions(service, sessions, edges_per_step)
                    engine.run(step_ms)
                    step = _collect_step_metrics(
                        engine=engine,
                        bl_voltages=bl_voltages,
                        b_class=b_class,
                        graph=service.graph,
                        sensory=sensory,
                        motor=motor,
                        step_ms=step_ms,
                        global_step=global_step,
                        sessions=sessions,
                        batch=batch,
                        batch_index=batch_index,
                    )
            except Exception as exc:
                if _is_clock_exhaustion(exc):
                    yield {
                        "type": "error",
                        "error": "clock_exhausted",
                        "batch": batch,
                        "message": (
                            "Persistent simulation clock exhausted during live sweep. "
                            "Reset via POST /api/simulation/session/reset and retry."
                        ),
                    }
                    yield {"type": "done"}
                    return
                raise

            yield {"type": "step", "data": step.to_dict()}
            global_step += 1

        if settle_ms > 0.0:
            try:
                if sim_lock is not None:
                    with sim_lock:
                        engine.run(settle_ms)
                else:
                    engine.run(settle_ms)
            except Exception as exc:
                if _is_clock_exhaustion(exc):
                    yield {
                        "type": "error",
                        "error": "clock_exhausted",
                        "batch": batch,
                        "message": (
                            "Persistent simulation clock exhausted during settle period. "
                            "Reset via POST /api/simulation/session/reset and retry."
                        ),
                    }
                    yield {"type": "done"}
                    return
                raise

    yield {"type": "done"}
