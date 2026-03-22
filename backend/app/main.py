from __future__ import annotations

from functools import lru_cache
import random as stdlib_random
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.connectome import (
    graph_to_data,
    get_connectome_graph,
    get_connectome_graph_data,
    get_connectome_summary,
)
from app.interventions.fault_detection import FaultDetectionService
from app.interventions.replacement_service import ReplacementService
from app.interventions.sweep import (
    Intervention,
    run_sweep,
    run_replacement_sweep,
    run_replacement_sweep_stream,
    run_live_sweep_stream,
)
from app.metrics.behavior_assays import (
    ForwardLocomotionProtocol,
    forward_locomotion_behavior_spec,
    run_forward_locomotion_assay,
)
from app.metrics.metrics import compute_baseline, failure_score
from app.simulation.factory import get_engine, list_engines
from app.simulation.session import PersistentSimulation

app = FastAPI(title="Synaptech API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------ #
#  Existing routes (unchanged)
# ------------------------------------------------------------------ #


@lru_cache(maxsize=1)
def _replacement_service() -> ReplacementService:
    return ReplacementService(get_connectome_graph())


@lru_cache(maxsize=1)
def _fault_detection_service() -> FaultDetectionService:
    return FaultDetectionService()


@lru_cache(maxsize=1)
def _persistent_simulation() -> PersistentSimulation:
    return PersistentSimulation(get_connectome_graph())


_ACTIVITY_AWARE_STRATEGIES = {"activity_balanced"}


def _estimate_activity_profile(
    graph,
    engine_name: str,
    neuron_model: str,
    burn_in_ms: float,
    duration_ms: float,
) -> dict[str, float]:
    """Estimate per-neuron firing rates over a baseline window."""
    engine = get_engine(engine_name)
    engine.build(graph.copy(), neuron_model=neuron_model)
    engine.run(max(0.0, burn_in_ms))
    engine.run(max(10.0, duration_ms))
    return engine.get_firing_rates()


@app.get("/api/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/message")
def read_message() -> dict[str, str]:
    return {"message": "FastAPI backend is running."}


@app.get("/api/connectome/graph")
def connectome_graph() -> dict[str, object]:
    return get_connectome_graph_data()


@app.get("/api/connectome/summary")
def connectome_summary() -> dict[str, object]:
    return get_connectome_summary()


@app.get("/api/simulation/live")
async def live_simulation(
    step_ms: float = Query(default=500.0, ge=50.0, le=5000.0),
) -> dict:
    """Advance the persistent simulation and return current firing rates."""
    sim = _persistent_simulation()
    return sim.step(step_ms)


@app.post("/api/simulation/session/reset")
async def simulation_session_reset(neuron_model: str = "lif") -> dict:
    """Reset the persistent simulation (rebuilds from scratch)."""
    _persistent_simulation.cache_clear()
    return {"status": "reset"}


# ------------------------------------------------------------------ #
#  Replacement workflow routes
# ------------------------------------------------------------------ #


class ReplacementStartRequest(BaseModel):
    faulty_neuron: str | None = None
    edge_order: Literal["random", "deterministic"] = "random"
    seed: int | None = None
    mode: Literal["instant", "ou"] = "instant"
    theta: float | None = None
    sigma: float | None = None


class ReplacementStepRequest(BaseModel):
    session_id: str
    edges_to_migrate: int = Field(default=1, ge=1, le=100)


@app.get("/api/replacement/faulty/random")
def replacement_faulty_random(
    count: int = Query(default=1, ge=1, le=50),
    seed: int | None = Query(default=None),
) -> dict[str, object]:
    service = _replacement_service()
    detector = _fault_detection_service()
    return {
        "neurons": detector.detect_faulty_neurons(
            graph=service.graph,
            count=count,
            seed=seed,
        ),
        "count": count,
        "source": "random_stub",
    }


@app.post("/api/replacement/start")
def replacement_start(req: ReplacementStartRequest) -> dict[str, object]:
    service = _replacement_service()
    detector = _fault_detection_service()
    faulty = req.faulty_neuron
    if faulty is None:
        faulty_candidates = detector.detect_faulty_neurons(
            graph=service.graph,
            count=1,
            seed=req.seed,
        )
        if not faulty_candidates:
            raise HTTPException(
                status_code=400,
                detail="No candidate neurons available for replacement.",
            )
        faulty = faulty_candidates[0]

    session = service.start_replacement(
        faulty_neuron=faulty,
        edge_order=req.edge_order,
        seed=req.seed,
        mode=req.mode,
        theta=req.theta,
        sigma=req.sigma,
    )
    return {"session": session.to_dict()}


@app.post("/api/replacement/step")
def replacement_step(req: ReplacementStepRequest) -> dict[str, object]:
    service = _replacement_service()
    session = service.step_replacement(
        session_id=req.session_id,
        edges_to_migrate=req.edges_to_migrate,
    )
    return {"session": session.to_dict()}


class ReplacementTickOURequest(BaseModel):
    session_id: str


@app.post("/api/replacement/tick-ou")
def replacement_tick_ou(req: ReplacementTickOURequest) -> dict[str, object]:
    service = _replacement_service()
    session = service.tick_ou(session_id=req.session_id)
    return {"session": session.to_dict()}


@app.get("/api/replacement/session/{session_id}")
def replacement_session(session_id: str) -> dict[str, object]:
    service = _replacement_service()
    return {"session": service.get_session(session_id).to_dict()}


@app.get("/api/replacement/graph")
def replacement_graph() -> dict[str, object]:
    service = _replacement_service()
    return graph_to_data(service.graph)


@app.post("/api/replacement/reset")
def replacement_reset() -> dict[str, str]:
    service = _replacement_service()
    service.reset()
    return {"status": "reset"}


# ------------------------------------------------------------------ #
#  Simulation routes
# ------------------------------------------------------------------ #


class BaselineRequest(BaseModel):
    engine: str = "brian2"
    neuron_model: str = "lif"
    duration_ms: float = 5000.0
    burn_in_ms: float = 2000.0


@app.post("/api/simulation/baseline")
def simulation_baseline(req: BaselineRequest) -> dict:
    graph = get_connectome_graph()
    engine = get_engine(req.engine)
    engine.build(graph, neuron_model=req.neuron_model)

    engine.run(req.burn_in_ms)
    engine.run(req.duration_ms)

    trains = engine.get_spike_trains()
    sensory = [n for n, d in graph.nodes(data=True) if d.get("type") == "S"]
    motor = [n for n, d in graph.nodes(data=True) if d.get("type") == "M"]

    bl = compute_baseline(trains, req.duration_ms, sensory, motor)
    rates = engine.get_firing_rates()

    return {
        "baseline": bl.to_dict(),
        "firing_rates": rates,
        "engine": req.engine,
        "neuron_model": req.neuron_model,
    }


class InterveneRequest(BaseModel):
    engine: str = "brian2"
    neuron_model: str = "lif"
    intervention: Intervention = "dropout"
    fraction: float = Field(0.1, ge=0.0, le=1.0)
    edge_order: Literal["random", "deterministic"] = "random"
    duration_ms: float = 5000.0
    burn_in_ms: float = 2000.0
    seed: int | None = None


@app.post("/api/simulation/intervene")
def simulation_intervene(req: InterveneRequest) -> dict:
    graph = get_connectome_graph()
    rng = stdlib_random.Random(req.seed)
    detector = _fault_detection_service()

    sensory = [n for n, d in graph.nodes(data=True) if d.get("type") == "S"]
    motor = [n for n, d in graph.nodes(data=True) if d.get("type") == "M"]

    # Baseline
    baseline_engine = get_engine(req.engine)
    baseline_engine.build(graph, neuron_model=req.neuron_model)
    baseline_engine.run(req.burn_in_ms)
    baseline_engine.run(req.duration_ms)
    bl_trains = baseline_engine.get_spike_trains()
    baseline = compute_baseline(bl_trains, req.duration_ms, sensory, motor)

    targets = detector.select_targets_by_fraction(
        graph=graph,
        fraction=req.fraction,
        seed=req.seed,
    )

    if req.intervention == "dropout":
        baseline_engine.silence_neurons(targets)
        baseline_engine.run(req.duration_ms)
        post_trains = baseline_engine.get_spike_trains()
        replacement_sessions: list[dict[str, object]] = []
    elif req.intervention == "replacement":
        service = ReplacementService(graph)
        replacement_sessions = []
        for target in targets:
            session = service.start_replacement(
                faulty_neuron=target,
                edge_order=req.edge_order,
                seed=rng.randint(0, 10**9),
            )
            while session.status != "completed":
                session = service.step_replacement(
                    session.session_id,
                    edges_to_migrate=max(1, min(100, len(session.pending))),
                )
            replacement_sessions.append(session.to_dict())

        post_engine = get_engine(req.engine)
        post_engine.build(service.graph, neuron_model=req.neuron_model)
        post_engine.run(req.burn_in_ms)
        post_engine.run(req.duration_ms)
        post_trains = post_engine.get_spike_trains()
    else:
        raise ValueError(f"Unknown intervention '{req.intervention}'")

    post = compute_baseline(post_trains, req.duration_ms, sensory, motor)
    score = failure_score(baseline, post)

    return {
        "baseline": baseline.to_dict(),
        "post_intervention": post.to_dict(),
        "failure_score": score,
        "target_selector": "random_faulty",
        "targeted_neurons": targets,
        "replacement_sessions": replacement_sessions,
        "engine": req.engine,
        "neuron_model": req.neuron_model,
        "intervention": req.intervention,
        "fraction": req.fraction,
    }


class SweepRequest(BaseModel):
    engine: str = "brian2"
    neuron_model: str = "lif"
    intervention: Intervention = "dropout"
    fractions: list[float] = Field(default=[0.01, 0.05, 0.10, 0.20, 0.50])
    n_trials: int = Field(3, ge=1, le=20)
    duration_ms: float = 5000.0
    burn_in_ms: float = 2000.0
    seed: int | None = None


@app.post("/api/simulation/sweep")
def simulation_sweep(req: SweepRequest) -> dict:
    graph = get_connectome_graph()
    results = run_sweep(
        graph=graph,
        fractions=req.fractions,
        intervention=req.intervention,
        n_trials=req.n_trials,
        engine_name=req.engine,
        neuron_model=req.neuron_model,
        burn_in_ms=req.burn_in_ms,
        duration_ms=req.duration_ms,
        seed=req.seed,
    )
    return {
        "results": [r.to_dict() for r in results],
        "engine": req.engine,
        "neuron_model": req.neuron_model,
        "intervention": req.intervention,
    }


class ReplacementSweepRequest(BaseModel):
    engine: str = "brian2"
    neuron_model: str = "hh"
    fraction: float = Field(0.1, ge=0.01, le=1.0)
    strategy: Literal[
        "random",
        "hub_first",
        "periphery_first",
        "peripheral_first",
        "redundancy_aware",
        "synchrony_preserving",
        "function_preserving",
        "activity_balanced",
    ] = "random"
    burn_in_ms: float = 2000.0
    baseline_ms: float = 5000.0
    step_ms: float = 500.0
    edges_per_step: int = Field(1, ge=1, le=50)
    seed: int | None = None
    replacement_mode: Literal["instant", "ou"] = "instant"
    ou_theta: float | None = None
    ou_sigma: float | None = None


@app.post("/api/simulation/replacement-sweep")
def simulation_replacement_sweep(req: ReplacementSweepRequest) -> dict:
    """Run one-by-one neuron replacement with per-step metric snapshots."""
    graph = get_connectome_graph()
    detector = FaultDetectionService()
    activity_profile: dict[str, float] | None = None
    if req.strategy in _ACTIVITY_AWARE_STRATEGIES:
        activity_profile = _estimate_activity_profile(
            graph=graph,
            engine_name=req.engine,
            neuron_model=req.neuron_model,
            burn_in_ms=req.burn_in_ms,
            duration_ms=req.baseline_ms,
        )
    targets = detector.select_targets_by_fraction(
        graph=graph,
        fraction=req.fraction,
        seed=req.seed,
        strategy=req.strategy,
        activity_by_neuron=activity_profile,
    )
    result = run_replacement_sweep(
        graph=graph,
        target_neurons=targets,
        engine_name=req.engine,
        neuron_model=req.neuron_model,
        burn_in_ms=req.burn_in_ms,
        baseline_ms=req.baseline_ms,
        step_ms=req.step_ms,
        edges_per_step=req.edges_per_step,
        seed=req.seed,
        replacement_mode=req.replacement_mode,
        ou_theta=req.ou_theta,
        ou_sigma=req.ou_sigma,
    )
    return result.to_dict()


@app.get("/api/simulation/replacement-sweep/stream")
def simulation_replacement_sweep_stream(
    fraction: float = Query(0.1, ge=0.01, le=1.0),
    strategy: str = "random",
    step_ms: float = 200.0,
    baseline_ms: float = 1000.0,
    edges_per_step: int = Query(5, ge=1, le=50),
    seed: int | None = None,
    replacement_mode: str = "instant",
    ou_theta: float | None = None,
    ou_sigma: float | None = None,
):
    """SSE endpoint: runs replacement sweep on the persistent simulation.

    Uses the already-running engine — no build or burn-in overhead.
    """
    import json
    from starlette.responses import StreamingResponse

    sim = _persistent_simulation()
    detector = FaultDetectionService()
    activity_profile: dict[str, float] | None = None
    if strategy in _ACTIVITY_AWARE_STRATEGIES:
        activity_profile = sim.engine.get_firing_rates()
        if not any(rate > 0.0 for rate in activity_profile.values()):
            sim.step(max(100.0, step_ms))
            activity_profile = sim.engine.get_firing_rates()
    targets = detector.select_targets_by_fraction(
        graph=sim.graph,
        fraction=fraction,
        seed=seed,
        strategy=strategy,
        activity_by_neuron=activity_profile,
    )

    def sse_generator():
        try:
            for event in run_live_sweep_stream(
                sim=sim,
                target_neurons=targets,
                step_ms=step_ms,
                baseline_ms=baseline_ms,
                edges_per_step=edges_per_step,
                seed=seed,
                replacement_mode=replacement_mode,
                ou_theta=ou_theta,
                ou_sigma=ou_sigma,
            ):
                yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
        except Exception as exc:
            event = {
                "type": "error",
                "error": "stream_failed",
                "message": str(exc),
            }
            yield f"event: error\ndata: {json.dumps(event)}\n\n"
            yield "event: done\ndata: {\"type\":\"done\"}\n\n"

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class ForwardProtocolRequest(BaseModel):
    targets: list[str] = Field(default_factory=lambda: ["DB01", "VB01"])
    amplitude_pA: float = Field(default=8.0, ge=0.0, le=200.0)
    period_ms: float = Field(default=200.0, gt=1.0, le=5000.0)
    duty_cycle: float = Field(default=0.5, gt=0.0, le=1.0)
    start_ms: float = Field(default=0.0, ge=0.0)
    stop_ms: float | None = Field(default=None, ge=0.0)


class BehaviorPerformanceRequest(BaseModel):
    behavior: Literal["forward_locomotion"] = "forward_locomotion"
    graph_source: Literal["canonical", "replacement"] = "canonical"
    engine: str = "brian2"
    neuron_model: str = "lif"
    burn_in_ms: float = Field(default=1500.0, ge=0.0, le=20000.0)
    duration_ms: float = Field(default=4000.0, gt=50.0, le=120000.0)
    integration_step_ms: float = Field(default=10.0, gt=0.0, le=500.0)
    include_traces: bool = False
    protocol: ForwardProtocolRequest = Field(default_factory=ForwardProtocolRequest)
    body_curvature: list[float] | None = None
    body_speed_mm_s: list[float] | None = None


@app.get("/api/simulation/behavior/specs")
def simulation_behavior_specs() -> dict[str, object]:
    graph = get_connectome_graph()
    return {"behaviors": [forward_locomotion_behavior_spec(graph)]}


@app.post("/api/simulation/behavior/performance")
def simulation_behavior_performance(req: BehaviorPerformanceRequest) -> dict[str, object]:
    if req.behavior != "forward_locomotion":
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported behavior '{req.behavior}'.",
        )

    if req.graph_source == "replacement":
        graph = _replacement_service().graph.copy()
    else:
        graph = get_connectome_graph()
    stop_ms = req.protocol.stop_ms if req.protocol.stop_ms is not None else req.duration_ms
    protocol = ForwardLocomotionProtocol(
        targets=req.protocol.targets,
        amplitude_pA=req.protocol.amplitude_pA,
        period_ms=req.protocol.period_ms,
        duty_cycle=req.protocol.duty_cycle,
        start_ms=req.protocol.start_ms,
        stop_ms=stop_ms,
    )

    try:
        result = run_forward_locomotion_assay(
            graph=graph,
            engine_name=req.engine,
            neuron_model=req.neuron_model,
            burn_in_ms=req.burn_in_ms,
            duration_ms=req.duration_ms,
            protocol=protocol,
            integration_step_ms=req.integration_step_ms,
            include_traces=req.include_traces,
            body_curvature=req.body_curvature,
            body_speed_mm_s=req.body_speed_mm_s,
        )
        assay_context = result.get("assay_context")
        if isinstance(assay_context, dict):
            assay_context["graph_source"] = req.graph_source
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        if "Clock has reached the end of its available times" in str(exc):
            raise HTTPException(
                status_code=503,
                detail=(
                    "Simulation clock exhausted while running behavioral assay. "
                    "Please retry (and reset session if needed)."
                ),
            ) from exc
        raise


@app.get("/api/simulation/spikes")
async def simulation_spikes(
    duration_ms: float = 5000.0,
) -> dict:
    """Run the persistent simulation forward and return spike trains."""
    sim = _persistent_simulation()
    sim.step(duration_ms)
    trains = sim.engine.get_spike_trains()
    # Strip neurons with no spikes to reduce payload
    sparse = {n: times for n, times in trains.items() if times}
    return {
        "duration_ms": duration_ms,
        "spike_trains": sparse,
        "neuron_count": len(trains),
        "active_count": len(sparse),
        "engine": sim.engine.name,
        "neuron_model": sim.neuron_model,
    }


@app.get("/api/simulation/engines")
def simulation_engines() -> dict:
    return list_engines()


@app.get("/api/simulation/strategies")
def simulation_strategies() -> dict:
    return {"strategies": FaultDetectionService.available_strategies()}
