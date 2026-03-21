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
from app.interventions.sweep import Intervention, run_sweep
from app.metrics.metrics import compute_baseline, failure_score
from app.simulation import run_live_activity
from app.simulation.factory import get_engine, list_engines

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
def live_simulation(
    duration_ms: float = Query(default=2_000.0, ge=100.0, le=30_000.0),
    burn_in_ms: float = Query(default=500.0, ge=0.0, le=30_000.0),
    seed: int | None = Query(default=7),
) -> dict[str, object]:
    return run_live_activity(
        duration_ms=duration_ms,
        burn_in_ms=burn_in_ms,
        seed=seed,
    )


# ------------------------------------------------------------------ #
#  Replacement workflow routes
# ------------------------------------------------------------------ #


class ReplacementStartRequest(BaseModel):
    faulty_neuron: str | None = None
    edge_order: Literal["random", "deterministic"] = "random"
    seed: int | None = None


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


@app.get("/api/simulation/spikes")
def simulation_spikes(
    engine: str = "numpy",
    neuron_model: str = "lif",
    duration_ms: float = 5000.0,
    burn_in_ms: float = 1000.0,
) -> dict:
    """Run a simulation and return raw spike trains for frontend playback.

    Uses the numpy engine by default for speed (~1s wall-clock).
    """
    graph = get_connectome_graph()
    eng = get_engine(engine)
    eng.build(graph, neuron_model=neuron_model)
    eng.run(burn_in_ms)
    eng.run(duration_ms)
    trains = eng.get_spike_trains()
    # Strip neurons with no spikes to reduce payload
    sparse = {n: times for n, times in trains.items() if times}
    return {
        "duration_ms": duration_ms,
        "spike_trains": sparse,
        "neuron_count": len(trains),
        "active_count": len(sparse),
        "engine": engine,
        "neuron_model": neuron_model,
    }


@app.get("/api/simulation/engines")
def simulation_engines() -> dict:
    return list_engines()


@app.get("/api/simulation/strategies")
def simulation_strategies() -> dict:
    # Backward-compatible endpoint name; now we expose the single selector.
    return {"strategies": ["random_faulty"]}
