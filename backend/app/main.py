from __future__ import annotations

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.connectome import (
    get_connectome_graph,
    get_connectome_graph_data,
    get_connectome_summary,
)
from app.interventions.strategies import (
    ALL_STRATEGIES,
    Strategy,
    apply_dropout,
    apply_graceful_fade,
    apply_replacement,
    select_neurons,
)
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
    strategy: Strategy = "random"
    fraction: float = Field(0.1, ge=0.0, le=1.0)
    connectivity_restore: float = Field(0.5, ge=0.0, le=1.0)
    duration_ms: float = 5000.0
    burn_in_ms: float = 2000.0
    seed: int | None = None


@app.post("/api/simulation/intervene")
def simulation_intervene(req: InterveneRequest) -> dict:
    import random as stdlib_random

    graph = get_connectome_graph()
    engine = get_engine(req.engine)
    engine.build(graph, neuron_model=req.neuron_model)
    rng = stdlib_random.Random(req.seed)

    sensory = [n for n, d in graph.nodes(data=True) if d.get("type") == "S"]
    motor = [n for n, d in graph.nodes(data=True) if d.get("type") == "M"]

    # Baseline
    engine.run(req.burn_in_ms)
    engine.run(req.duration_ms)
    bl_trains = engine.get_spike_trains()
    bl = compute_baseline(bl_trains, req.duration_ms, sensory, motor)

    # Intervention
    targets = select_neurons(graph, req.fraction, req.strategy, rng=rng)

    if req.intervention == "dropout":
        apply_dropout(engine, targets)
    elif req.intervention == "replacement":
        apply_replacement(engine, graph, targets, req.connectivity_restore, rng=rng)
    elif req.intervention == "graceful":
        apply_graceful_fade(engine, graph, targets)

    # Post-intervention
    engine.run(req.duration_ms)
    post_trains = engine.get_spike_trains()
    post = compute_baseline(post_trains, req.duration_ms, sensory, motor)

    score = failure_score(bl, post)

    return {
        "baseline": bl.to_dict(),
        "post_intervention": post.to_dict(),
        "failure_score": score,
        "targeted_neurons": targets,
        "engine": req.engine,
        "neuron_model": req.neuron_model,
        "intervention": req.intervention,
        "strategy": req.strategy,
        "fraction": req.fraction,
    }


class SweepRequest(BaseModel):
    engine: str = "brian2"
    neuron_model: str = "lif"
    intervention: Intervention = "dropout"
    fractions: list[float] = Field(default=[0.01, 0.05, 0.10, 0.20, 0.50])
    strategies: list[Strategy] = Field(default=["random", "hub_targeted", "hub_sparing"])
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
        strategies=req.strategies,
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
    return {"strategies": ALL_STRATEGIES}
