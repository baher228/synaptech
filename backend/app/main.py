from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from app.connectome import get_connectome_summary
from app.simulation import run_live_activity

app = FastAPI(title="Synaptech API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/message")
def read_message() -> dict[str, str]:
    return {"message": "FastAPI backend is running."}


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
