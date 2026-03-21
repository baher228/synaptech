from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.connectome import get_connectome_summary

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
