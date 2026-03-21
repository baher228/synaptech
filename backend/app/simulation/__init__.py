"""Simulation package: vectorized live LIF, Brian2 engine, factory."""

from app.simulation.factory import get_engine, list_engines
from app.simulation.live_lif import (
    LIFConfig,
    LIFNetworkSimulator,
    LiveDriveConfig,
    run_live_activity,
)

__all__ = [
    "LIFConfig",
    "LIFNetworkSimulator",
    "LiveDriveConfig",
    "get_engine",
    "list_engines",
    "run_live_activity",
]
