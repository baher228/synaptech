"""Engine factory — returns the configured simulation backend."""

from __future__ import annotations

from app.simulation.protocol import SimulationEngine


_AVAILABLE_ENGINES = ("brian2",)
_DEFAULT_ENGINE = "brian2"


def get_engine(name: str = _DEFAULT_ENGINE) -> SimulationEngine:
    """Instantiate a simulation engine by name.

    Accepted values: ``"brian2"`` (default).
    """
    if name == "brian2":
        from app.simulation.brian2_engine import Brian2Engine
        return Brian2Engine()

    raise ValueError(
        f"Unknown engine '{name}'. Available: {', '.join(_AVAILABLE_ENGINES)}"
    )


def list_engines() -> dict:
    return {"engines": list(_AVAILABLE_ENGINES), "default": _DEFAULT_ENGINE}
