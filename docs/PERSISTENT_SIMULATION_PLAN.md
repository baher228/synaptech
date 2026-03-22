# Plan: Persistent Simulation Session

## Context

The `/api/simulation/live` endpoint currently builds a **brand new** LIF simulation from scratch on every poll (every 12 seconds), runs it for 2 seconds, returns firing rates, and throws it away. This means:

- The "living organism" illusion is broken — every sample comes from an independent simulation
- Firing rate charts are nearly flat because each 2s simulation converges to the same steady state
- There is no continuity — the network has no memory between polls

We want the C. elegans brain to be **continuously alive** on the backend, with the frontend sampling its evolving state over time.

### Design approach

Follow the existing `ReplacementService` singleton pattern: a `@lru_cache` singleton holds a built Brian2 engine that persists across requests. Each time the frontend polls, the backend **advances** the persistent engine by a small step and returns the current firing rates. The engine's internal state (membrane potentials, synaptic conductances, spike history) carries forward between polls.

---

## Files to modify

| File | Action | Description |
|------|--------|-------------|
| `backend/app/simulation/session.py` | **Create** | `PersistentSimulation` class holding a long-lived Brian2 engine |
| `backend/app/main.py` | Modify | Replace `/api/simulation/live` to use persistent session; add `/api/simulation/session/reset` |
| `frontend/src/simulation/firing-engine.ts` | Modify | Reduce poll interval from 12s to ~2s; remove `durationMs`/`burnInMs`/`seed` params |
| `frontend/src/api.ts` | Modify | Update `fetchLiveSimulation` to match new endpoint signature |
| `frontend/src/types.ts` | Modify | Update `LiveSimulationResponse` if needed |

---

## Phase 1 — Backend: PersistentSimulation

### New file: `backend/app/simulation/session.py`

```python
class PersistentSimulation:
    """Long-lived simulation that persists across HTTP requests."""

    def __init__(self, graph, neuron_model="lif"):
        self.engine = get_engine("brian2")
        self.engine.build(graph, neuron_model=neuron_model)
        self.engine.run(burn_in_ms)  # one-time burn-in
        self._ready = True
        self._step_count = 0

    def step(self, duration_ms=500.0) -> dict:
        """Advance the simulation and return current state."""
        self.engine.run(duration_ms)
        self._step_count += 1
        trains = self.engine.get_spike_trains()
        rates = self.engine.get_firing_rates()
        # Compute and return summary
        ...

    def reset(self, graph, neuron_model="lif"):
        """Rebuild from scratch."""
        ...
```

Key properties:
- `engine.run()` is additive — internal state persists between calls (membrane potentials, synaptic conductances, spike monitors)
- Each `step()` advances by a short window (e.g. 500ms sim time) and returns firing rates from that window
- `get_spike_trains()` already filters to the most recent `run()` window via `_run_start_ms`
- `get_firing_rates()` computes Hz over the most recent window

### Singleton in `main.py`

```python
@lru_cache(maxsize=1)
def _persistent_simulation() -> PersistentSimulation:
    return PersistentSimulation(get_connectome_graph())
```

### Updated `/api/simulation/live`

```python
@app.get("/api/simulation/live")
def simulation_live(step_ms: float = 500.0) -> dict:
    sim = _persistent_simulation()
    return sim.step(step_ms)
```

Each call advances the simulation by `step_ms` and returns current firing rates. No more building from scratch.

### New `/api/simulation/session/reset`

```python
@app.post("/api/simulation/session/reset")
def simulation_session_reset(neuron_model: str = "lif") -> dict:
    _persistent_simulation.cache_clear()
    # Next call to _persistent_simulation() will rebuild
    return {"status": "reset"}
```

---

## Phase 2 — Frontend: Faster polling

### `firing-engine.ts` changes

- Reduce `LIVE_REFRESH_MS` from `12_000` to `2_000` (2 second poll interval)
- Simplify `refreshLiveModel()` — remove `durationMs`, `burnInMs`, `seed` params since the backend handles all of that
- The faster polling + persistent simulation means the firing rate chart updates every 2 seconds with genuine temporal variation

### `api.ts` changes

Simplify `fetchLiveSimulation`:
```typescript
export async function fetchLiveSimulation(): Promise<LiveSimulationResponse> {
  const response = await fetch('/api/simulation/live')
  ...
}
```

No more params needed — the backend manages its own session.

---

## Phase 3 — Return format

The `PersistentSimulation.step()` return value should match `LiveSimulationResponse` to keep the frontend types unchanged:

```python
{
    "node_count": 302,
    "population_spike_rate_hz": float,
    "firing_summary_hz": {
        "overall_mean_hz": float,
        "sensory_mean_hz": float,
        "interneuron_mean_hz": float,
        "motor_mean_hz": float,
        "active_fraction": float,
    },
    "firing_rates_hz_by_node": { "AVAL": 12.5, ... },
    "top_firing_neurons": [ { "name": "AVAL", "firing_rate_hz": 12.5 }, ... ],
}
```

This is the same shape as what `live_lif.py`'s `run_live_activity()` returns. The existing frontend code (`firing-engine.ts` and `metrics-panel.ts`) won't need type changes.

---

## What becomes obsolete

- `backend/app/simulation/live_lif.py` — the standalone numpy LIF simulator. It was only used by the old stateless `/api/simulation/live`. The persistent session uses the Brian2 engine instead. We can keep it for now but it's dead code.

---

## Verification

1. Start backend, call `GET /api/simulation/live` twice in succession. The second call should return different firing rates than the first (the simulation has advanced).
2. Frontend idle chart should show genuine variation over time — rising and falling firing rates as the network dynamics evolve.
3. `POST /api/simulation/session/reset` clears state; next live call rebuilds from scratch.
