# Implementation Plan: Stability Metrics & Live Replacement Infrastructure

> Follow-on from [METRICS.md](METRICS.md) · Extends [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) Phase 3-4

## Context

[METRICS.md](METRICS.md) proposes three stability metrics (Kuramoto Order Parameter, PCA Attractor Deviation, Voltage-State Shannon Entropy) for evaluating neuron replacement strategies in our C. elegans simulation.

The current codebase can't support this as-is:
- The **replacement service** only mutates the NetworkX graph — it is not connected to a running simulation engine
- The **metrics module** works exclusively on spike trains — the proposed metrics require voltage traces
- **Target selection** is random-only — we need hub-first and periphery-first ordering to compare strategies
- The **NumPy engine** is LIF-only and less capable than Brian2 — we are removing it to simplify

This plan bridges those gaps so we can measure network stability at every step of a one-by-one neuron replacement.

### Key design decision: all-to-all pre-allocated synapses

Brian2's `Synapses.connect()` can only be called before the network starts running. To support live edge migration (moving synaptic connections from a faulty neuron to a replacement neuron during simulation), we pre-connect all possible neuron pairs at build time with weight 0, then "activate" connections by setting weights to non-zero. At 350 neurons (302 real + 48 replacement slots) this is ~122K synapses per type — negligible memory.

### Key design decision: spike-phase Kuramoto (not Hilbert)

METRICS.md describes using the Hilbert transform on voltage traces. This works for smooth CTRNN dynamics but not for our LIF/Izhikevich spiking neurons which produce sawtooth waveforms. We use spike-phase extraction instead: assign phase linearly between consecutive spikes (0 -> 2pi), which is the standard approach for spiking networks.

---

## Files to modify

| File | Changes |
|------|---------|
| `backend/app/simulation/numpy_engine.py` | **Delete** |
| `backend/app/simulation/brian2_engine.py` | All-to-all synapses, neuron pool, new methods |
| `backend/app/simulation/protocol.py` | New method signatures |
| `backend/app/simulation/factory.py` | Remove numpy path |
| `backend/app/simulation/__init__.py` | Docstring update |
| `backend/app/connectome.py` | Add `B_CLASS_MOTOR_NEURONS` constant |
| `backend/app/metrics/metrics.py` | 3 new metric functions, expand fingerprint + scoring |
| `backend/app/interventions/fault_detection.py` | Hub-first / periphery-first strategies |
| `backend/app/interventions/replacement_service.py` | Wire to running engine |
| `backend/app/interventions/sweep.py` | New per-step measurement loop |
| `backend/app/main.py` | Update defaults, new endpoint |
| `backend/requirements.txt` | Add `scikit-learn` |

---

## Phase 1 — Independent changes (parallelizable)

### 1A. Remove NumPy engine

- **Delete** `backend/app/simulation/numpy_engine.py`
- **`factory.py`**: Remove `"numpy"` from `_AVAILABLE_ENGINES`, remove the `if name == "numpy"` branch
- **`__init__.py`**: Update docstring ("Brian2/numpy" -> "Brian2")
- **`main.py`**: Change `/api/simulation/spikes` default engine from `"numpy"` to `"brian2"`

### 1B. Add B-class motor neuron constant

**`connectome.py`** — Add after `GABAERGIC_NEURONS`, following the same pattern:

```python
def _b_class_motor_neuron_set() -> set[str]:
    names: set[str] = set()
    names.update(f"VB{i:02d}" for i in range(1, 12))   # VB01-VB11
    names.update(f"DB{i:02d}" for i in range(1, 8))     # DB01-DB07
    return names

B_CLASS_MOTOR_NEURONS: set[str] = _b_class_motor_neuron_set()
```

18 neurons total. Same naming convention as `GABAERGIC_NEURONS` (e.g. `VB01` not `VB1`).

### 1C. Hub-first / periphery-first target selection

**`fault_detection.py`** — Add `strategy` parameter to `detect_faulty_neurons()` and `select_targets_by_fraction()`:

- `"random"` (default): existing behavior
- `"hub_first"`: sort candidates by `degree_centrality` descending, take top N
- `"periphery_first"`: sort ascending, take top N

`degree_centrality` is already computed on every node in `connectome.py`.

---

## Phase 2 — Engine changes + metrics (depends on Phase 1)

### 2A. All-to-all pre-allocated synapses in Brian2

**`brian2_engine.py`** changes:

**Pool size**: Add `_POOL_SIZE = 350` constant (302 real + 48 replacement slots). `self._n` becomes `max(pool_size, len(graph.nodes))`. Track `self._n_real` and `self._next_free_slot`. Mark slots >= `_n_real` as `is_alive = 0`.

**`_build_chemical_synapses()`**: Replace edge-only connect with:
1. Create `_exc_syn` and `_inh_syn` always (never None)
2. Call `.connect()` with no args (all-to-all, N*N synapses per object)
3. Set all weights to 0
4. Loop through graph edges, set non-zero weights via flat index: `syn_idx = si * self._n + ti`

Brian2's parameterless `connect()` produces synapses in row-major order, verified by assertion.

**`_build_gap_junctions()`**: Same all-to-all pattern.

**`set_weights()`**: Replace O(N) mask-based lookup with O(1) direct indexing.

**New methods**:
- `set_gap_weights(edges, weights)` — same as `set_weights()` but for gap junctions
- `activate_neuron(name)` — set `is_alive[idx] = 1`
- `add_neuron_from_slot(replacement_name, copy_params_from=None)` — allocate next free slot, activate, return index
- `get_voltage_matrix(window_ms=None)` — return `(n_real, T)` ndarray; if `window_ms` given, return only the last window

**`protocol.py`**: Add new method signatures.

**`requirements.txt`**: Add `scikit-learn`.

### 2B. Three new voltage-based metrics

**`metrics.py`** — Add three functions:

**`kuramoto_order_parameter(spike_trains, motor_names, duration_ms) -> float`**
- Spike-phase extraction: assign phase linearly 0 -> 2pi between consecutive spikes
- R(t) = |mean(exp(i * phi_j(t)))| across B-class motor population
- Returns mean R over window; 0.0 if insufficient spikes

**`pca_attractor_deviation(baseline_voltages, post_voltages, n_components=3) -> (D, sigma)`**
- PCA fit on baseline, project post onto same axes
- D = mean nearest-point Euclidean distance to baseline trajectory
- sigma = baseline attractor diameter (mean pairwise distance)
- Uses `sklearn.decomposition.PCA`, `scipy.spatial.distance.cdist`

**`voltage_state_entropy(voltages, duration_ms, bin_size_ms=10.0) -> float`**
- Binarize each neuron's voltage vs its temporal median
- Bin into time windows, hash N-bit states, count frequencies
- Returns H in bits

---

## Phase 3 — Integration (depends on Phase 2)

### 3A. Wire ReplacementService to running engine

**`replacement_service.py`** — Add optional `engine` parameter to `__init__`:

- `start_replacement()`: call `engine.add_neuron_from_slot()` to allocate a simulation slot
- `_apply_edge_migration()`: zero old weights, set new weights via `engine.set_weights()` / `set_gap_weights()`
- `_ghost_faulty_neuron()`: call `engine.silence_neurons()`

When `engine is None`, behavior is unchanged (graph-only, for the existing UI workflow).

### 3B. Expand BaselineFingerprint and failure_score

**`metrics.py`** — Add `kuramoto_r`, `pca_deviation`, `pca_sigma`, `voltage_entropy` fields (default 0.0 for backward compat). Update `compute_baseline()` to compute new metrics when voltages provided. Update `failure_score()` with new weight distribution summing to 1.0.

---

## Phase 4 — Orchestration (depends on Phase 3)

### 4A. Per-step measurement loop

**`sweep.py`** — New dataclasses and function:

```python
@dataclass
class StepMetrics:
    step_index: int
    neuron_being_replaced: str
    edges_migrated: int
    total_edges: int
    kuramoto_r: float
    pca_deviation: float
    voltage_entropy: float
    firing_rate_mean: float
    synchrony: float
    pathway_fidelity: float

@dataclass
class ReplacementTimeSeries:
    strategy: str
    neuron_model: str
    replacement_order: list[str]
    baseline: dict
    steps: list[StepMetrics]
```

**`run_replacement_sweep(graph, target_neurons, ...) -> ReplacementTimeSeries`**:
1. Build engine, burn in, capture baseline (voltages + spikes)
2. Create `ReplacementService(graph, engine=engine)`
3. For each neuron in order:
   - `start_replacement()` — allocates engine slot
   - While session not completed:
     - `step_replacement()` — mutates graph AND engine
     - `engine.run(step_ms)` — simulate measurement window
     - Compute all metrics from voltage window + spikes
     - Record `StepMetrics`
4. Return `ReplacementTimeSeries`

Existing `run_sweep()` is left untouched.

### 4B. API endpoint

**`main.py`** — Add `POST /api/simulation/replacement-sweep`:
- Params: `fraction`, `strategy`, `neuron_model`, `step_ms`, `edges_per_step`, `seed`
- Returns `ReplacementTimeSeries` as JSON

---

## Verification

1. **Phase 1**: `B_CLASS_MOTOR_NEURONS` has 18 members. Factory rejects `"numpy"`, accepts `"brian2"`.
2. **Phase 2**: Engine builds with 350*350 synapses. `get_voltage_matrix(window_ms=100)` returns shape `(302, 100)`. Metric unit tests with synthetic data.
3. **Phase 3**: ReplacementService step changes engine weights at expected synapse index. `compute_baseline()` with voltages returns non-zero `kuramoto_r`.
4. **Phase 4**: `run_replacement_sweep(graph, ["AVAL"], step_ms=100)` returns correct number of steps with numeric metrics. API endpoint returns valid JSON.

**End-to-end**: Run a small sweep (2 neurons, hub_first, HH model) and verify metric time-series degrades as expected during replacement.
