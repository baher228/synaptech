# Build Plan: Dual Simulation Engine Architecture

> Final reference for the implemented system. See also:
> [TASK_SUMMARY.md](./TASK_SUMMARY.md) (challenge brief),
> [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md) (original concept),
> [APPROACH_COMPARISON.md](./APPROACH_COMPARISON.md) (NumPy vs Brian2 analysis).

---

## What Was Built

Two fully functional simulation engines for the C. elegans nervous system
(302 neurons, ~4,400 synapses), sharing a common protocol so callers never
need to know which is running. The Brian2 engine is wired as the default
in the FastAPI API.

### File Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                              # FastAPI — 4 original + 5 new simulation routes
│   ├── connectome.py                        # NetworkX graph from Cook 2019 CSVs (unchanged)
│   ├── simulation/
│   │   ├── __init__.py
│   │   ├── protocol.py                      # SimulationEngine Protocol
│   │   ├── numpy_engine.py                  # Option A: vectorised NumPy LIF
│   │   ├── brian2_engine.py                 # Option B: Brian2 (LIF / Izhikevich / HH)
│   │   └── factory.py                       # get_engine(name) → engine instance
│   ├── metrics/
│   │   ├── __init__.py
│   │   └── metrics.py                       # Stability metrics + composite failure score
│   └── interventions/
│       ├── __init__.py
│       ├── strategies.py                    # Neuron selection + dropout/replacement/fade
│       └── sweep.py                         # Full sweep orchestration
├── data/roundworm/                          # Cook 2019 chemical + gap junction CSVs
└── requirements.txt                         # fastapi, uvicorn, networkx, numpy, scipy, brian2
```

---

## Simulation Engines

### Shared Protocol (`protocol.py`)

Both engines implement:

| Method | Purpose |
|--------|---------|
| `build(graph, neuron_model)` | Initialise from connectome |
| `run(duration_ms)` | Advance simulation (state persists between calls) |
| `get_spike_trains()` | `{neuron_name: [spike_time_ms, ...]}` |
| `get_firing_rates()` | `{neuron_name: Hz}` |
| `get_voltages()` | `{neuron_name: np.ndarray}` |
| `silence_neurons(names)` | Dropout / kill |
| `set_weights(edges, weights)` | Modify synapse strengths |
| `reset()` | Clear state for fresh run |

### NumPy LIF Engine (`numpy_engine.py`)

- Vectorised Leaky Integrate-and-Fire across all 302 neurons per timestep
- Weight matrices `W_chem` (directed) and `W_gap` (symmetric, voltage-dependent)
- Sensory drive (tonic + Gaussian noise) keeps the network alive
- Dropout via active mask + weight zeroing
- Parameters: V_rest=-65mV, V_thresh=-50mV, tau=20ms, dt=0.5ms

### Brian2 Engine (`brian2_engine.py`)

Three selectable neuron models, all using physical units (amp-based currents):

| Model | Equation type | Parameters cross-checked against |
|-------|--------------|----------------------------------|
| `lif` | Leaky Integrate-and-Fire | c302 Level A (C=3pF, g_leak=0.1nS, thresh=-30mV) |
| `izhikevich` | Quadratic + recovery variable | Izhikevich 2007 (k=0.7nS/mV, Cm=100pF) |
| `hh` | Hodgkin-Huxley (Na/K/leak channels) | Classic HH (gNa=120nS, gK=36nS, dt=0.05ms) |

- Chemical synapses: `on_pre` spike delivery (voltage kick scaled by synapse count)
- Gap junctions: continuous `w_gap * (v_pre - v_post)` current (summed)
- Built-in SpikeMonitor and StateMonitor
- Dropout via per-neuron `is_alive` flag (0/1)
- Segmented runs for mid-simulation interventions

---

## Metrics (`metrics.py`)

| Metric | What it captures |
|--------|-----------------|
| `firing_rate_distribution` | Per-neuron mean Hz |
| `network_synchrony` | Mean pairwise correlation of binned spike trains |
| `shannon_entropy` | Complexity of temporal activity patterns (bits) |
| `pathway_fidelity` | Sensory-to-motor signal propagation (cross-correlation) |
| `failure_score` | Weighted composite [0,1]: 0=healthy, 1=collapsed |

All functions operate on the engine-agnostic `{name: [spike_times]}` format.

---

## Interventions (`strategies.py`)

### Selection strategies

| Strategy | Description |
|----------|-------------|
| `random` | Uniform random selection |
| `hub_targeted` | Highest degree-centrality first (worst case) |
| `hub_sparing` | Lowest degree-centrality first (best case) |
| `type_sensory` / `type_motor` / `type_interneuron` | Target by neuron class |
| `region_head` / `region_body` / `region_tail` | Target by anatomical region |

### Intervention types

| Type | Method |
|------|--------|
| `dropout` | Silence neurons (simulates death) |
| `replacement` | Silence then re-insert with partial connectivity |
| `graceful` | Gradually fade out weights over N steps, then replace |

### Sweep orchestration (`sweep.py`)

Runs the full grid: `fraction × strategy × trial`, returning structured results
with baseline, post-intervention metrics, and failure scores.

---

## API Endpoints

| Route | Method | Purpose |
|-------|--------|---------|
| `/api/connectome/graph` | GET | Full graph data (nodes + edges) |
| `/api/connectome/summary` | GET | Summary statistics |
| `/api/simulation/engines` | GET | List available engines |
| `/api/simulation/strategies` | GET | List selection strategies |
| `/api/simulation/baseline` | POST | Run baseline and return metrics |
| `/api/simulation/intervene` | POST | Single intervention experiment |
| `/api/simulation/sweep` | POST | Full sweep across fractions/strategies |

All simulation endpoints accept `engine` (default `"brian2"`) and
`neuron_model` (default `"lif"`) parameters.

---

## Verified Results

Smoke tests confirmed:

| Engine | Model | 1s run | Active neurons | Total spikes |
|--------|-------|--------|----------------|-------------|
| NumPy | LIF | 100ms | 83/302 | — |
| Brian2 | LIF | 1s | 83/302 | 3,355 |
| Brian2 | Izhikevich | 500ms | 85/302 | 340 |
| Brian2 | HH | 500ms | 86/302 | 86 |

End-to-end intervention (Brian2 LIF, 20% random dropout):
- Baseline: 11.3 Hz mean, synchrony=0.985, entropy=4.27
- Post-dropout: 8.7 Hz mean, synchrony=1.000, entropy≈0
- **Failure score: 0.319**

---

## Dependencies

```
fastapi==0.116.1
uvicorn[standard]==0.35.0
networkx==3.5
numpy
scipy
brian2
```
