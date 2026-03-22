# Synaptech System Overview

## What This Project Does

Synaptech is a full-stack simulation platform for studying **progressive neural replacement** in the *C. elegans* connectome. It addresses a core question in regenerative neuroscience: *can neural tissue be gradually replaced without disrupting memory, identity, or network stability?*

The platform models the complete *C. elegans* nervous system (302 neurons, Cook et al. 2019 connectome), simulates biophysically realistic neural dynamics with Brian2, replaces neurons one-by-one while monitoring three rigorous stability metrics, and visualises the entire process in an interactive browser UI.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    FRONTEND  (Vite + TypeScript)                 │
│                                                                 │
│  Sigma.js graph ── Firing engine ── Metrics panel ── Controls   │
└────────────────────────┬────────────────────────────────────────┘
                         │  HTTP / REST / SSE
┌────────────────────────▼────────────────────────────────────────┐
│                    BACKEND  (FastAPI + Python)                   │
│                                                                 │
│  ┌──────────┐  ┌────────────┐  ┌──────────┐  ┌─────────────┐  │
│  │Connectome│  │ Brian2     │  │ Metrics  │  │ Replacement │  │
│  │  Loader  │→ │ Simulation │→ │ Framework│  │  Service    │  │
│  └──────────┘  │  Engine    │  └──────────┘  └─────────────┘  │
│                └────────────┘                                   │
│                                                                 │
│  Data: Cook et al. 2019 connectome (CSV)                        │
└─────────────────────────────────────────────────────────────────┘
```

The backend is a FastAPI application serving REST endpoints and Server-Sent Events (SSE). The frontend is a Vite + TypeScript single-page application that renders an interactive graph of the connectome and streams live simulation data. Communication is exclusively over HTTP — there is no WebSocket layer.

---

## The Connectome

### Data Source

The connectome comes from Cook et al. (2019), the most complete electron-microscopy reconstruction of the *C. elegans* wiring diagram. It is vendored as CSV files in `backend/data/roundworm/`:

- `chemical_synapse_nodes.csv` / `chemical_synapse_edges.csv` — directed chemical synapses
- `gap_junction_synapse_nodes.csv` / `gap_junction_synapse_edges.csv` — bidirectional electrical synapses (gap junctions)

### Graph Construction (`backend/app/connectome.py`)

On startup, the backend loads both datasets into a single NetworkX `DiGraph` with 302 neurons and ~5,000 edges. Each neuron node carries:

| Attribute | Description |
|-----------|-------------|
| `type` | S (sensory), I (interneuron), or M (motor) |
| `region` | head, body, or tail |
| `degree_centrality` | Normalised degree centrality in the full graph |
| `x_pos`, `y_pos` | Spatial coordinates for visualisation layout |

The 302 neurons break down as: **85 sensory**, **90 interneurons**, **127 motor**.

Each edge carries:

| Attribute | Description |
|-----------|-------------|
| `chemical_weight` | Number of chemical synapses (integer count from EM data) |
| `gap_weight` | Number of gap junctions (integer count) |
| `total` | Sum of chemical + gap weights |

Gap junctions are bidirectional — each one produces two directed edges (A→B and B→A) in the graph.

### Neurotransmitter Classification

Neurons are classified as **GABAergic** (inhibitory) or **cholinergic** (excitatory) based on known neurotransmitter identity. GABAergic neurons include the RME, AVL, DVB, RIS, and DD/VD motor neuron classes. All other neurons default to excitatory (acetylcholine). This classification determines the reversal potential of outgoing chemical synapses.

---

## Simulation Engine (`backend/app/simulation/brian2_engine.py`)

### Overview

The simulation engine wraps Brian2, a clock-driven spiking neural network simulator. It translates the connectome graph into a Brian2 `Network` object with biophysically parameterised neurons and synapses.

### Neuron Models

Three models are available, selectable at build time:

**1. Leaky Integrate-and-Fire (LIF)** — default

The simplest spiking model. Each neuron's membrane voltage evolves as:

```
dv/dt = (g_leak*(E_leak - v) + I_gap + I_exc + I_inh + I_ext) / Cm
```

where:
- `g_leak = 0.1 nS`, `E_leak = -50 mV` — passive leak conductance
- `I_gap` — gap junction current from electrically coupled neighbours
- `I_exc`, `I_inh` — excitatory and inhibitory chemical synaptic currents
- `I_ext = drive + noise_amp * randn()` — external tonic input + stochastic noise
- `Cm = 3 pF` — membrane capacitance (from c302 Level A parameters)

A spike fires when `v > -30 mV` (and the neuron is alive). After spiking, voltage resets to `-50 mV` with a 2 ms absolute refractory period.

**2. Izhikevich** — intermediate complexity, reproduces diverse firing patterns (regular spiking, bursting, chattering).

**3. Hodgkin-Huxley** — full biophysical model with sodium, potassium, and leak ion channels. Slow but accurate.

### External Drive

Sensory neurons receive a strong tonic current (`~3.5 pA`) representing sensory input from the environment. Interneurons and motor neurons receive a weak background current (`~0.12 pA`) plus low-amplitude Gaussian noise. This asymmetry creates a feedforward signal flow: **S → I → M**.

### Synapse Models

**Chemical synapses** use exponential conductance-based dynamics. When a presynaptic neuron spikes, the postsynaptic conductance increments instantly and decays exponentially (τ = 5 ms). The current depends on the difference between membrane voltage and reversal potential:
- Excitatory synapses: reversal at 0 mV (AMPA-like)
- Inhibitory synapses (GABAergic): reversal at -80 mV (GABA-like)

Each synapse from the connectome contributes `0.005 nS` of peak conductance, scaled by the edge weight (synapse count).

**Gap junctions** are modelled as ohmic electrical coupling: `I_gap = g_gap * (v_pre - v_post)`, where `g_gap = 0.005 nS` per junction. They are bidirectional and voltage-dependent — current flows from the more depolarised neuron to the less depolarised one.

### Pre-Allocation for Live Replacement

A key design decision: the engine pre-allocates a **pool of 350 neuron slots** (302 real + 48 replacement) and **all-to-all synapse objects** with initial weight zero. This allows the replacement service to activate new neurons and redirect synaptic connections at runtime by mutating weight values, without needing to reconstruct the Brian2 network.

### Persistent Session (`backend/app/simulation/session.py`)

The simulation runs as a **singleton long-lived session** (held in memory via `@lru_cache`). On first access:

1. The connectome graph is loaded
2. A Brian2 engine is built with the LIF model
3. A 2000 ms burn-in runs to reach steady-state dynamics

Subsequent API calls advance the simulation incrementally (default 500 ms steps). The engine is **thread-safe** via an internal lock — necessary because Brian2 is not reentrant and HTTP requests arrive concurrently.

The persistent session means that membrane potentials, synaptic conductances, and all internal state carry over between API calls. The simulation is continuous, not restarted per request.

---

## Stability Metrics (`backend/app/metrics/metrics.py`)

Three metrics quantify different aspects of circuit stability during replacement. Each captures a distinct failure mode.

### Metric 1: Kuramoto Order Parameter R(t) — "Identity"

Measures **phase synchrony** among the 18 B-class motor neurons (VB01–VB11, DB01–DB07) that drive the forward locomotion wave.

**Computation:**
1. Extract spike trains for B-class motor neurons
2. Compute instantaneous phase φ_j(t) by linear interpolation between consecutive spikes
3. Compute R(t) = |mean(exp(iφ_j(t)))| — the magnitude of the mean complex phase vector

**Interpretation:**
- R = 1.0: perfect synchrony (all neurons fire in lockstep)
- R = 0.0: complete desynchronisation
- Baseline: R ≈ 0.978
- Failure threshold: R < 0.8 — "the worm can no longer crawl"

This is the most intuitive metric: it directly answers *"is the motor pattern still coherent?"*

### Metric 2: PCA Attractor Deviation D(t) — "Dynamics"

Measures whether the network's **global dynamical state** has drifted from its healthy attractor.

**Computation:**
1. Fit PCA (3 components) on the baseline voltage matrix (302 neurons × T timesteps)
2. Project post-replacement voltage trajectories onto the same PCA axes
3. Compute D(t) = Euclidean distance to the nearest point on the baseline trajectory
4. Baseline attractor diameter σ = mean pairwise distance within baseline trajectory

**Interpretation:**
- D < σ: network state remains within the normal attractor
- D > 2σ: "attractor escape" — the network has entered a qualitatively different dynamical regime
- This is the **most sensitive early-warning metric**, detecting drift at replacement fractions as low as 2%

### Metric 3: Shannon Entropy H — "Information"

Measures the **computational diversity** of the network — how many distinct activity patterns it explores.

**Computation:**
1. Binarise each neuron's voltage relative to its own temporal median (adaptive threshold)
2. Bin time into 10 ms windows → each bin is a 302-bit state vector
3. Count unique state frequencies → probability distribution p(x)
4. H = -Σ p(x) log₂ p(x)

**Interpretation:**
- High H: the network visits many distinct states (rich computation)
- Low H: the network is frozen in a few states (computational collapse)
- Baseline: H ≈ 4.7 bits
- Failure threshold: H < 50% of baseline

### Composite Failure Score

For sweep experiments, a weighted composite combines all metrics:

| Metric | Weight |
|--------|--------|
| Kuramoto R | 20% |
| Firing rate change | 15% |
| Network synchrony change | 15% |
| PCA deviation | 15% |
| Voltage entropy change | 15% |
| Shannon entropy change | 10% |
| Pathway fidelity change | 10% |

---

## Replacement Service (`backend/app/interventions/`)

### Target Selection (`fault_detection.py`)

Three strategies determine the order in which neurons are selected for replacement:

- **Random**: uniform random sampling — the typical case
- **Hub-first**: highest degree centrality first — worst-case attack on network robustness
- **Periphery-first**: lowest degree centrality first — best-case, replacing the least-connected neurons

### Replacement Workflow (`replacement_service.py`)

Replacement is **edge-by-edge**, not instantaneous. For each target neuron:

1. **Create replacement neuron**: a new node is added to the graph, copying the original's attributes but with an offset position (golden-angle fan-out for visual separation). In the Brian2 engine, a pre-allocated slot is activated.

2. **List edges to migrate**: all incoming and outgoing edges of the target neuron are enumerated and optionally shuffled (random edge order).

3. **Migrate edges one-by-one**: each migration:
   - Removes the edge from the original (faulty) neuron
   - Creates the same edge on the replacement neuron
   - Preserves both chemical and gap junction weights
   - In Brian2: the old synapse weight is zeroed, the new synapse weight is set

4. **Ghost the original**: once all edges are migrated, the original neuron is marked as "ghosted" (alive flag set to 0 in Brian2, `is_ghosted=True` in the graph). It no longer participates in the network.

This gradual process models a biologically plausible scenario where a replacement cell progressively integrates into the circuit before the original is fully decommissioned.

### Data Model

```
ReplacementSession:
  session_id: str
  faulty_neuron: str          # e.g. "VB01"
  replacement_neuron: str     # e.g. "VB01__rep_000"
  pending: [EdgeMigration]    # edges still to migrate
  completed: [EdgeMigration]  # edges already migrated
  status: "in_progress" | "completed"

EdgeMigration:
  old_source / old_target     # original edge endpoints
  new_source / new_target     # replacement edge endpoints
  chemical_weight / gap_weight
```

---

## Sweep Orchestration (`backend/app/interventions/sweep.py`)

For systematic experiments, the sweep framework runs batch trials across replacement fractions and strategies:

1. Build a fresh Brian2 engine
2. Run burn-in, capture baseline metrics
3. Apply intervention (select neurons by strategy, replace at given fraction)
4. Capture post-intervention metrics
5. Compute failure score

The **SSE streaming sweep** (`/api/simulation/replacement-sweep/stream`) performs replacement in real time on the persistent session, yielding metric updates after each neuron or edge batch. The frontend plots these as live charts.

---

## API Endpoints

### Connectome
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/connectome/graph` | Full graph (nodes + edges) as JSON |
| GET | `/api/connectome/summary` | Neuron counts, type distribution, top hubs |

### Simulation
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/simulation/live` | Poll current firing rates (advances simulation by `step_ms`) |
| POST | `/api/simulation/session/reset` | Reset persistent session |
| GET | `/api/simulation/spikes` | Retrieve spike train data |
| POST | `/api/simulation/baseline` | Run fresh baseline capture |
| POST | `/api/simulation/intervene` | Run intervention on a fresh engine |

### Replacement
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/replacement/faulty/random` | Pick random faulty neuron(s) |
| POST | `/api/replacement/start` | Begin replacement session |
| POST | `/api/replacement/step` | Migrate N edges |
| GET | `/api/replacement/session/{id}` | Query session state |
| GET | `/api/replacement/graph` | Graph with replacements + ghosts |
| POST | `/api/replacement/reset` | Clear all sessions |

### Sweep
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/simulation/replacement-sweep/stream` | SSE stream of metrics during live replacement |
| POST | `/api/simulation/replacement-sweep` | Non-streaming batch sweep |
| POST | `/api/simulation/sweep` | Multi-trial grid sweep |

---

## Frontend

### Technology

The frontend is built with **Vite** and **TypeScript**. It uses **Sigma.js** (GPU-accelerated WebGL) with **Graphology** for graph rendering, and **uPlot** for lightweight real-time charting.

### Boot Sequence (`frontend/src/main.ts`)

1. Fetch the connectome graph (or replacement graph if a session exists)
2. Create the Sigma.js renderer with node colours by type and sizes by centrality
3. Fetch spike train data if available
4. Start the firing engine (replay or live polling mode)
5. Mount UI overlays: legend, timescale control, replacement panel, metrics panel, tooltip

### Graph Visualisation (`frontend/src/graph/renderer.ts`)

Neurons are rendered as coloured circles on a WebGL canvas:
- **Sensory (S)**: one colour
- **Interneurons (I)**: another colour
- **Motor (M)**: a third colour
- Node size is proportional to degree centrality
- Edges represent synaptic connections, weighted by synapse count

Users can click neurons to select them, hover for details, and pan/zoom the graph.

### Firing Engine (`frontend/src/simulation/firing-engine.ts`)

Two modes:

**Replay mode** (when spike data is available): Spike trains are loaded from the backend and replayed at adjustable speed. When a neuron fires, it glows and its postsynaptic targets receive a smaller propagated glow, creating a visual wave of activity across the graph.

**Live polling mode** (fallback): The frontend polls `/api/simulation/live` every 2 seconds. Each response contains per-neuron firing rates from the last simulation step. Neurons are assigned firing probabilities proportional to their rates, producing cosmetic spike animations that reflect real simulation activity.

### Replacement Control (`frontend/src/ui/replacement-control.ts`)

A control panel that drives the replacement workflow:
- **Pick Random Faulty**: selects a target neuron and highlights it
- **Start Replacement**: creates the replacement neuron and displays pending edges
- **Migrate One Edge**: advances replacement by one edge
- **Auto Complete**: loops migration (8 edges every 140 ms) until done
- **Reset Graph**: clears all replacements and restores the original graph

### Metrics Panel (`frontend/src/ui/metrics-panel.ts`)

Operates in two modes:

**Idle mode**: A single chart showing mean firing rates for sensory, interneuron, and motor populations over time, plus a top-10 most active neurons list.

**Sweep mode**: Three separate charts tracking Kuramoto R, PCA deviation, and voltage entropy in real time as neurons are replaced. Data arrives via SSE and is appended to the charts live.

---

## Data Flow Examples

### Live Simulation Polling

```
Frontend (every 2s)
  → GET /api/simulation/live?step_ms=500
  → Backend: advance Brian2 by 500 ms, extract spike trains, compute firing rates
  ← JSON: { mean_hz, active_count, per_neuron_rates, top_10 }
  → Frontend: update node glow intensities, refresh metrics panel
```

### Single Neuron Replacement

```
User clicks "Pick Random Faulty"
  → GET /api/replacement/faulty/random?count=1
  ← { neurons: ["VB03"] }
  → Frontend highlights VB03 in red

User clicks "Start Replacement"
  → POST /api/replacement/start { faulty_neuron: "VB03" }
  ← ReplacementSession { pending: 14 edges, completed: 0 }
  → Frontend shows replacement neuron VB03__rep_000 in blue

User clicks "Migrate One Edge" (repeatedly or via Auto Complete)
  → POST /api/replacement/step { session_id: "...", edges_to_migrate: 1 }
  ← Updated session { pending: 13, completed: 1 }
  → Frontend updates graph: edge moved, progress bar advances

Final step: all edges migrated
  → VB03 ghosted, VB03__rep_000 has all connections
```

### Replacement Sweep with Metrics

```
User sets strategy=hub_first, fraction=0.1, clicks "Run Sweep"
  → Frontend opens SSE: /api/simulation/replacement-sweep/stream?fraction=0.1&strategy=hub_first

Backend:
  1. Burn-in → capture baseline metrics
  ← SSE event "baseline": { kuramoto_r: 0.978, pca_deviation: 0.0, entropy: 4.68 }

  2. For each target neuron (highest centrality first):
     a. Start replacement session
     b. Migrate edges in batches of 5
     c. Run simulation step, compute metrics
     ← SSE event "step": { neuron: "AVAR", kuramoto_r: 0.95, pca_deviation: 0.3, ... }

  3. All neurons replaced
  ← SSE event "done"

Frontend: plots each metric on its own chart as events arrive
```

---

## Running the System

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend dev server runs at `http://localhost:5173` and proxies API requests to `http://localhost:8000`.

---

## Known Limitations

See [SIMULATION_LIMITATIONS.md](SIMULATION_LIMITATIONS.md) for full details. The main issue is that motor neurons and most interneurons are silent in the current LIF simulation because the model lacks proprioceptive feedback, intrinsic excitability diversity, and neuromodulation that the real worm depends on. The metrics framework is correctly implemented and would work on a better-tuned model — the simulation tuning is the bottleneck, not the analysis pipeline.
