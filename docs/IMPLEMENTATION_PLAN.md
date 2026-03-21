# Implementation Plan: Living Worm Brain Replacement Simulator

> Addressing [Challenge 3: Gradual Neural Replacement Without Functional Disruption](./TASK_SUMMARY.md)

## Core Idea

Build the complete C. elegans nervous system (302 neurons, ~7,000 synapses) as a **living simulation** — neurons fire continuously, signals propagate through real connectome topology. Establish baseline "healthy brain" metrics, then progressively replace or drop out neurons and measure when and how the system breaks.

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  Data Layer                      │
│  C. elegans connectome (OpenWorm / Cook 2019)    │
│  302 neurons, ~5300 chemical + ~1750 gap jxns    │
│  Node metadata: type, name, region               │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│              Simulation Engine                   │
│  Leaky Integrate-and-Fire (LIF) neuron model     │
│  Running on real connectome graph (NetworkX)      │
│  Continuous dynamics: input → spike → propagate   │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│             Measurement Layer                    │
│  Baseline capture → Intervention → Failure score │
│  Entropy, synchrony, pathway fidelity, firing    │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│             Visualization / UI                   │
│  Real-time network state, metric dashboards,     │
│  safe vs unstable regime maps                    │
└─────────────────────────────────────────────────┘
```

---

## Phase 1: Build the Brain

### 1.1 Load the Connectome

Source: [OpenWorm Connectome Toolbox](https://openworm.org/ConnectomeToolbox/) — Cook et al. 2019 dataset.

- Download the chemical synapse edgelist (directed, weighted by synapse count)
- Download the gap junction edgelist (undirected, weighted)
- Build a NetworkX DiGraph with 302 neuron nodes
- Attach node metadata:
  - **Type**: sensory (S), motor (M), interneuron (I)
  - **Name**: e.g. AVAL, AVBL, VA01, DA01
  - **Region**: head, body, tail
  - **Degree centrality**: how connected this neuron is (hub score)

### 1.2 Neuron Dynamics Model — Leaky Integrate-and-Fire (LIF)

Each neuron `i` has a membrane potential `V_i(t)` governed by:

```
dV_i/dt = -(V_i - V_rest) / tau + I_syn(t) + I_ext(t)
```

- `V_rest` = resting potential (-65 mV)
- `tau` = membrane time constant (20 ms)
- `I_syn` = synaptic input from connected neurons (weighted sum of incoming spikes)
- `I_ext` = external drive (sensory input or background noise)
- When `V_i > V_threshold` → spike → reset to `V_rest`, propagate to downstream neurons

Chemical synapses: spike in pre-neuron delivers weighted current to post-neuron (directed, excitatory or inhibitory based on neurotransmitter type).

Gap junctions: bidirectional current proportional to voltage difference between coupled neurons.

### 1.3 Make It "Live"

The simulation runs continuously in discrete timesteps (dt = 0.5 ms):

- **Sensory drive**: Inject tonic + noisy current into sensory neurons to mimic environmental input. This is the "heartbeat" that keeps the system alive.
- **Spontaneous activity**: Small background noise current to all neurons to prevent dead silence.
- **Result**: The network settles into a self-sustaining activity pattern — oscillations, propagating waves, and stable firing rate distributions emerge from the real topology.

**Tech**: NumPy vectorized simulation. 302 neurons at 0.5ms timesteps is trivial — thousands of simulated seconds per real second.

---

## Phase 2: Measure the Baseline

Run the living brain for a burn-in period (~5 simulated seconds), then record baseline metrics over a measurement window (~10 seconds):

### Metrics

| Metric | What It Captures | How to Compute |
|--------|-----------------|----------------|
| **Firing rate distribution** | How active each neuron is | Spike count / time per neuron |
| **Network synchrony** | Coordinated activity across the brain | Mean pairwise spike-train correlation |
| **Signal propagation fidelity** | Can signals travel from sensory → motor? | Stimulate sensory neurons, measure motor neuron response latency + reliability |
| **Shannon entropy of activity** | Richness/complexity of network dynamics | Entropy of binned spike count distribution |
| **Functional connectivity** | Which neurons co-activate | Pearson correlation matrix of firing rates → compare to structural connectivity |
| **Attractor stability** | Does the network return to the same state? | Perturb, then measure trajectory divergence |

These become the **"healthy brain" fingerprint** — the reference against which all interventions are scored.

---

## Phase 3: Replacement and Dropout Experiments

### 3.1 Intervention Types

**Dropout (neurodegeneration model):**
- Neuron is silenced — no more firing, all its outgoing synapses go dead
- Simulates neuron death / degeneration

**Replacement (regeneration model):**
- Neuron is removed, then a new neuron is inserted at the same position
- New neuron has *randomized* or *partially restored* connectivity (imperfect integration)
- Simulates a stem-cell-derived replacement neuron that hasn't fully wired in

**Graceful replacement:**
- Old neuron is gradually faded out (decreasing synaptic weights) while new neuron is faded in
- The key question: does gradual handoff preserve stability where abrupt swap doesn't?

### 3.2 Replacement Strategies

| Strategy | Description |
|----------|-------------|
| **Random uniform** | Drop/replace neurons randomly regardless of type |
| **Hub-targeted** | Preferentially hit high-degree hub neurons (worst case) |
| **Hub-sparing** | Protect hubs, replace low-degree neurons first (best case?) |
| **Type-targeted** | Replace only sensory, only motor, or only interneurons |
| **Region-targeted** | Replace neurons in head, body, or tail |
| **Gradual ramp** | Replace 1% per time window, increasing |

### 3.3 Experimental Protocol

For each strategy, sweep replacement fraction from 0% to 100% in steps:

```
for fraction in [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.0]:
    for strategy in [random, hub_targeted, hub_sparing, type_targeted, ...]:
        for trial in range(N_trials):  # stochastic — run multiple times
            1. Load fresh brain, run to baseline
            2. Apply intervention (dropout/replace fraction of neurons using strategy)
            3. Run post-intervention simulation
            4. Compute all metrics
            5. Compare to baseline → failure score
```

---

## Phase 4: Measure Failure

### Failure Score

A composite metric capturing how far the brain has drifted from healthy function:

```
failure_score = w1 * firing_rate_divergence
             + w2 * synchrony_loss
             + w3 * pathway_fidelity_loss
             + w4 * entropy_change
```

Normalized to [0, 1] where 0 = identical to baseline, 1 = total functional collapse.

### Key Outputs (Mapped to Challenge Goals)

| Challenge Goal | Our Output |
|---|---|
| Replacement rate thresholds | Failure score vs. replacement fraction curves — where does the knee occur? |
| Stability boundaries | Phase diagrams: strategy × fraction → stable / degraded / collapsed |
| Network resilience under node substitution | Comparison across strategies — which neurons can you safely replace? |
| Tipping points for dysfunction | The critical fraction where failure score jumps nonlinearly |
| Optimal replacement-rate curves | For gradual replacement: what ramp speed avoids instability? |
| Visualization of safe vs unstable regimes | Heatmaps, phase plots, animated network state |

---

## Phase 5: Visualization

### Dashboard Components

1. **Live network view**: 302 neurons rendered as a force-directed graph, color-coded by firing rate. Replaced/dropped neurons highlighted. Edges pulse on spike transmission.

2. **Failure curve plot**: X = replacement fraction, Y = failure score. Multiple lines for different strategies. The critical finding: where is the tipping point?

3. **Phase diagram heatmap**: Strategy × fraction → color = failure score. Shows the safe operating envelope at a glance.

4. **Metric time series**: Entropy, synchrony, firing rate over simulation time — watch the brain degrade in real time as neurons are replaced.

5. **Neuron importance map**: Each neuron ranked by how much the failure score increases when it's removed. Identifies critical vs. dispensable neurons.

---

## Tech Stack

| Component | Tool | Reason |
|-----------|------|--------|
| Connectome data | OpenWorm CSV + NetworkX | Direct neuron-synapse-neuron graph, no preprocessing |
| Neuron simulation | NumPy (vectorized LIF) | 302 neurons is tiny — no need for Brian2/NEST overhead |
| Metrics | SciPy, NumPy | Entropy, correlation, signal analysis |
| Visualization | Plotly / Dash or React + D3 | Interactive dashboards, real-time network rendering |
| Orchestration | Python | Single language, fast iteration |

---

## Stretch Goals

- **Scale to Drosophila subcircuit** (~1,000-10,000 neurons from FlyWire) to show the framework generalizes beyond C. elegans
- **Replacement learning**: Can a replacement neuron gradually learn its correct connectivity through Hebbian plasticity?
- **Optimal replacement planner**: Given a set of neurons that need replacing, find the order and rate that minimizes peak failure score (optimization / RL)
- **Comparison to theoretical predictions**: Test whether scale-free network theory (targeted hub attack) predicts real connectome vulnerability

---

## Build Order

| Step | Task | Est. Time |
|------|------|-----------|
| 1 | Load C. elegans connectome into NetworkX graph | 30 min |
| 2 | Implement LIF neuron model (vectorized NumPy) | 1-2 hrs |
| 3 | Wire LIF to connectome, add sensory drive — get a living brain | 1-2 hrs |
| 4 | Implement baseline metric capture | 1 hr |
| 5 | Implement dropout/replacement interventions | 1 hr |
| 6 | Run sweep experiments, collect failure curves | 1 hr |
| 7 | Build visualization dashboard | 2-3 hrs |
| 8 | Polish, edge cases, stretch goals | remaining |

**Total core implementation: ~8-10 hours**
