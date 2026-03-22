# Next Steps: Integration Strategies, Batch Replacement & Extended Selection

> Building on the 7 selection strategies already implemented, this plan adds two missing experimental axes — **how replacement neurons wire in** and **how many neurons are replaced simultaneously** — plus a few additional selection strategies to round out the comparison matrix.

---

## Overview

| Change | Files touched | What it unlocks |
|--------|--------------|-----------------|
| **A. Integration strategies** | `integration_strategy.py` (new), `replacement_service.py`, `sweep.py`, `main.py` | Vary *how* a replacement neuron connects — the biggest missing experimental axis |
| **B. Batch replacement** | `sweep.py`, `main.py` | Replace N neurons simultaneously, find rate-dependent tipping points |
| **C. Extra selection strategies** | `fault_detection.py`, `main.py` | `betweenness_first`, `community_aware`, `weakest_synapses_first` |

Priority order: **A > B > C** (integration strategies are the most novel and highest-impact for the hackathon).

---

## A. Integration Strategies

### A.1 Create `backend/app/interventions/integration_strategy.py`

Define a `Protocol` and four concrete implementations. Each strategy transforms the list of `EdgeMigration` objects that `_build_edge_migrations()` produces — same interface, different wiring outcomes.

```python
from __future__ import annotations
from typing import Protocol, runtime_checkable
import random as stdlib_random
import networkx as nx
from app.interventions.replacement_service import EdgeMigration


@runtime_checkable
class IntegrationStrategy(Protocol):
    """Transforms edge migrations to vary how a replacement neuron wires in."""

    name: str

    def transform(
        self,
        graph: nx.DiGraph,
        faulty: str,
        replacement: str,
        migrations: list[EdgeMigration],
        rng: stdlib_random.Random,
    ) -> list[EdgeMigration]:
        ...
```

#### A.1.1 `MirrorStrategy` (current behaviour — baseline control)

```python
class MirrorStrategy:
    """Exact copy of all edges with identical weights. Current default."""
    name = "mirror"

    def transform(self, graph, faulty, replacement, migrations, rng):
        return migrations  # no-op
```

No changes to weights or topology. This exists so every sweep can explicitly name its integration strategy, and so the baseline has a label in results.

#### A.1.2 `PartialInheritStrategy`

```python
class PartialInheritStrategy:
    """Copy all edges but scale weights by a fraction.
    Simulates immature synapses that haven't reached full strength."""
    name = "partial_inherit"

    def __init__(self, weight_fraction: float = 0.5):
        self.weight_fraction = weight_fraction

    def transform(self, graph, faulty, replacement, migrations, rng):
        return [
            EdgeMigration(
                migration_id=m.migration_id,
                old_source=m.old_source,
                old_target=m.old_target,
                new_source=m.new_source,
                new_target=m.new_target,
                chemical_weight=m.chemical_weight * self.weight_fraction,
                gap_weight=m.gap_weight * self.weight_fraction,
            )
            for m in migrations
        ]
```

**What it tests:** Does arriving at reduced synaptic strength cause less disruption than instant full-strength replacement? In biology, new synapses strengthen over time (LTP). This models the initial weak-synapse phase.

**Parameter to sweep:** `weight_fraction` in `[0.25, 0.5, 0.75, 1.0]`.

#### A.1.3 `LocalOnlyStrategy`

```python
class LocalOnlyStrategy:
    """Only inherit edges to/from neurons in the same region.
    Long-range connections are dropped entirely."""
    name = "local_only"

    def transform(self, graph, faulty, replacement, migrations, rng):
        faulty_region = graph.nodes[faulty].get("region", "")
        kept = []
        for m in migrations:
            # Identify the *other* neuron (the one that isn't faulty/replacement)
            partner = m.old_target if m.old_source == faulty else m.old_source
            partner_region = graph.nodes.get(partner, {}).get("region", "")
            if partner_region == faulty_region:
                kept.append(m)
        return kept
```

**What it tests:** How much does exact long-range topology matter? If local-only wiring is nearly as good as full mirror, it suggests replacement neurons can get away with only local synaptogenesis — a much easier biological engineering target.

**Edge case:** If the faulty neuron has *only* long-range edges, `kept` will be empty. The replacement neuron will exist but have zero connectivity. This is a valid experimental outcome (total isolation → high disruption), but we should log a warning.

#### A.1.4 `RewireStrategy`

```python
class RewireStrategy:
    """Keep a fraction of original edges; randomly redirect the rest
    to other neurons of the same type (S/I/M) that are within 2 hops."""
    name = "rewire"

    def __init__(self, rewire_fraction: float = 0.3):
        self.rewire_fraction = rewire_fraction

    def transform(self, graph, faulty, replacement, migrations, rng):
        n_rewire = int(len(migrations) * self.rewire_fraction)
        to_rewire = set(rng.sample(range(len(migrations)), min(n_rewire, len(migrations))))

        # Build a pool of valid rewire targets: same type, within 2 hops
        faulty_type = graph.nodes[faulty].get("type", "")
        candidates = [
            n for n, d in graph.nodes(data=True)
            if d.get("type") == faulty_type
            and n != faulty
            and n != replacement
            and not d.get("is_ghosted", False)
        ]

        result = []
        for i, m in enumerate(migrations):
            if i in to_rewire and candidates:
                new_partner = rng.choice(candidates)
                if m.old_source == faulty:
                    # outgoing edge: rewire target
                    result.append(EdgeMigration(
                        migration_id=m.migration_id,
                        old_source=m.old_source,
                        old_target=m.old_target,
                        new_source=replacement,
                        new_target=new_partner,
                        chemical_weight=m.chemical_weight,
                        gap_weight=m.gap_weight,
                    ))
                else:
                    # incoming edge: rewire source
                    result.append(EdgeMigration(
                        migration_id=m.migration_id,
                        old_source=m.old_source,
                        old_target=m.old_target,
                        new_source=new_partner,
                        new_target=replacement,
                        chemical_weight=m.chemical_weight,
                        gap_weight=m.gap_weight,
                    ))
            else:
                result.append(m)
        return result
```

**What it tests:** How much does *exact* connectivity matter vs. approximate degree/type preservation? If 30% rewired edges cause minimal disruption, it means the network is robust to topological noise — the degree distribution matters more than exact wiring.

**Parameter to sweep:** `rewire_fraction` in `[0.1, 0.2, 0.3, 0.5]`.

### A.2 Strategy registry (same file)

```python
INTEGRATION_STRATEGIES: dict[str, type] = {
    "mirror": MirrorStrategy,
    "partial_inherit": PartialInheritStrategy,
    "local_only": LocalOnlyStrategy,
    "rewire": RewireStrategy,
}

def get_integration_strategy(
    name: str,
    **kwargs,
) -> IntegrationStrategy:
    cls = INTEGRATION_STRATEGIES.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown integration strategy '{name}'. "
            f"Available: {list(INTEGRATION_STRATEGIES.keys())}"
        )
    return cls(**kwargs)
```

### A.3 Modify `ReplacementService.start_replacement()`

**File:** `backend/app/interventions/replacement_service.py`

Add an `integration` parameter that accepts either a strategy name (string) or an `IntegrationStrategy` instance. Apply it after `_build_edge_migrations()` returns.

```python
def start_replacement(
    self,
    faulty_neuron: str,
    edge_order: str = "random",
    seed: int | None = None,
    mode: Literal["instant", "ou"] = "instant",
    theta: float | None = None,
    sigma: float | None = None,
    # NEW
    integration: str = "mirror",
    integration_params: dict | None = None,
) -> ReplacementSession:
```

Insert after the existing `migrations = self._build_edge_migrations(...)` call (currently line 138-141):

```python
    migrations = self._build_edge_migrations(
        faulty_neuron=faulty_neuron,
        replacement_neuron=replacement_neuron,
    )

    # --- NEW: apply integration strategy ---
    from app.interventions.integration_strategy import get_integration_strategy
    strategy = get_integration_strategy(integration, **(integration_params or {}))
    rng = stdlib_random.Random(seed)
    migrations = strategy.transform(
        graph=self.graph,
        faulty=faulty_neuron,
        replacement=replacement_neuron,
        migrations=migrations,
        rng=rng,
    )
    # --- END NEW ---

    if edge_order == "random":
        rng_order = stdlib_random.Random(seed)
        rng_order.shuffle(migrations)
    ...
```

**Important:** The integration strategy runs *before* edge ordering and *before* the OU manager sees the migrations. This means OU mode will gradually ramp in the *transformed* edges, not the original ones. This is correct — the integration strategy defines the target topology, the replacement mode defines the speed.

### A.4 Thread through sweep functions

**File:** `backend/app/interventions/sweep.py`

Add `integration` and `integration_params` parameters to:
- `run_replacement_sweep()` (line 195)
- `run_replacement_sweep_stream()` (line 301)
- `run_live_sweep_stream()` (line 413)

All three pass them through to `service.start_replacement()`:

```python
def run_replacement_sweep(
    graph: nx.DiGraph,
    target_neurons: list[str],
    ...
    # NEW
    integration: str = "mirror",
    integration_params: dict | None = None,
) -> ReplacementTimeSeries:
```

In the replacement loop (e.g. line 243):

```python
    session = service.start_replacement(
        faulty_neuron=neuron_name,
        edge_order="random",
        seed=rng.randint(0, 10**9),
        mode=replacement_mode,
        theta=ou_theta,
        sigma=ou_sigma,
        integration=integration,                  # NEW
        integration_params=integration_params,     # NEW
    )
```

Also update `ReplacementTimeSeries` to record the integration strategy used:

```python
@dataclass
class ReplacementTimeSeries:
    strategy: str
    neuron_model: str
    replacement_order: list[str]
    baseline: dict
    integration: str = "mirror"          # NEW
    steps: list[StepMetrics] = field(default_factory=list)
```

### A.5 Expose via API

**File:** `backend/app/main.py`

Add to `ReplacementSweepRequest`:

```python
class ReplacementSweepRequest(BaseModel):
    ...
    integration: str = "mirror"
    integration_params: dict | None = None
```

Pass through in `simulation_replacement_sweep()` and the SSE stream endpoint.

Add an endpoint to list available integration strategies:

```python
@app.get("/api/simulation/integration-strategies")
def simulation_integration_strategies() -> dict:
    from app.interventions.integration_strategy import INTEGRATION_STRATEGIES
    return {"strategies": list(INTEGRATION_STRATEGIES.keys())}
```

### A.6 Frontend dropdown

**File:** `frontend/src/ui/metrics-panel.ts`

Add a second dropdown next to the existing strategy selector. Mirror the pattern used for the selection strategy dropdown (around line 173-181). The dropdown options come from the `/api/simulation/integration-strategies` endpoint, falling back to `["mirror", "partial_inherit", "local_only", "rewire"]` if the fetch fails.

Wire the selected value into the sweep request body as `integration`.

---

## B. Batch Replacement

### B.1 Add `batch_size` and `settle_ms` to sweep functions

**File:** `backend/app/interventions/sweep.py`

Currently the replacement loop in `run_replacement_sweep()` (line 242) iterates neurons one at a time:

```python
for neuron_name in target_neurons:
    session = service.start_replacement(...)
    while session.status != "completed":
        ...
```

Change to batch-based processing:

```python
def run_replacement_sweep(
    graph: nx.DiGraph,
    target_neurons: list[str],
    ...
    # NEW
    batch_size: int = 1,
    settle_ms: float = 0.0,
) -> ReplacementTimeSeries:
```

Replace the replacement loop with:

```python
    # Replacement loop — process neurons in batches
    service = ReplacementService(graph=working_graph, engine=engine)
    steps: list[StepMetrics] = []
    global_step = 0

    # Chunk target_neurons into batches
    batches = [
        target_neurons[i : i + batch_size]
        for i in range(0, len(target_neurons), batch_size)
    ]

    for batch in batches:
        # Start all replacements in this batch simultaneously
        sessions = []
        for neuron_name in batch:
            session = service.start_replacement(
                faulty_neuron=neuron_name,
                edge_order="random",
                seed=rng.randint(0, 10**9),
                mode=replacement_mode,
                theta=ou_theta,
                sigma=ou_sigma,
                integration=integration,
                integration_params=integration_params,
            )
            sessions.append(session)

        # Drive all sessions to completion in lockstep
        while any(s.status != "completed" for s in sessions):
            for session in sessions:
                if session.status == "completed":
                    continue
                if session.mode == "ou":
                    service.tick_ou(session.session_id)
                else:
                    service.step_replacement(
                        session.session_id,
                        edges_to_migrate=edges_per_step,
                    )

            engine.run(step_ms)

            # Measure after each tick (all concurrent replacements contribute)
            trains = engine.get_spike_trains()
            voltages = engine.get_voltage_matrix(window_ms=step_ms)
            rates = firing_rate_distribution(trains, step_ms)
            rate_vals = np.array(list(rates.values()))
            pca_d, pca_s = pca_attractor_deviation(bl_voltages, voltages)

            # Use first active session for edge counts (approximate)
            active = [s for s in sessions if s.status != "completed"]
            edges_migrated = sum(len(s.completed) for s in sessions)
            total_edges = sum(
                len(s.pending) + len(s.completed) for s in sessions
            )
            neurons_in_batch = ", ".join(n for n in batch)

            ou_conv = None
            if any(s.ou_manager is not None for s in sessions):
                convs = [
                    s.ou_manager.convergence_fraction()
                    for s in sessions
                    if s.ou_manager is not None
                ]
                ou_conv = sum(convs) / len(convs) if convs else None

            steps.append(StepMetrics(
                step_index=global_step,
                neuron_being_replaced=neurons_in_batch,
                edges_migrated=edges_migrated,
                total_edges=total_edges,
                kuramoto_r=kuramoto_order_parameter(trains, b_class, step_ms),
                pca_deviation=pca_d,
                pca_sigma=pca_s,
                voltage_entropy=voltage_state_entropy(voltages, step_ms),
                firing_rate_mean=float(np.mean(rate_vals)) if len(rate_vals) else 0.0,
                synchrony=network_synchrony(trains, step_ms),
                pathway_fidelity_val=pathway_fidelity(trains, sensory, motor, step_ms),
                ou_convergence=ou_conv,
            ))
            global_step += 1

        # Settle period between batches
        if settle_ms > 0.0:
            engine.run(settle_ms)
```

**Note on `neuron_being_replaced`:** When batch_size > 1, this field will contain comma-separated neuron names. This is a minor schema change — the frontend tooltip and any downstream parsing should handle the comma-separated format. Alternatively, add a `neurons_in_batch: list[str]` field to `StepMetrics` and keep `neuron_being_replaced` as the first neuron for backward compatibility.

### B.2 Apply the same change to streaming variants

Apply identical batching logic to `run_replacement_sweep_stream()` and `run_live_sweep_stream()`. The structure is the same — just yield `StepMetrics` dicts instead of appending to a list.

### B.3 Update `StepMetrics` (optional but recommended)

```python
@dataclass
class StepMetrics:
    step_index: int
    neuron_being_replaced: str          # kept for backward compat
    neurons_in_batch: list[str] = field(default_factory=list)  # NEW
    batch_index: int = 0                # NEW — which batch this step belongs to
    edges_migrated: int
    total_edges: int
    ...
```

### B.4 API and frontend

**File:** `backend/app/main.py`

Add to `ReplacementSweepRequest`:

```python
class ReplacementSweepRequest(BaseModel):
    ...
    batch_size: int = Field(1, ge=1, le=30)
    settle_ms: float = Field(0.0, ge=0.0, le=10000.0)
```

Pass through to sweep functions.

**Frontend:** Add a "Batch size" numeric input and a "Settle time (ms)" input in the sweep control panel. Default to 1 and 0 respectively (preserving current behaviour).

---

## C. Additional Selection Strategies

### C.1 `betweenness_first`

**File:** `backend/app/interventions/fault_detection.py`

```python
def _betweenness_scores(
    self,
    graph: nx.DiGraph,
    candidates: list[str],
) -> dict[str, float]:
    bc = nx.betweenness_centrality(graph)
    return {n: float(bc.get(n, 0.0)) for n in candidates}
```

This is different from `hub_first` (which uses degree centrality + rich-club). Betweenness measures how often a neuron lies on shortest paths between other neurons — high betweenness = bridge node. Removing bridges can partition the network even if they aren't hubs.

Add to `_ordered_candidates()`:

```python
if strategy == "betweenness_first":
    scores = self._betweenness_scores(graph, candidates)
    return sorted(
        candidates,
        key=lambda n: (scores[n], self._stable_noise(n, seed)),
        reverse=True,
    )
```

### C.2 `community_aware`

```python
def _community_aware_order(
    self,
    graph: nx.DiGraph,
    candidates: list[str],
) -> list[str]:
    """Replace within detected communities before crossing boundaries.
    Uses Louvain on the undirected projection."""
    undirected = graph.to_undirected()
    communities = nx.community.louvain_communities(undirected, seed=42)

    # Group candidates by community
    node_to_comm: dict[str, int] = {}
    for i, comm in enumerate(communities):
        for node in comm:
            node_to_comm[node] = i

    comm_groups: dict[int, list[str]] = {}
    for c in candidates:
        ci = node_to_comm.get(c, -1)
        comm_groups.setdefault(ci, []).append(c)

    # Within each community, sort by degree (low first = less disruptive)
    degree_centrality = nx.degree_centrality(graph)
    for ci in comm_groups:
        comm_groups[ci].sort(key=lambda n: degree_centrality.get(n, 0.0))

    # Exhaust each community before moving to the next
    # Order communities by size (smallest first — less critical clusters)
    order: list[str] = []
    for ci in sorted(comm_groups, key=lambda ci: len(comm_groups[ci])):
        order.extend(comm_groups[ci])
    return order
```

**What it tests:** Whether confining replacement to one community at a time is less disruptive than interleaving across communities. The hypothesis is that replacing within a single module preserves inter-module communication.

### C.3 `weakest_synapses_first`

```python
def _total_synaptic_weight(
    self,
    graph: nx.DiGraph,
    candidates: list[str],
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for n in candidates:
        in_w = sum(
            float(d.get("weight", 0.0))
            for _, _, d in graph.in_edges(n, data=True)
        )
        out_w = sum(
            float(d.get("weight", 0.0))
            for _, _, d in graph.out_edges(n, data=True)
        )
        scores[n] = in_w + out_w
    return scores
```

Add to `_ordered_candidates()`:

```python
if strategy == "weakest_synapses_first":
    weights = self._total_synaptic_weight(graph, candidates)
    return sorted(
        candidates,
        key=lambda n: (weights[n], self._stable_noise(n, seed)),
    )
```

**What it tests:** Whether neurons with weak total synaptic weight are the safest to replace (they contribute least current to the network).

### C.4 Register new strategies

Update `SUPPORTED_STRATEGIES` tuple (line 27-35):

```python
SUPPORTED_STRATEGIES = (
    "random",
    "hub_first",
    "peripheral_first",
    "redundancy_aware",
    "synchrony_preserving",
    "function_preserving",
    "activity_balanced",
    "betweenness_first",      # NEW
    "community_aware",         # NEW
    "weakest_synapses_first",  # NEW
)
```

### C.5 Update API and frontend

**File:** `backend/app/main.py`

Add the three new strategies to the `Literal` type in `ReplacementSweepRequest.strategy` (line 384-393).

**Frontend:** The strategy dropdown in `metrics-panel.ts` should pull from `/api/simulation/strategies` dynamically, so it will pick up the new options automatically. Verify this is the case; if the dropdown is hardcoded, add the new entries.

---

## Testing strategy

### Unit tests

1. **Integration strategies (highest priority):**
   - `MirrorStrategy` returns migrations unchanged
   - `PartialInheritStrategy(0.5)` halves all weights
   - `LocalOnlyStrategy` drops cross-region edges (use a small test graph with known regions)
   - `RewireStrategy(0.3)` changes ~30% of partner neurons, preserves type, preserves total count

2. **Batch replacement:**
   - `batch_size=1` produces identical results to current code (regression test)
   - `batch_size=3` starts 3 sessions before ticking any forward
   - `settle_ms > 0` adds extra simulation time between batches

3. **New selection strategies:**
   - `betweenness_first` returns neurons sorted by betweenness centrality
   - `community_aware` groups neurons by community
   - `weakest_synapses_first` returns lowest-weight neurons first

### Integration test

Run a small sweep (3-5 neurons, `fraction=0.02`) with each integration strategy and verify:
- No crashes
- Metrics are finite (no NaN/Inf)
- `mirror` and current code produce identical metric trajectories (given same seed)

---

## Experimental matrix (what to run for the hackathon demo)

Once all three changes are implemented, the full comparison grid is:

| Axis | Values |
|------|--------|
| **Selection strategy** (7+3) | `random`, `hub_first`, `peripheral_first`, `redundancy_aware`, `synchrony_preserving`, `function_preserving`, `activity_balanced`, `betweenness_first`, `community_aware`, `weakest_synapses_first` |
| **Integration strategy** (4) | `mirror`, `partial_inherit(0.5)`, `local_only`, `rewire(0.3)` |
| **Batch size** (3) | `1`, `3`, `5` |
| **Replacement mode** (2) | `instant`, `ou` |

That's 10 x 4 x 3 x 2 = **240 configurations**. Each produces a time-series of 7 metrics. Even running a subset (e.g. 3 selection x 4 integration x 2 batch x 2 mode = 48 runs) gives a rich dataset for the presentation.

**Key questions each axis answers:**
- **Selection**: "Which neuron should we replace next?"
- **Integration**: "How should the new neuron wire in?"
- **Batch size**: "How many at once is safe?"
- **Replacement mode**: "Should replacement be instantaneous or gradual?"
