# Approach Comparison: Custom NumPy Engine vs. Brian2 + c302 Data

Two candidate architectures for the simulation engine. Both use the same connectome
(Cook 2019, 302 neurons) and feed into the same intervention/metrics/visualization layers.

---

## Option A: Custom NumPy LIF Engine

Build a vectorised Leaky Integrate-and-Fire simulator from scratch in NumPy.
The full simulation loop, synapse propagation, spike detection, and metric
collection are all our own code.

```
Our CSVs ──→ connectome.py (NetworkX) ──→ NumPy arrays ──→ LIF loop ──→ metrics
```

### What we write ourselves

| Component | Approx. size |
|-----------|-------------|
| LIF membrane equation solver | ~30 lines |
| Chemical synapse propagation | ~20 lines |
| Gap junction current | ~10 lines |
| Spike detection + reset | ~10 lines |
| External / noise current | ~10 lines |
| Spike recording | ~15 lines |
| Metric computation (entropy, synchrony, pathway fidelity) | ~80 lines |
| Intervention logic (dropout, replacement, fade) | ~60 lines |
| Experiment sweep orchestration | ~40 lines |
| **Total** | **~275 lines** |

### Strengths

- **Zero new dependencies.** Only NumPy + SciPy (already implicit via NetworkX).
- **Total control over the simulation loop.** Dropping a neuron mid-timestep,
  fading synapse weights, injecting replacement neurons — all trivial because we
  own every line.
- **Easy to expose to the API.** The loop runs in-process; stream state to
  FastAPI/WebSocket with no IPC or file parsing.
- **Fast at this scale.** 302 neurons × 0.5 ms timestep ≈ millions of steps
  per second on a laptop. No compilation step, no JIT warmup.
- **Easy to debug.** Step through the loop in a debugger. Print any variable at
  any timestep. No abstraction layers to peer through.

### Weaknesses

- **Biophysical credibility.** A hand-rolled LIF is the simplest possible
  neuron model. Reviewers or judges familiar with computational neuroscience may
  question whether the dynamics are realistic enough for the conclusions to hold.
- **No built-in monitors.** We have to build spike recording, rate monitors,
  and state logging ourselves (straightforward but not free).
- **Model flexibility.** Switching to Izhikevich or Hodgkin-Huxley means
  rewriting the core loop. Not hard, but not a one-line change either.
- **Doesn't scale.** If we want to run on a Drosophila subcircuit (10K+ neurons),
  pure NumPy will hit performance limits. Would need to rewrite in a compiled
  framework anyway.
- **Not on the challenge tool list.** The challenge brief explicitly lists
  Brian2, NEST, and NeuroML as suggested tools. A custom engine doesn't tick
  that box.

---

## Option B: Brian2 + c302 Connectome Data

Use Brian2 as the simulation engine. Load the connectome from our existing CSVs
(or c302's data readers). Define neuron equations in Brian2's equation DSL, let
Brian2 handle integration, spike propagation, and monitoring. Our code handles
interventions and metrics on top.

```
Our CSVs ──→ connectome.py (NetworkX) ──→ Brian2 NeuronGroup + Synapses ──→ monitors ──→ metrics
                                              ↑
                                     c302 neuron type classifications,
                                     neurotransmitter signs (exc/inh)
```

### What we write ourselves

| Component | Approx. size |
|-----------|-------------|
| Brian2 network builder (NeuronGroup, Synapses from graph) | ~60 lines |
| Intervention logic (active flags, weight zeroing, re-wiring) | ~50 lines |
| Metric computation (read from Brian2 monitors) | ~60 lines |
| Experiment sweep orchestration | ~40 lines |
| **Total** | **~210 lines** |

### What Brian2 gives us for free

- **Neuron model integration** — Euler, RK2, exponential Euler, exact solvers
- **Spike detection, propagation, synaptic delays**
- **SpikeMonitor, StateMonitor, PopulationRateMonitor** — built-in recording
- **C++ code generation** — equations compile to C++, runs 10-100× faster than
  interpreted Python
- **Unit system** — catches dimensional errors at definition time (mV, ms, nS)
- **Stochastic terms** — `xi` noise in equations, proper Wiener process

### What we take from c302

We don't run c302's NeuroML/jNeuroML pipeline. We use it as a **reference**:

| From c302 | How we use it |
|-----------|--------------|
| Neuron classifications (sensory / motor / interneuron) | Already in our `connectome.py` |
| Neurotransmitter identity per connection (ACh, GABA, etc.) | Determines excitatory vs inhibitory sign — c302's readers include `synclass` |
| Parameter values (membrane tau, threshold, reversal potentials) | Cross-check our Brian2 parameters against c302's Level A / B values |
| Prior art citation | "Our framework extends the c302 model by adding progressive replacement dynamics" |

### Strengths

- **On the challenge tool list.** Brian2 is explicitly listed under "Tools &
  Libraries" in the challenge brief. Using it signals we engaged with the
  recommended ecosystem.
- **Biophysical credibility.** Brian2 is a peer-reviewed simulator (eLife 2019)
  used across computational neuroscience. Results from Brian2 carry more weight
  than a hand-rolled loop.
- **Model flexibility.** Swapping neuron models is changing one string:

  ```python
  # LIF
  eqs = 'dv/dt = -(v - V_rest)/tau + I/Cm : volt'

  # Izhikevich
  eqs = '''dv/dt = (0.04/mV*v**2 + 5*v + 140*mV - u + I*mohm)/ms : volt
           du/dt = a*(b*v - u)/ms : volt'''

  # Hodgkin-Huxley
  eqs = '''dv/dt = (gNa*m**3*h*(ENa-v) + gK*n**4*(EK-v) + gL*(EL-v) + I)/Cm : volt
           dm/dt = ... : 1
           ...'''
  ```

  Same connectome, same interventions, same metrics. Run all three and show
  tipping points are robust across model complexity → very strong result.
- **Scales to larger networks.** Brian2's C++ codegen handles 100K+ neurons
  efficiently. If we demo a Drosophila subcircuit as a stretch goal, no rewrite
  needed.
- **Built-in monitors.** SpikeMonitor, PopulationRateMonitor, StateMonitor
  give us spike trains, firing rates, and membrane voltages with one line each.
- **Gap junction support.** Brian2 has native `Synapses` with continuous
  interaction (`(v_pre - v_post)`) — maps directly to electrical synapses.

### Weaknesses

- **New dependency.** `brian2` is a non-trivial package (~50 MB) with a C++
  compiler requirement for code generation. `pip install brian2` usually works,
  but can fail on some systems if no C compiler is present (falls back to
  slower NumPy mode).
- **Intervention friction.** Brian2 networks are defined at creation time. You
  can't add/remove neurons mid-simulation. Workarounds:
  - **Dropout**: set `neurons.active[i] = False` or threshold to `+inf`
  - **Weight zeroing**: `synapses.w[condition] = 0`
  - **Replacement**: stop → modify → run again (segment-based simulation)

  All feasible, but less fluid than direct array manipulation in NumPy.
- **API integration.** Brian2 simulations produce Monitor objects in-process.
  Extracting data to send over FastAPI/WebSocket requires converting to NumPy
  arrays then JSON — a small but real serialization step.
- **Learning curve.** Brian2 has its own DSL for equations, its own unit system,
  and specific patterns for network construction. Teammates unfamiliar with it
  will need ~30 min to get oriented.
- **Simulation is batch-oriented.** Brian2's `run(duration)` is a blocking call.
  For real-time streaming to a dashboard, we'd run short segments
  (`run(100*ms)`) in a loop and push state after each — slightly more complex
  than a raw NumPy loop where we control every timestep.

---

## Side-by-Side Summary

| Dimension | Option A: NumPy LIF | Option B: Brian2 + c302 data |
|-----------|---------------------|------------------------------|
| **Dependencies** | None new | `brian2` (+ C compiler for codegen) |
| **Lines of custom code** | ~275 | ~210 |
| **Neuron model fidelity** | LIF only (unless we rewrite) | LIF / Izhikevich / HH — one-line swap |
| **Biophysical credibility** | Low — custom loop | High — peer-reviewed simulator |
| **Challenge tool list** | Not listed | Explicitly listed |
| **Mid-sim intervention** | Trivial (array ops) | Feasible (active flags, weight mods, segmented runs) |
| **Performance at 302 neurons** | Very fast (millions of steps/sec) | Very fast (C++ codegen) |
| **Scales to 10K+ neurons** | Slow | Yes |
| **Built-in monitoring** | No — build ourselves | SpikeMonitor, StateMonitor, RateMonitor |
| **API/dashboard integration** | Direct (arrays in-process) | Extract from monitors → serialize |
| **Debug / inspect** | Full control, step through | Brian2 abstraction layer |
| **Multi-model comparison** | Requires rewrite per model | Change equation string, rerun |
| **Time to implement** | ~3-4 hrs (engine + metrics) | ~2-3 hrs (engine + metrics) |
| **Time to learn** | Zero (just NumPy) | ~30 min for Brian2 DSL |

---

## Recommendation

**Use Brian2 (Option B).**

The deciding factors:

1. **It's on the challenge tool list.** Using a suggested tool shows we engaged
   with the brief and chose an appropriate tool for the job.

2. **Multi-model comparison is the killer feature.** Running the same replacement
   experiment on LIF, Izhikevich, and HH — and showing the tipping points are
   consistent — is a result that a hand-rolled LIF engine can't produce without
   significant extra work. This directly addresses the challenge goal of
   identifying robust stability boundaries.

3. **Credibility matters at a hackathon.** "We simulated progressive neural
   replacement in Brian2 on the real C. elegans connectome" lands better with
   judges than "we wrote a NumPy loop." The science is the same, but
   perception matters.

4. **The intervention friction is manageable.** Dropout via active flags, weight
   zeroing, and segmented `run()` calls all work. The 302-neuron network
   rebuilds in milliseconds, so even full network reconstruction between
   intervention steps is viable.

5. **Stretch goal compatibility.** If we get to the Drosophila subcircuit,
   Brian2 handles it natively. NumPy would need a rewrite.

### What we take from c302

- Neurotransmitter sign data (excitatory/inhibitory per connection)
- Parameter cross-validation (membrane tau, threshold values)
- Citation as prior art

### What we build ourselves

- Brian2 network wired from our `connectome.py` graph
- Intervention engine (dropout, replacement, graceful fade)
- Metric computation (entropy, synchrony, pathway fidelity, failure score)
- Experiment sweep orchestration
- Visualization dashboard
