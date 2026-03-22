# Simulation Limitations

## Motor Neuron Silence Problem

**Observed:** All 127 motor neurons and most interneurons (85/90) are silent in the LIF simulation. Only sensory neurons fire (55/85 active, 12-44 Hz). The signal propagation chain S→I→M is broken.

**Impact:** Kuramoto R (B-class motor neuron phase synchrony) is permanently 0 because the B-class neurons (VB01-VB11, DB01-DB07) never spike. Additionally, 4 B-class neurons (DB05, DB06, VB08, VB09) have zero incoming edges in the connectome and can never fire regardless of tuning.

### Root Cause

The Cook et al. 2019 connectome is structurally accurate — it maps every synapse from EM reconstruction. The problem is that our simulation lacks several biological drive mechanisms that the real nervous system relies on:

1. **Proprioceptive feedback (biggest gap):** In the real worm, body bending activates stretch-sensitive mechanisms that feed back into motor neurons, creating a self-sustaining locomotion wave. Our simulation has no body, so this loop is entirely absent.

2. **Intrinsic excitability:** Real C. elegans neurons have diverse ion channel profiles. Command interneurons (AVB) are tonically depolarized during forward locomotion — they fire by default and are inhibited to stop. Our LIF model treats all neurons identically.

3. **Neuromodulation:** Neuropeptides and monoamines (serotonin, dopamine, tyramine) pervasively modulate circuit excitability. Absent from our model.

4. **Synaptic strength calibration:** The connectome gives synapse counts, but actual postsynaptic current depends on receptor density, release probability, and receptor type. Our `chemical_scale=1.0` is an arbitrary conversion.

### Current Tuning Parameters (live_lif.py)

| Parameter | Value | Effect |
|-----------|-------|--------|
| `sensory_tonic_current` | 0.95 | Drives sensory neurons — works |
| `background_tonic_current` | 0.12 | Too weak to push I/M neurons over threshold |
| `background_noise_std` | 0.06 | Too low to stochastically trigger firing |
| `chemical_scale` | 1.0 | Insufficient for S→I→M signal relay |

### For Hackathon Purposes

This does not invalidate the metrics framework. The stability metrics (Kuramoto R, PCA deviation, voltage entropy) are correctly implemented and would work on a better-tuned model. Increasing background drive or chemical scale will get motor neurons firing — it won't be a biophysically faithful reproduction of C. elegans locomotion, but it demonstrates the replacement-stability analysis pipeline, which is the point of the challenge.

### References

- Cook et al. 2019 — C. elegans connectome (our dataset)
- OpenWorm project — same connectome, same class of problems, 10+ years of work
- c302 — parameterised C. elegans models at multiple abstraction levels (vendored in this repo)
