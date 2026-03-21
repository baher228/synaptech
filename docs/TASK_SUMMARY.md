# Challenge 3: Gradual Neural Replacement Without Functional Disruption

**Submitted by:** RESCUE – Imperial College London

**Contact Person:** Roberto Portillo Lara ([r.portillo-lara@imperial.ac.uk](mailto:r.portillo-lara@imperial.ac.uk)) [IN PERSON MENTOR]

**Challenge Summary**

How can neural tissue be gradually replaced without disrupting memory, identity, or network stability? This challenge seeks computational models that simulate progressive neural replacement and identify safe intervention strategies in ageing or degenerative brains.

**Background & Motivation**

Ageing and neurodegeneration involve progressive loss of neurons and network coherence. Regenerative strategies may allow tissue replacement, but abrupt or poorly timed integration risks destabilizing circuit dynamics.

The brain is a dynamic system. Replacing even a small proportion of nodes (neurons) may shift oscillatory regimes, alter attractor states, or disrupt information encoding.

If neural replacement is to become viable, we must understand:

- Replacement rate thresholds
- Stability boundaries
- Network resilience under progressive node substitution

This is a systems-level modelling challenge aligned with the Replacement theme of the hackathon.

**Challenge Goal**

Develop simulation frameworks that:

- Model progressive neuron replacement in graph-based networks
- Quantify stability metrics (entropy, coherence, attractor persistence)
- Identify safe replacement trajectories
- Predict tipping points for dysfunction

Expected Outputs:

- A dynamical network simulation platform
- Stability and entropy metrics
- Optimal replacement-rate curves
- Visualization of safe vs unstable regimes

**Available Resources**

Datasets:

- [Allen Mouse Brain Connectivity Atlas](https://connectivity.brain-map.org/)
- [Human Connectome Project](https://www.humanconnectome.org/study/hcp-young-adult)
- [Mouse Connectome Project](https://www.mouseconnectome.org/)
- [OpenNeuro](https://openneuro.org/)

Tools & Libraries:

- [NetworkX](https://networkx.org/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [Brian2](https://brian2.readthedocs.io/)
- [NEST Simulator](https://www.nest-simulator.org/)
- [NeuroML](https://www.opensourcebrain.org/)

**Suggested Literature**

- Sporns, Olaf. “Networks of the Brain”
- Deco et al., “The Dynamic Brain” (PLoS Computational Biology)
- Tononi et al., “Integrated Information Theory: From Consciousness to Its Physical Substrate” (Nature Reviews Neuroscience)

**Potential Impact**

This framework could define the engineering principles required for safe neural rejuvenation, which are central to defeating cognitive entropy.