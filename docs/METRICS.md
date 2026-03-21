# Three Rigorous Stability Metrics for the Living Worm Brain Replacement Simulator

## Implementation Summary

- **Metrics implemented:** All three below, as specified.
- **Applied to:** CTRNN C. elegans simulation (302 neurons, Cook et al. 2019 connectome).
- **Experiments run:** 366 (2 protocols × 3 strategies × 61 replacement fractions).
- **Key result:** Identified precise Critical Replacement Rates (\( R_c \)).

---

## Metric 1: **Kuramoto Order Parameter R(t)** — *"Identity" / Coherence*

**Equation:**
\[
R(t) = \left| \frac{1}{N} \sum_j \exp(i\phi_j(t)) \right|
\]

**Implementation Steps:**
1. Extract voltage traces for 18 B-class motor neurons (VB1–VB11, DB1–DB7).
2. Demean each signal and apply `scipy.signal.hilbert()` to extract the instantaneous phase (\( \phi_j(t) \)).
3. Compute \( R(t) \) as the magnitude of the mean complex phase vector.
4. **Baseline:** \( R = 0.978 \) (strong synchrony; motor neurons fire in coordinated wave).

**Results:**
- **Ablation:** \( R \) drops below 0.8 at 17% (random), 35% (hub-first), 18% (periphery-first) replacement.
- **Shadowing (cross-fade, 50% noise):** \( R \) never drops below 0.8 (min \( R = 0.955 \) at 100% replacement). Motor crawl pattern is preserved.

---

## Metric 2: **PCA Attractor Persistence D(t)** — *"Dynamical" Metric*

**Implementation Steps:**
1. Fit `sklearn.decomposition.PCA(n_components=3)` on baseline voltage matrix (302 × 1000 timepoints).
    - PC1: 97.7% variance, PC2: 2.1%, PC3: 0.15%.
2. Project post-replacement states onto same 3 PCA axes.
3. Compute \( D(t) \): Euclidean distance to the nearest baseline trajectory point (`scipy.spatial.distance.cdist`).
4. **Baseline:** Attractor diameter \( \sigma = 0.528 \); escape threshold = \( 2\sigma = 1.056 \).

**Results:**
- **Ablation:** Attractor escape is immediate (\( D > 1.056 \) at just 2–7% replacement for all strategies). Hub-first: D spikes to 9.7 with only 5% ablated (very sensitive early warning).
- **Shadowing:** \( D \) stays near baseline for most conditions (random/periphery: \( D ≈ 0.7–1.0 \) up to ~75%). Hub-first: noisy hub replacement pushes \( D \) to ~2.3 for 10–30% replacement.

---

## Metric 3: **Shannon Entropy H** — *"Information" / Complexity Metric*

**Equation:**
\[
H = -\sum p(x) \log_2 p(x)
\]

**Implementation Steps:**
1. Binarize voltage matrix: each neuron's voltage compared to its **own temporal median** (adaptive thresholding avoids trivial all-1 or all-0).
2. Temporal binning at 10 ms → 302-bit brain states per bin.
3. Count state frequencies, estimate probability distribution \( p(x) \).
4. Compute entropy with `scipy.stats.entropy(probs, base=2)`.
5. **Baseline:** \( H = 4.68 \) bits (max ~6.8 for 111 time bins).

**Results:**
- **Ablation:** Entropy drops below 50% baseline (\( H < 2.34 \)) at 15% (random), 63% (hub-first), 7% (periphery-first). High ablation: entropy collapses to ~1.0 bits (network "frozen").
- **Shadowing:** Entropy remains stable (range: 3.4–4.8 bits, baseline 4.68) across all strategies, even at 100% replacement.

---

## Critical Replacement Rates (\( R_c \)) — *Complete Table*

| Protocol    | Strategy         | Kuramoto | Attractor | Entropy | First Failure |
|-------------|------------------|----------|-----------|---------|---------------|
| ABLATION    | Random           | 17%      | 2%        | 15%     | 2%            |
|             | Hub-First        | 35%      | 2%        | 63%     | 2%            |
|             | Periphery-First  | 18%      | 7%        | 7%      | 7%            |
| SHADOWING   | Random           | >100%    | 75%       | >100%   | 75%           |
|             | Hub-First        | >100%    | 5%        | >100%   | 5%            |
|             | Periphery-First  | >100%    | 79%       | >100%   | 79%           |

---

## Key Insights for the Hackathon

- **Kuramoto Order Parameter** is the clearest demo metric: directly answers “Can the worm still crawl?” Baseline \( R = 0.978 \). Ablation: degrades to \( R ≈ 0.2–0.7 \). Shadowing: stays at \( R > 0.95 \) even at 100% replacement — a single-number “Ship of Theseus” proof.
- **PCA Attractor Deviation** is the most sensitive early warning: detects first deviation from healthy dynamics (e.g. \( R_c = 2\% \) under ablation). Baseline attractor is tight; any change is immediately seen.
- **Shannon Entropy** measures network information processing. Baseline \( H ≈ 4.7 \) bits. Ablation collapses H (\( H \to 1 \)); shadowing maintains \( H \) near baseline, confirming preservation of computational complexity.
- **Metrics capture different failure modes:** Kuramoto detects motor desynchronization; PCA-D detects global state drift; entropy detects loss of computational diversity.
- **Together, these provide a comprehensive failure fingerprint.**

---

## Output Files

- `three_metrics_replacement_analysis.png` — 6-panel figure (3 metrics × 2 protocols)
- `critical_replacement_rate_three_metrics.png` — Normalized composite showing \( R_c \) for each condition
- `three_metrics_experiment_results.csv` — 366 rows of raw results
- `three_metrics_baseline_values.csv` — Reference values and thresholds

---

## Discretionary Analytical Decisions

- **Kuramoto vs. PLV:** Used Kuramoto Order Parameter, a global synchrony measure (\( R = |\mathrm{mean}(\exp(i\phi))| \)), over pairwise PLV. Kuramoto is a direct, interpretable "collective rhythm" score.
- **Hilbert Transform:** Applied to demeaned raw voltage traces (not sigmoid activations); raw dynamics are more directly oscillatory and suited for `scipy.signal.hilbert`.
- **Adaptive Threshold for Entropy:** Compared each neuron's voltage to its own median; avoids all-1 states and maximizes entropy sensitivity.
- **Temporal Bin Size:** Opted for 10 ms bins for entropy, giving 111 bins in 1 second and 67 unique states (baseline \( H = 5.43 \)), balancing sensitivity and noise.
- **Thresholds:**  
  - Kuramoto: \( R < 0.8 \) = "identity lost" (synchronization literature standard).
  - Attractor: Escape = \( 2\sigma \) from baseline PCA trajectory center.
  - Entropy: Failure if \( H < 50\% \) of baseline.
- **PCA-Distance Robustness:** Used *nearest-point* distance rather than pointwise alignment across trial length/phase.
- **Shadowing Noise:** Used `noise_level=0.5` (50% Gaussian noise on connectivity) as a moderate imperfection scenario.
- **Replacement Batch Size:** 5 neurons per step for full experiment sweep.
- **Cross-fade:** 10 sub-steps (\( \alpha = 0.1, 0.2, ..., 1.0 \)) over 0.5 s used for shadowing protocol.
- **Reproducibility:** Seeded all randomness (replacement order, connectivity noise) with 42.

---