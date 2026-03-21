"""Stability metrics and composite failure scoring.

Spike-train metrics accept ``{neuron_name: [spike_time_ms, ...]}``.
Voltage-based metrics accept ``(N, T)`` numpy arrays in mV.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, asdict

import numpy as np
from scipy import stats as sp_stats
from scipy.spatial.distance import cdist


# ------------------------------------------------------------------ #
#  Spike-train metrics (existing)
# ------------------------------------------------------------------ #

def firing_rate_distribution(
    spike_trains: dict[str, list[float]],
    duration_ms: float,
) -> dict[str, float]:
    """Per-neuron mean firing rate in Hz."""
    dur_s = duration_ms / 1000.0
    if dur_s <= 0:
        return {n: 0.0 for n in spike_trains}
    return {name: len(times) / dur_s for name, times in spike_trains.items()}


def _bin_spike_trains(
    spike_trains: dict[str, list[float]],
    duration_ms: float,
    bin_size_ms: float = 10.0,
) -> np.ndarray:
    """Return (n_neurons, n_bins) matrix of spike counts."""
    names = sorted(spike_trains.keys())
    n_bins = max(1, int(np.ceil(duration_ms / bin_size_ms)))
    mat = np.zeros((len(names), n_bins))
    for row, name in enumerate(names):
        for t in spike_trains[name]:
            b = min(int(t / bin_size_ms), n_bins - 1)
            mat[row, b] += 1
    return mat


def network_synchrony(
    spike_trains: dict[str, list[float]],
    duration_ms: float,
    bin_size_ms: float = 10.0,
) -> float:
    """Mean pairwise Pearson correlation of binned spike trains.

    Returns 0.0 when there is insufficient activity to compute correlation.
    """
    mat = _bin_spike_trains(spike_trains, duration_ms, bin_size_ms)
    active_rows = mat.sum(axis=1) > 0
    mat = mat[active_rows]
    if mat.shape[0] < 2:
        return 0.0

    corr = np.corrcoef(mat)
    np.fill_diagonal(corr, np.nan)
    mean_corr = float(np.nanmean(corr))
    return mean_corr if np.isfinite(mean_corr) else 0.0


def shannon_entropy(
    spike_trains: dict[str, list[float]],
    duration_ms: float,
    bin_size_ms: float = 10.0,
) -> float:
    """Shannon entropy of the network spike-count distribution (bits)."""
    mat = _bin_spike_trains(spike_trains, duration_ms, bin_size_ms)
    total_per_bin = mat.sum(axis=0)
    total = total_per_bin.sum()
    if total == 0:
        return 0.0
    p = total_per_bin / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def pathway_fidelity(
    spike_trains: dict[str, list[float]],
    sensory_names: list[str],
    motor_names: list[str],
    duration_ms: float,
    bin_size_ms: float = 10.0,
) -> float:
    """Cross-correlation between aggregate sensory and motor spike trains.

    A high value means sensory input still drives motor output.
    Returns 0.0 if either population is silent.
    """
    n_bins = max(1, int(np.ceil(duration_ms / bin_size_ms)))
    sensory_bins = np.zeros(n_bins)
    motor_bins = np.zeros(n_bins)

    for name in sensory_names:
        for t in spike_trains.get(name, []):
            b = min(int(t / bin_size_ms), n_bins - 1)
            sensory_bins[b] += 1

    for name in motor_names:
        for t in spike_trains.get(name, []):
            b = min(int(t / bin_size_ms), n_bins - 1)
            motor_bins[b] += 1

    if sensory_bins.sum() == 0 or motor_bins.sum() == 0:
        return 0.0

    corr, _ = sp_stats.pearsonr(sensory_bins, motor_bins)
    return float(corr) if np.isfinite(corr) else 0.0


# ------------------------------------------------------------------ #
#  Voltage-based metrics (new)
# ------------------------------------------------------------------ #

def kuramoto_order_parameter(
    spike_trains: dict[str, list[float]],
    motor_neuron_names: list[str],
    duration_ms: float,
    dt_ms: float = 1.0,
) -> float:
    """Kuramoto order parameter R for a motor neuron population.

    Uses spike-phase extraction: assigns phase linearly 0 -> 2*pi between
    consecutive spikes. R(t) = |mean(exp(i * phi_j(t)))| over the population.

    Returns mean R over the window, or 0.0 if fewer than 2 neurons have >= 2 spikes.
    """
    n_steps = max(1, int(duration_ms / dt_ms))

    phases = []
    for name in motor_neuron_names:
        times = sorted(spike_trains.get(name, []))
        if len(times) < 2:
            continue
        phi = np.zeros(n_steps)
        for k in range(len(times) - 1):
            t0 = times[k]
            t1 = times[k + 1]
            i0 = max(0, int(t0 / dt_ms))
            i1 = min(n_steps, int(t1 / dt_ms))
            if i1 <= i0:
                continue
            phi[i0:i1] = np.linspace(0, 2 * np.pi, i1 - i0, endpoint=False)
        # After last spike: no phase info
        last_idx = min(n_steps, int(times[-1] / dt_ms))
        phi[last_idx:] = 0.0
        phases.append(phi)

    if len(phases) < 2:
        return 0.0

    phase_matrix = np.array(phases)  # (n_active_motors, n_steps)
    r_t = np.abs(np.mean(np.exp(1j * phase_matrix), axis=0))
    return float(np.mean(r_t))


def pca_attractor_deviation(
    baseline_voltages: np.ndarray,
    post_voltages: np.ndarray,
    n_components: int = 3,
) -> tuple[float, float]:
    """PCA attractor deviation from baseline trajectory.

    Parameters
    ----------
    baseline_voltages : (N, T_baseline) array in mV
    post_voltages : (N, T_post) array in mV

    Returns
    -------
    (D, sigma) where D = mean nearest-point distance from post to baseline
    in PCA space, sigma = baseline attractor diameter.
    """
    from sklearn.decomposition import PCA

    # Transpose: PCA expects (samples, features) = (T, N)
    bl = baseline_voltages.T
    post = post_voltages.T

    n_comp = min(n_components, bl.shape[1], bl.shape[0])
    if n_comp < 1:
        return 0.0, 0.0

    pca = PCA(n_components=n_comp)
    bl_proj = pca.fit_transform(bl)
    post_proj = pca.transform(post)

    # Attractor diameter: mean pairwise distance in baseline (subsample if large)
    max_samples = 2000
    if bl_proj.shape[0] > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(bl_proj.shape[0], max_samples, replace=False)
        bl_sub = bl_proj[idx]
    else:
        bl_sub = bl_proj
    bl_dists = cdist(bl_sub, bl_sub)
    sigma = float(np.mean(bl_dists[np.triu_indices_from(bl_dists, k=1)]))

    # Nearest-point distance from post to baseline
    dist_matrix = cdist(post_proj, bl_proj)
    min_dists = dist_matrix.min(axis=1)
    D = float(np.mean(min_dists))

    return D, max(sigma, 1e-9)


def voltage_state_entropy(
    voltages: np.ndarray,
    duration_ms: float,
    bin_size_ms: float = 10.0,
) -> float:
    """Shannon entropy of binarized voltage states.

    Each neuron's voltage is compared to its temporal median (adaptive threshold).
    States are binned in time, hashed, and counted to estimate H in bits.
    """
    N, T = voltages.shape
    if N == 0 or T == 0:
        return 0.0

    # Binarize: each neuron vs its own temporal median
    medians = np.median(voltages, axis=1, keepdims=True)
    binary = (voltages > medians).astype(np.uint8)

    # Bin into time windows (majority vote within bin)
    samples_per_bin = max(1, int(bin_size_ms))
    n_bins = max(1, int(np.ceil(T / samples_per_bin)))

    state_hashes = []
    for b in range(n_bins):
        start = b * samples_per_bin
        end = min((b + 1) * samples_per_bin, T)
        if start >= T:
            break
        state = (binary[:, start:end].mean(axis=1) >= 0.5).astype(np.uint8)
        state_hashes.append(state.tobytes())

    if not state_hashes:
        return 0.0

    counts = Counter(state_hashes)
    total = sum(counts.values())
    probs = np.array(list(counts.values())) / total
    return float(-np.sum(probs * np.log2(probs + 1e-15)))


# ------------------------------------------------------------------ #
#  Baseline fingerprint
# ------------------------------------------------------------------ #

@dataclass
class BaselineFingerprint:
    mean_firing_rate: float
    std_firing_rate: float
    synchrony: float
    entropy: float
    pathway_fidelity: float
    # Voltage-based metrics (default 0.0 for backward compat)
    kuramoto_r: float = 0.0
    pca_deviation: float = 0.0
    pca_sigma: float = 0.0
    voltage_entropy: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def compute_baseline(
    spike_trains: dict[str, list[float]],
    duration_ms: float,
    sensory_names: list[str],
    motor_names: list[str],
    bin_size_ms: float = 10.0,
    voltages: np.ndarray | None = None,
    b_class_names: list[str] | None = None,
) -> BaselineFingerprint:
    rates = firing_rate_distribution(spike_trains, duration_ms)
    rate_vals = np.array(list(rates.values()))

    # Spike-based metrics
    mfr = float(np.mean(rate_vals)) if len(rate_vals) else 0.0
    sfr = float(np.std(rate_vals)) if len(rate_vals) else 0.0
    sync = network_synchrony(spike_trains, duration_ms, bin_size_ms)
    ent = shannon_entropy(spike_trains, duration_ms, bin_size_ms)
    pf = pathway_fidelity(spike_trains, sensory_names, motor_names, duration_ms, bin_size_ms)

    # Voltage-based metrics (only when voltages provided)
    k_r = 0.0
    pca_d = 0.0
    pca_s = 0.0
    v_ent = 0.0

    if voltages is not None:
        if b_class_names is not None:
            k_r = kuramoto_order_parameter(spike_trains, b_class_names, duration_ms)
        pca_d, pca_s = pca_attractor_deviation(voltages, voltages)
        v_ent = voltage_state_entropy(voltages, duration_ms)

    return BaselineFingerprint(
        mean_firing_rate=mfr,
        std_firing_rate=sfr,
        synchrony=sync,
        entropy=ent,
        pathway_fidelity=pf,
        kuramoto_r=k_r,
        pca_deviation=pca_d,
        pca_sigma=pca_s,
        voltage_entropy=v_ent,
    )


# ------------------------------------------------------------------ #
#  Failure score
# ------------------------------------------------------------------ #

_DEFAULT_WEIGHTS = {
    "firing_rate": 0.15,
    "synchrony": 0.15,
    "entropy": 0.10,
    "pathway_fidelity": 0.10,
    "kuramoto": 0.20,
    "pca_deviation": 0.15,
    "voltage_entropy": 0.15,
}


def failure_score(
    baseline: BaselineFingerprint,
    current: BaselineFingerprint,
    weights: dict[str, float] | None = None,
) -> float:
    """Composite failure score in [0, 1].

    0 = identical to baseline, 1 = total functional collapse.
    """
    w = weights or _DEFAULT_WEIGHTS

    def _safe_div(a: float, b: float) -> float:
        return abs(a - b) / max(abs(b), 1e-9)

    fr_div = _safe_div(current.mean_firing_rate, baseline.mean_firing_rate)
    sync_loss = max(0.0, baseline.synchrony - current.synchrony)
    ent_change = _safe_div(current.entropy, baseline.entropy)
    path_loss = max(0.0, baseline.pathway_fidelity - current.pathway_fidelity)

    # Kuramoto: drop from baseline R
    kur_loss = max(0.0, baseline.kuramoto_r - current.kuramoto_r)

    # PCA: deviation normalized by attractor diameter
    pca_norm = current.pca_deviation / (2 * baseline.pca_sigma) if baseline.pca_sigma > 0 else 0.0

    # Voltage entropy: relative change
    v_ent_change = _safe_div(current.voltage_entropy, baseline.voltage_entropy)

    raw = (
        w.get("firing_rate", 0) * min(fr_div, 1.0)
        + w.get("synchrony", 0) * min(sync_loss, 1.0)
        + w.get("entropy", 0) * min(ent_change, 1.0)
        + w.get("pathway_fidelity", 0) * min(path_loss, 1.0)
        + w.get("kuramoto", 0) * min(kur_loss, 1.0)
        + w.get("pca_deviation", 0) * min(pca_norm, 1.0)
        + w.get("voltage_entropy", 0) * min(v_ent_change, 1.0)
    )
    return float(np.clip(raw, 0.0, 1.0))
