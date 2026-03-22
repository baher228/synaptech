"""Ornstein-Uhlenbeck gradual replacement process.

Models the replacement synapse's weight convergence as:

    dX_t = theta * (mu - X_t) * dt + sigma * dW_t

where mu is the target weight from the Cook 2019 connectome, theta
controls convergence speed, and sigma captures biological noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.interventions.replacement_service import EdgeMigration
from app.simulation.protocol import SimulationEngine


# Strategy-dependent defaults: hub neurons get slow cautious replacement,
# periphery neurons can be swapped faster.
STRATEGY_DEFAULTS: dict[str, dict[str, float]] = {
    "random": {"theta": 2.0, "sigma": 0.3},
    "hub_first": {"theta": 1.0, "sigma": 0.2},
    "periphery_first": {"theta": 3.0, "sigma": 0.5},
}

DEFAULT_THETA = 2.0
DEFAULT_SIGMA = 0.3


@dataclass
class OUEdgeState:
    """Tracks the OU state for one edge being migrated."""

    new_source: str
    new_target: str
    old_source: str
    old_target: str
    mu_chemical: float
    mu_gap: float
    x_chemical: float = 0.0
    x_gap: float = 0.0
    old_frac: float = 1.0
    converged: bool = False


@dataclass
class OUReplacementManager:
    """Orchestrates OU-based gradual weight transitions for all edges of a neuron."""

    theta: float = DEFAULT_THETA
    sigma: float = DEFAULT_SIGMA
    dt_ms: float = 500.0
    convergence_epsilon: float = 0.05

    _edges: list[OUEdgeState] = field(default_factory=list)
    _rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())

    def __init__(
        self,
        theta: float = DEFAULT_THETA,
        sigma: float = DEFAULT_SIGMA,
        dt_ms: float = 500.0,
        convergence_epsilon: float = 0.05,
        rng_seed: int | None = None,
    ) -> None:
        self.theta = theta
        self.sigma = sigma
        self.dt_ms = dt_ms
        self.convergence_epsilon = convergence_epsilon
        self._edges = []
        self._rng = np.random.default_rng(rng_seed)

    def add_edges(self, migrations: list[EdgeMigration]) -> None:
        """Initialise OU state for each edge migration. New edges start at weight 0."""
        for m in migrations:
            self._edges.append(
                OUEdgeState(
                    new_source=m.new_source,
                    new_target=m.new_target,
                    old_source=m.old_source,
                    old_target=m.old_target,
                    mu_chemical=m.chemical_weight,
                    mu_gap=m.gap_weight,
                    x_chemical=0.0,
                    x_gap=0.0,
                    old_frac=1.0,
                    converged=False,
                )
            )

    def tick(self, engine: SimulationEngine) -> bool:
        """Advance one OU step for all non-converged edges.

        Returns True when ALL edges have converged.
        """
        dt = self.dt_ms / 1000.0  # seconds
        sqrt_dt = np.sqrt(dt)

        for edge in self._edges:
            if edge.converged:
                continue

            # OU update for chemical weight
            if edge.mu_chemical > 0:
                noise = self.sigma * sqrt_dt * self._rng.normal()
                dx = self.theta * (edge.mu_chemical - edge.x_chemical) * dt + noise
                edge.x_chemical = float(
                    np.clip(edge.x_chemical + dx, 0.0, edge.mu_chemical * 1.5)
                )

            # OU update for gap weight
            if edge.mu_gap > 0:
                noise = self.sigma * sqrt_dt * self._rng.normal()
                dx = self.theta * (edge.mu_gap - edge.x_gap) * dt + noise
                edge.x_gap = float(
                    np.clip(edge.x_gap + dx, 0.0, edge.mu_gap * 1.5)
                )

            # Ramp down old edge
            edge.old_frac = max(0.0, edge.old_frac - self.theta * dt)

            # Apply weights to engine
            # Old edge ramps down
            if edge.mu_chemical > 0:
                engine.set_weights(
                    [(edge.old_source, edge.old_target)],
                    [edge.mu_chemical * edge.old_frac],
                )
            if edge.mu_gap > 0:
                engine.set_gap_weights(
                    [(edge.old_source, edge.old_target)],
                    [edge.mu_gap * edge.old_frac],
                )
            # New edge ramps up
            if edge.mu_chemical > 0:
                engine.set_weights(
                    [(edge.new_source, edge.new_target)],
                    [edge.x_chemical],
                )
            if edge.mu_gap > 0:
                engine.set_gap_weights(
                    [(edge.new_source, edge.new_target)],
                    [edge.x_gap],
                )

            # Check convergence
            chem_ok = edge.mu_chemical <= 0 or (
                abs(edge.x_chemical - edge.mu_chemical)
                < self.convergence_epsilon * max(edge.mu_chemical, 1e-6)
            )
            gap_ok = edge.mu_gap <= 0 or (
                abs(edge.x_gap - edge.mu_gap)
                < self.convergence_epsilon * max(edge.mu_gap, 1e-6)
            )
            old_ok = edge.old_frac < self.convergence_epsilon
            edge.converged = chem_ok and gap_ok and old_ok

        return self.all_converged()

    def all_converged(self) -> bool:
        return all(e.converged for e in self._edges)

    def convergence_fraction(self) -> float:
        """Return fraction of edges that have converged (0.0 to 1.0)."""
        if not self._edges:
            return 1.0
        return sum(1 for e in self._edges if e.converged) / len(self._edges)

    def get_state(self) -> list[dict]:
        """Return current state of all edges for API/UI consumption."""
        return [
            {
                "new_source": e.new_source,
                "new_target": e.new_target,
                "old_source": e.old_source,
                "old_target": e.old_target,
                "x_chemical": e.x_chemical,
                "x_gap": e.x_gap,
                "mu_chemical": e.mu_chemical,
                "mu_gap": e.mu_gap,
                "old_frac": e.old_frac,
                "converged": e.converged,
            }
            for e in self._edges
        ]
