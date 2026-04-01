"""Integration strategies — vary how a replacement neuron wires into the network.

Each strategy transforms the list of EdgeMigration objects produced by
``ReplacementService._build_edge_migrations()``.  The selection strategy
decides *which* neuron to replace; the integration strategy decides *how*
the replacement connects.
"""

from __future__ import annotations

import random as stdlib_random
from typing import Protocol, runtime_checkable

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
    ) -> list[EdgeMigration]: ...


# ------------------------------------------------------------------ #
#  Concrete strategies
# ------------------------------------------------------------------ #


class MirrorStrategy:
    """Exact copy of all edges with identical weights. Current default."""

    name = "mirror"

    def transform(
        self,
        graph: nx.DiGraph,
        faulty: str,
        replacement: str,
        migrations: list[EdgeMigration],
        rng: stdlib_random.Random,
    ) -> list[EdgeMigration]:
        return migrations


class PartialInheritStrategy:
    """Copy all edges but scale weights by *weight_fraction*.

    Simulates immature synapses that haven't reached full strength.
    """

    name = "partial_inherit"

    def __init__(self, weight_fraction: float = 0.5) -> None:
        self.weight_fraction = weight_fraction

    def transform(
        self,
        graph: nx.DiGraph,
        faulty: str,
        replacement: str,
        migrations: list[EdgeMigration],
        rng: stdlib_random.Random,
    ) -> list[EdgeMigration]:
        f = self.weight_fraction
        return [
            EdgeMigration(
                migration_id=m.migration_id,
                old_source=m.old_source,
                old_target=m.old_target,
                new_source=m.new_source,
                new_target=m.new_target,
                chemical_weight=m.chemical_weight * f,
                gap_weight=m.gap_weight * f,
            )
            for m in migrations
        ]


class LocalOnlyStrategy:
    """Only inherit edges to/from neurons in the same region.

    Long-range connections are dropped entirely.
    """

    name = "local_only"

    def transform(
        self,
        graph: nx.DiGraph,
        faulty: str,
        replacement: str,
        migrations: list[EdgeMigration],
        rng: stdlib_random.Random,
    ) -> list[EdgeMigration]:
        faulty_region = graph.nodes[faulty].get("region", "")
        kept: list[EdgeMigration] = []
        for m in migrations:
            partner = m.old_target if m.old_source == faulty else m.old_source
            partner_region = graph.nodes.get(partner, {}).get("region", "")
            if partner_region == faulty_region:
                kept.append(m)
        return kept


class RewireStrategy:
    """Keep a fraction of original edges; randomly redirect the rest
    to other neurons of the same type (S/I/M)."""

    name = "rewire"

    def __init__(self, rewire_fraction: float = 0.3) -> None:
        self.rewire_fraction = rewire_fraction

    def transform(
        self,
        graph: nx.DiGraph,
        faulty: str,
        replacement: str,
        migrations: list[EdgeMigration],
        rng: stdlib_random.Random,
    ) -> list[EdgeMigration]:
        if not migrations:
            return migrations

        n_rewire = int(len(migrations) * self.rewire_fraction)
        to_rewire = set(
            rng.sample(range(len(migrations)), min(n_rewire, len(migrations)))
        )

        faulty_type = graph.nodes[faulty].get("type", "")
        candidates = [
            n
            for n, d in graph.nodes(data=True)
            if d.get("type") == faulty_type
            and n != faulty
            and n != replacement
            and not d.get("is_ghosted", False)
        ]

        result: list[EdgeMigration] = []
        for i, m in enumerate(migrations):
            if i in to_rewire and candidates:
                new_partner = rng.choice(candidates)
                if m.old_source == faulty:
                    result.append(
                        EdgeMigration(
                            migration_id=m.migration_id,
                            old_source=m.old_source,
                            old_target=m.old_target,
                            new_source=replacement,
                            new_target=new_partner,
                            chemical_weight=m.chemical_weight,
                            gap_weight=m.gap_weight,
                        )
                    )
                else:
                    result.append(
                        EdgeMigration(
                            migration_id=m.migration_id,
                            old_source=m.old_source,
                            old_target=m.old_target,
                            new_source=new_partner,
                            new_target=replacement,
                            chemical_weight=m.chemical_weight,
                            gap_weight=m.gap_weight,
                        )
                    )
            else:
                result.append(m)
        return result


# ------------------------------------------------------------------ #
#  Registry
# ------------------------------------------------------------------ #

INTEGRATION_STRATEGIES: dict[str, type] = {
    "mirror": MirrorStrategy,
    "partial_inherit": PartialInheritStrategy,
    "local_only": LocalOnlyStrategy,
    "rewire": RewireStrategy,
}


def get_integration_strategy(name: str, **kwargs: object) -> IntegrationStrategy:
    cls = INTEGRATION_STRATEGIES.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown integration strategy '{name}'. "
            f"Available: {list(INTEGRATION_STRATEGIES.keys())}"
        )
    return cls(**kwargs)  # type: ignore[call-arg]
