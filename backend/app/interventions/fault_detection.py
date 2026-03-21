from __future__ import annotations

import random as stdlib_random

import networkx as nx


class FaultDetectionService:
    """Faulty-neuron selection service.

    Supports three targeting strategies:
    - ``"random"``: uniform random selection (default)
    - ``"hub_first"``: highest degree_centrality first (worst-case attack)
    - ``"periphery_first"``: lowest degree_centrality first
    """

    @staticmethod
    def candidate_neurons(graph: nx.DiGraph) -> list[str]:
        return [
            node
            for node, attrs in graph.nodes(data=True)
            if not attrs.get("is_replacement", False) and not attrs.get("is_ghosted", False)
        ]

    def detect_faulty_neurons(
        self,
        graph: nx.DiGraph,
        count: int = 1,
        seed: int | None = None,
        strategy: str = "random",
    ) -> list[str]:
        if count < 1:
            raise ValueError("count must be >= 1")

        candidates = self.candidate_neurons(graph)
        if not candidates:
            return []

        if strategy == "hub_first":
            ranked = sorted(
                candidates,
                key=lambda n: graph.nodes[n].get("degree_centrality", 0.0),
                reverse=True,
            )
            return ranked[:count]
        elif strategy == "periphery_first":
            ranked = sorted(
                candidates,
                key=lambda n: graph.nodes[n].get("degree_centrality", 0.0),
            )
            return ranked[:count]
        else:  # "random"
            rng = stdlib_random.Random(seed)
            if count >= len(candidates):
                return candidates
            return rng.sample(candidates, count)

    def select_targets_by_fraction(
        self,
        graph: nx.DiGraph,
        fraction: float,
        seed: int | None = None,
        strategy: str = "random",
    ) -> list[str]:
        if fraction <= 0.0:
            return []

        candidates = self.candidate_neurons(graph)
        if not candidates:
            return []

        n_target = max(1, int(len(candidates) * fraction))
        return self.detect_faulty_neurons(
            graph=graph, count=n_target, seed=seed, strategy=strategy,
        )
