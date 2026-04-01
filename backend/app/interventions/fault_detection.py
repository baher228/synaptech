from __future__ import annotations

import hashlib
import random as stdlib_random
from typing import Iterable

import networkx as nx
import numpy as np


class FaultDetectionService:
    """Faulty-neuron selection service.

    Supports strategy-driven target ordering:
    - ``"random"``: uniform random selection (default)
    - ``"hub_first"``: highest hubness first (degree + rich-club tendency)
    - ``"peripheral_first"`` / ``"periphery_first"``: lowest hubness first
    - ``"redundancy_aware"``: replace high-connectivity-redundancy neurons first
    - ``"synchrony_preserving"``: interleave gap-junction clusters, core last
    - ``"function_preserving"``: low locomotion-impact neurons first
    - ``"activity_balanced"``: homeostatic low/mid/high activity interleaving
    """

    STRATEGY_ALIASES = {
        "periphery_first": "peripheral_first",
    }
    SUPPORTED_STRATEGIES = (
        "random",
        "hub_first",
        "peripheral_first",
        "redundancy_aware",
        "synchrony_preserving",
        "function_preserving",
        "activity_balanced",
        "betweenness_first",
        "community_aware",
        "weakest_synapses_first",
    )

    @staticmethod
    def candidate_neurons(graph: nx.DiGraph) -> list[str]:
        return [
            node
            for node, attrs in graph.nodes(data=True)
            if not attrs.get("is_replacement", False) and not attrs.get("is_ghosted", False)
        ]

    @classmethod
    def available_strategies(cls) -> list[str]:
        return list(cls.SUPPORTED_STRATEGIES)

    @classmethod
    def _normalise_strategy(cls, strategy: str) -> str:
        canonical = cls.STRATEGY_ALIASES.get(strategy, strategy)
        if canonical not in cls.SUPPORTED_STRATEGIES:
            supported = ", ".join(cls.SUPPORTED_STRATEGIES)
            raise ValueError(f"Unknown strategy '{strategy}'. Supported strategies: {supported}")
        return canonical

    @staticmethod
    def _stable_noise(name: str, seed: int | None) -> float:
        token = f"{seed if seed is not None else 0}:{name}"
        digest = hashlib.md5(token.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], byteorder="big") / float(2**64)

    @staticmethod
    def _jaccard(a: set[str], b: set[str]) -> float:
        if not a and not b:
            return 1.0
        union = a | b
        if not union:
            return 0.0
        return len(a & b) / len(union)

    @staticmethod
    def _weighted_overlap(a: dict[str, float], b: dict[str, float]) -> float:
        keys = set(a.keys()) | set(b.keys())
        if not keys:
            return 1.0
        mins = 0.0
        maxs = 0.0
        for key in keys:
            av = abs(float(a.get(key, 0.0)))
            bv = abs(float(b.get(key, 0.0)))
            mins += min(av, bv)
            maxs += max(av, bv)
        return mins / max(maxs, 1e-9)

    def _hubness_scores(
        self,
        graph: nx.DiGraph,
        candidates: list[str],
    ) -> dict[str, float]:
        degree_centrality = nx.degree_centrality(graph)
        undirected = graph.to_undirected()
        weighted_degree = {
            node: float(undirected.degree(node, weight="weight"))
            for node in candidates
        }
        values = np.array(list(weighted_degree.values()), dtype=float)
        if values.size == 0:
            return {n: 0.0 for n in candidates}

        threshold = float(np.percentile(values, 90))
        rich_nodes = {n for n, val in weighted_degree.items() if val >= threshold}

        scores: dict[str, float] = {}
        for node in candidates:
            neigh = set(undirected.neighbors(node))
            neigh = {n for n in neigh if n in candidates}
            if neigh:
                rich_neighbor_fraction = len(neigh & rich_nodes) / len(neigh)
            else:
                rich_neighbor_fraction = 0.0
            rich_membership = 1.0 if node in rich_nodes else 0.0
            rich_score = 0.7 * rich_membership + 0.3 * rich_neighbor_fraction
            hubness = float(degree_centrality.get(node, 0.0)) + 0.5 * rich_score
            scores[node] = hubness
        return scores

    def _redundancy_scores(
        self,
        graph: nx.DiGraph,
        candidates: list[str],
    ) -> dict[str, float]:
        # Similarity from neighborhood overlap + weight-profile overlap + neuron type.
        in_sets = {n: set(graph.predecessors(n)) for n in candidates}
        out_sets = {n: set(graph.successors(n)) for n in candidates}
        in_w = {
            n: {src: float(graph[src][n].get("weight", 0.0)) for src in in_sets[n]}
            for n in candidates
        }
        out_w = {
            n: {dst: float(graph[n][dst].get("weight", 0.0)) for dst in out_sets[n]}
            for n in candidates
        }
        ntype = {n: str(graph.nodes[n].get("type", "")) for n in candidates}

        top_k = max(1, min(6, len(candidates) - 1))
        scores: dict[str, float] = {}
        for node in candidates:
            sims: list[float] = []
            for other in candidates:
                if other == node:
                    continue
                same_type = 1.0 if ntype[node] == ntype[other] else 0.0
                j_in = self._jaccard(in_sets[node], in_sets[other])
                j_out = self._jaccard(out_sets[node], out_sets[other])
                w_in = self._weighted_overlap(in_w[node], in_w[other])
                w_out = self._weighted_overlap(out_w[node], out_w[other])
                sim = 0.30 * j_in + 0.30 * j_out + 0.20 * w_in + 0.10 * w_out + 0.10 * same_type
                sims.append(float(sim))
            sims.sort(reverse=True)
            scores[node] = float(np.mean(sims[:top_k])) if sims else 0.0
        return scores

    def _gap_cluster_order(
        self,
        graph: nx.DiGraph,
        candidates: list[str],
    ) -> list[str]:
        gap = nx.Graph()
        gap.add_nodes_from(candidates)
        candidate_set = set(candidates)
        for src, tgt, payload in graph.edges(data=True):
            gw = float(payload.get("gap_weight", 0.0))
            if gw <= 0.0:
                continue
            if src in candidate_set and tgt in candidate_set:
                if gap.has_edge(src, tgt):
                    gap[src][tgt]["weight"] += gw
                else:
                    gap.add_edge(src, tgt, weight=gw)

        # One replacement per cluster per "round"; within each cluster, edge nodes first.
        queues: list[list[str]] = []
        components = list(nx.connected_components(gap))
        for component in components:
            nodes = sorted(component)
            sub = gap.subgraph(nodes)
            core_strength = {
                n: float(sub.degree(n, weight="weight"))
                for n in nodes
            }
            ordered = sorted(nodes, key=lambda n: (core_strength[n], n))
            queues.append(ordered)

        queues.sort(key=len, reverse=True)
        order: list[str] = []
        while any(queues):
            for queue in queues:
                if queue:
                    order.append(queue.pop(0))
        return order

    @staticmethod
    def _locomotion_core_neurons(nodes: Iterable[str]) -> set[str]:
        command_prefixes = ("AVA", "AVB", "AVD", "AVE", "PVC")
        motor_prefixes = ("DB", "VB", "DD", "VD")
        core: set[str] = set()
        for name in nodes:
            if name.startswith(command_prefixes) or name.startswith(motor_prefixes):
                core.add(name)
        return core

    def _functional_importance_scores(
        self,
        graph: nx.DiGraph,
        candidates: list[str],
        lesion_impact_by_neuron: dict[str, float] | None = None,
    ) -> dict[str, float]:
        core = self._locomotion_core_neurons(candidates)
        degree_centrality = nx.degree_centrality(graph)
        lesion_impact_by_neuron = lesion_impact_by_neuron or {}

        scores: dict[str, float] = {}
        for node in candidates:
            is_command = 1.0 if node.startswith(("AVA", "AVB", "AVD", "AVE", "PVC")) else 0.0
            is_motor_chain = 1.0 if node.startswith(("DB", "VB", "DD", "VD")) else 0.0
            is_core = 1.0 if node in core else 0.0
            degree_term = float(degree_centrality.get(node, 0.0))
            lesion_term = float(lesion_impact_by_neuron.get(node, 0.0))
            scores[node] = (
                1.3 * is_command
                + 0.9 * is_motor_chain
                + 0.5 * is_core
                + 0.7 * degree_term
                + 1.2 * lesion_term
            )
        return scores

    def _activity_balanced_order(
        self,
        candidates: list[str],
        activity_by_neuron: dict[str, float] | None,
        seed: int | None = None,
    ) -> list[str]:
        values = np.array(
            [float((activity_by_neuron or {}).get(n, 0.0)) for n in candidates],
            dtype=float,
        )
        if values.size == 0:
            return []
        q1, q2 = np.percentile(values, [33.33, 66.67])

        low: list[tuple[float, float, str]] = []
        mid: list[tuple[float, float, str]] = []
        high: list[tuple[float, float, str]] = []
        for node in candidates:
            val = float((activity_by_neuron or {}).get(node, 0.0))
            item = (val, self._stable_noise(node, seed), node)
            if val <= q1:
                low.append(item)
            elif val <= q2:
                mid.append(item)
            else:
                high.append(item)

        low.sort()
        mid.sort()
        high.sort()

        buckets = [low, mid, high]
        order: list[str] = []
        while any(buckets):
            for bucket in buckets:
                if bucket:
                    order.append(bucket.pop(0)[2])
        return order

    def _betweenness_scores(
        self,
        graph: nx.DiGraph,
        candidates: list[str],
    ) -> dict[str, float]:
        bc = nx.betweenness_centrality(graph)
        return {n: float(bc.get(n, 0.0)) for n in candidates}

    def _community_aware_order(
        self,
        graph: nx.DiGraph,
        candidates: list[str],
    ) -> list[str]:
        """Replace within detected communities before crossing boundaries.

        Uses Louvain on the undirected projection.  Exhausts each community
        (smallest first) before moving to the next; within each community,
        neurons are sorted by degree centrality (low first).
        """
        undirected = graph.to_undirected()
        communities = nx.community.louvain_communities(undirected, seed=42)

        node_to_comm: dict[str, int] = {}
        for i, comm in enumerate(communities):
            for node in comm:
                node_to_comm[node] = i

        comm_groups: dict[int, list[str]] = {}
        for c in candidates:
            ci = node_to_comm.get(c, -1)
            comm_groups.setdefault(ci, []).append(c)

        degree_centrality = nx.degree_centrality(graph)
        for ci in comm_groups:
            comm_groups[ci].sort(key=lambda n: degree_centrality.get(n, 0.0))

        order: list[str] = []
        for ci in sorted(comm_groups, key=lambda ci: len(comm_groups[ci])):
            order.extend(comm_groups[ci])
        return order

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

    def _ordered_candidates(
        self,
        graph: nx.DiGraph,
        candidates: list[str],
        strategy: str,
        seed: int | None = None,
        activity_by_neuron: dict[str, float] | None = None,
        lesion_impact_by_neuron: dict[str, float] | None = None,
    ) -> list[str]:
        strategy = self._normalise_strategy(strategy)

        if strategy == "random":
            rng = stdlib_random.Random(seed)
            shuffled = candidates[:]
            rng.shuffle(shuffled)
            return shuffled

        if strategy in {"hub_first", "peripheral_first"}:
            hubness = self._hubness_scores(graph, candidates)
            reverse = strategy == "hub_first"
            return sorted(
                candidates,
                key=lambda n: (
                    hubness[n],
                    self._stable_noise(n, seed),
                ),
                reverse=reverse,
            )

        if strategy == "redundancy_aware":
            redundancy = self._redundancy_scores(graph, candidates)
            # High redundancy first; unique nodes naturally drift to the tail.
            return sorted(
                candidates,
                key=lambda n: (
                    redundancy[n],
                    self._stable_noise(n, seed),
                ),
                reverse=True,
            )

        if strategy == "synchrony_preserving":
            return self._gap_cluster_order(graph, candidates)

        if strategy == "function_preserving":
            impact = self._functional_importance_scores(
                graph=graph,
                candidates=candidates,
                lesion_impact_by_neuron=lesion_impact_by_neuron,
            )
            return sorted(
                candidates,
                key=lambda n: (
                    impact[n],
                    self._stable_noise(n, seed),
                ),
            )

        if strategy == "activity_balanced":
            return self._activity_balanced_order(
                candidates=candidates,
                activity_by_neuron=activity_by_neuron,
                seed=seed,
            )

        if strategy == "betweenness_first":
            scores = self._betweenness_scores(graph, candidates)
            return sorted(
                candidates,
                key=lambda n: (
                    scores[n],
                    self._stable_noise(n, seed),
                ),
                reverse=True,
            )

        if strategy == "community_aware":
            return self._community_aware_order(graph, candidates)

        if strategy == "weakest_synapses_first":
            weights = self._total_synaptic_weight(graph, candidates)
            return sorted(
                candidates,
                key=lambda n: (
                    weights[n],
                    self._stable_noise(n, seed),
                ),
            )

        # Guard for mypy; all strategies handled by now.
        return candidates

    def detect_faulty_neurons(
        self,
        graph: nx.DiGraph,
        count: int = 1,
        seed: int | None = None,
        strategy: str = "random",
        activity_by_neuron: dict[str, float] | None = None,
        lesion_impact_by_neuron: dict[str, float] | None = None,
    ) -> list[str]:
        if count < 1:
            raise ValueError("count must be >= 1")

        candidates = self.candidate_neurons(graph)
        if not candidates:
            return []
        ordered = self._ordered_candidates(
            graph=graph,
            candidates=candidates,
            strategy=strategy,
            seed=seed,
            activity_by_neuron=activity_by_neuron,
            lesion_impact_by_neuron=lesion_impact_by_neuron,
        )
        return ordered[:count]

    def select_targets_by_fraction(
        self,
        graph: nx.DiGraph,
        fraction: float,
        seed: int | None = None,
        strategy: str = "random",
        activity_by_neuron: dict[str, float] | None = None,
        lesion_impact_by_neuron: dict[str, float] | None = None,
    ) -> list[str]:
        if fraction <= 0.0:
            return []

        candidates = self.candidate_neurons(graph)
        if not candidates:
            return []

        n_target = max(1, int(len(candidates) * fraction))
        return self.detect_faulty_neurons(
            graph=graph,
            count=n_target,
            seed=seed,
            strategy=strategy,
            activity_by_neuron=activity_by_neuron,
            lesion_impact_by_neuron=lesion_impact_by_neuron,
        )
