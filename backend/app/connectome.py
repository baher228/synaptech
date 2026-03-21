from __future__ import annotations

import csv
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import networkx as nx

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "roundworm"
CHEMICAL_NODES_PATH = DATA_ROOT / "chemical_synapse.csv" / "nodes.csv"
CHEMICAL_EDGES_PATH = DATA_ROOT / "chemical_synapse.csv" / "edges.csv"
GAP_NODES_PATH = DATA_ROOT / "gap_junction_synapse.csv" / "nodes.csv"
GAP_EDGES_PATH = DATA_ROOT / "gap_junction_synapse.csv" / "edges.csv"

CANONICAL_NODE_COUNT = 302

PHARYNGEAL_NEURONS = {
    "I1L",
    "I1R",
    "I2L",
    "I2R",
    "I3",
    "I4",
    "I5",
    "I6",
    "M1",
    "M2L",
    "M2R",
    "M3L",
    "M3R",
    "M4",
    "M5",
    "MCL",
    "MCR",
    "MI",
    "NSML",
    "NSMR",
}

NON_NEURON_MOTOR_CELLS = {
    "CEPshDL",
    "CEPshDR",
    "CEPshVL",
    "CEPshVR",
    "GLRDL",
    "GLRDR",
    "GLRL",
    "GLRR",
    "GLRVL",
    "GLRVR",
    "exc_gl",
    "exc_cell",
    "hmc",
    "hyp",
    "mu_intL",
    "mu_intR",
    "mu_anal",
    "mu_sph",
}


def _gabaergic_neuron_set() -> set[str]:
    """Known GABAergic (inhibitory) neurons in C. elegans.

    Sources: McIntire et al. 1993, WormAtlas.  Conservative subset covering
    RME head motor neurons, RIS interneuron, AVL/DVB defecation circuit,
    DD01-DD06 dorsal D-type motor neurons, VD01-VD13 ventral D-type motor neurons.
    """
    names = {"AVL", "DVB", "RMEV", "RMED", "RMEL", "RMER", "RIS"}
    names.update({f"DD{idx:02d}" for idx in range(1, 7)})
    names.update({f"VD{idx:02d}" for idx in range(1, 14)})
    return names


GABAERGIC_NEURONS: set[str] = _gabaergic_neuron_set()

BODY_PREFIXES = (
    "DA",
    "DB",
    "DD",
    "VA",
    "VB",
    "VD",
    "AS",
)

BODY_EXACT = {
    "PDA",
    "PDB",
    "DVA",
    "DVB",
    "DVC",
    "CANL",
    "CANR",
    "HSNL",
    "HSNR",
}

TAIL_PREFIXES = (
    "PHA",
    "PHB",
    "PHC",
    "PLM",
    "PLN",
    "PVN",
    "PVW",
    "PQR",
    "LUA",
)


@dataclass(frozen=True)
class NodeRecord:
    index: int
    node_type: str
    node_subtype: str
    name: str
    pos: tuple[float, float] | None = None


_POS_RE = re.compile(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", re.IGNORECASE)


def _parse_pos(raw: str) -> tuple[float, float] | None:
    matches = _POS_RE.findall(raw)
    if len(matches) >= 2:
        return (float(matches[0]), float(matches[1]))
    return None


def _iter_csv_rows(path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    with path.open(newline="", encoding="utf-8") as file_handle:
        for line in file_handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            rows.append([column.strip() for column in next(csv.reader([stripped]))])
    return rows


def _read_nodes(path: Path) -> dict[int, NodeRecord]:
    nodes: dict[int, NodeRecord] = {}
    for row in _iter_csv_rows(path):
        index, node_type, node_subtype, name, raw_pos = row
        node = NodeRecord(
            index=int(index),
            node_type=node_type,
            node_subtype=node_subtype,
            name=name,
            pos=_parse_pos(raw_pos),
        )
        nodes[node.index] = node
    return nodes


def _read_edges(path: Path) -> list[tuple[int, int, int]]:
    edges: list[tuple[int, int, int]] = []
    for row in _iter_csv_rows(path):
        source, target, synapses = row
        edges.append((int(source), int(target), int(synapses)))
    return edges


def _is_canonical_neuron(node: NodeRecord) -> bool:
    if node.node_type in {"SENSORY NEURONS", "INTERNEURONS"}:
        return True
    if node.node_type == "MOTOR NEURONS":
        return (
            node.node_subtype != "BODYWALL MUSCLES"
            and node.name not in NON_NEURON_MOTOR_CELLS
        )
    if node.node_type == "SEX-SPECIFIC CELLS":
        return node.node_subtype == "MOTOR NEURONS"
    if node.node_type == "PHARYNX":
        return node.name in PHARYNGEAL_NEURONS
    return False


def _neuron_type(node: NodeRecord) -> str:
    if node.node_type == "SENSORY NEURONS":
        return "S"
    if node.node_type == "INTERNEURONS":
        return "I"
    if node.node_type in {"MOTOR NEURONS", "SEX-SPECIFIC CELLS"}:
        return "M"
    if node.name.startswith("I") or node.name == "MI":
        return "I"
    if node.name.startswith("NSM"):
        return "S"
    return "M"


def _neuron_region(name: str) -> str:
    if name in BODY_EXACT or name.startswith("VC") or name.startswith(BODY_PREFIXES):
        return "body"
    if name.startswith(TAIL_PREFIXES):
        return "tail"
    return "head"


def _upsert_edge(
    graph: nx.DiGraph,
    source: str,
    target: str,
    chemical_weight: int = 0,
    gap_weight: int = 0,
) -> None:
    if not graph.has_edge(source, target):
        graph.add_edge(
            source,
            target,
            chemical_weight=0,
            gap_weight=0,
            weight=0,
        )

    edge_payload = graph[source][target]
    edge_payload["chemical_weight"] += chemical_weight
    edge_payload["gap_weight"] += gap_weight
    edge_payload["weight"] = edge_payload["chemical_weight"] + edge_payload["gap_weight"]


def build_connectome_graph() -> nx.DiGraph:
    chemical_nodes = _read_nodes(CHEMICAL_NODES_PATH)
    gap_nodes = _read_nodes(GAP_NODES_PATH)
    chemical_edges = _read_edges(CHEMICAL_EDGES_PATH)
    gap_edges = _read_edges(GAP_EDGES_PATH)

    canonical_nodes = {
        node.index: node
        for node in chemical_nodes.values()
        if _is_canonical_neuron(node)
    }
    if len(canonical_nodes) != CANONICAL_NODE_COUNT:
        raise ValueError(
            f"Expected {CANONICAL_NODE_COUNT} canonical neurons, got {len(canonical_nodes)}."
        )

    canonical_names = {node.name for node in canonical_nodes.values()}

    graph = nx.DiGraph(name="c_elegans_connectome")
    for node in canonical_nodes.values():
        pos_x, pos_y = node.pos if node.pos else (0.0, 0.0)
        graph.add_node(
            node.name,
            type=_neuron_type(node),
            name=node.name,
            region=_neuron_region(node.name),
            neurotransmitter="GABA" if node.name in GABAERGIC_NEURONS else "ACh",
            source_type=node.node_type,
            source_subtype=node.node_subtype,
            pos_x=pos_x,
            pos_y=pos_y,
        )

    for source_index, target_index, synapses in chemical_edges:
        source = chemical_nodes.get(source_index)
        target = chemical_nodes.get(target_index)
        if source is None or target is None:
            continue
        if source.name not in canonical_names or target.name not in canonical_names:
            continue
        _upsert_edge(
            graph,
            source=source.name,
            target=target.name,
            chemical_weight=synapses,
        )

    gap_pair_weights: dict[tuple[str, str], int] = {}
    for source_index, target_index, synapses in gap_edges:
        source = gap_nodes.get(source_index)
        target = gap_nodes.get(target_index)
        if source is None or target is None:
            continue
        if source.name not in canonical_names or target.name not in canonical_names:
            continue

        if source.name == target.name:
            _upsert_edge(
                graph,
                source=source.name,
                target=target.name,
                gap_weight=synapses,
            )
            continue

        pair = tuple(sorted((source.name, target.name)))
        gap_pair_weights[pair] = max(synapses, gap_pair_weights.get(pair, 0))

    for (left, right), synapses in gap_pair_weights.items():
        _upsert_edge(
            graph,
            source=left,
            target=right,
            gap_weight=synapses,
        )
        _upsert_edge(
            graph,
            source=right,
            target=left,
            gap_weight=synapses,
        )

    centrality_by_node = nx.degree_centrality(graph)
    nx.set_node_attributes(graph, centrality_by_node, "degree_centrality")
    return graph


@lru_cache(maxsize=1)
def _cached_connectome_graph() -> nx.DiGraph:
    return build_connectome_graph()


def get_connectome_graph() -> nx.DiGraph:
    return _cached_connectome_graph().copy()


def graph_to_data(graph: nx.DiGraph) -> dict[str, object]:
    nodes = []
    for node_id, attrs in graph.nodes(data=True):
        nodes.append({
            "id": node_id,
            "type": attrs["type"],
            "region": attrs["region"],
            "degree_centrality": round(attrs["degree_centrality"], 6),
            "pos_x": attrs["pos_x"],
            "pos_y": attrs["pos_y"],
            "in_degree": graph.in_degree(node_id),
            "out_degree": graph.out_degree(node_id),
        })
    edges = []
    for src, tgt, attrs in graph.edges(data=True):
        edges.append({
            "source": src,
            "target": tgt,
            "chemical_weight": attrs["chemical_weight"],
            "gap_weight": attrs["gap_weight"],
            "weight": attrs["weight"],
        })
    return {"nodes": nodes, "edges": edges}


def get_connectome_graph_data() -> dict[str, object]:
    return graph_to_data(_cached_connectome_graph())


def get_connectome_summary() -> dict[str, object]:
    graph = _cached_connectome_graph()
    type_counts = Counter(nx.get_node_attributes(graph, "type").values())
    region_counts = Counter(nx.get_node_attributes(graph, "region").values())

    return {
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "chemical_edge_count": sum(
            1 for _, _, edge_data in graph.edges(data=True) if edge_data["chemical_weight"] > 0
        ),
        "gap_edge_count": sum(
            1 for _, _, edge_data in graph.edges(data=True) if edge_data["gap_weight"] > 0
        ),
        "type_counts": dict(type_counts),
        "region_counts": dict(region_counts),
        "top_degree_centrality": sorted(
            (
                {
                    "name": node,
                    "degree_centrality": graph.nodes[node]["degree_centrality"],
                }
                for node in graph.nodes
            ),
            key=lambda item: item["degree_centrality"],
            reverse=True,
        )[:10],
    }
