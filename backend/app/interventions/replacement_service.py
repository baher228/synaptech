from __future__ import annotations

from dataclasses import dataclass, field
import math
import random as stdlib_random
from typing import Literal

import networkx as nx

from app.connectome import get_connectome_graph
from app.interventions.fault_detection import FaultDetectionService
from app.interventions.ou_process import OUReplacementManager
from app.simulation.protocol import SimulationEngine


@dataclass
class EdgeMigration:
    migration_id: str
    old_source: str
    old_target: str
    new_source: str
    new_target: str
    chemical_weight: float
    gap_weight: float

    def to_dict(self) -> dict[str, object]:
        return {
            "migration_id": self.migration_id,
            "old_source": self.old_source,
            "old_target": self.old_target,
            "new_source": self.new_source,
            "new_target": self.new_target,
            "chemical_weight": self.chemical_weight,
            "gap_weight": self.gap_weight,
        }


@dataclass
class ReplacementSession:
    session_id: str
    faulty_neuron: str
    replacement_neuron: str
    pending: list[EdgeMigration] = field(default_factory=list)
    completed: list[EdgeMigration] = field(default_factory=list)
    status: str = "in_progress"
    mode: Literal["instant", "ou"] = "instant"
    ou_manager: OUReplacementManager | None = None
    ou_params: dict | None = None

    def to_dict(self) -> dict[str, object]:
        next_edge = self.pending[0].to_dict() if self.pending else None
        d: dict[str, object] = {
            "session_id": self.session_id,
            "faulty_neuron": self.faulty_neuron,
            "replacement_neuron": self.replacement_neuron,
            "status": self.status,
            "pending_count": len(self.pending),
            "completed_count": len(self.completed),
            "next_edge": next_edge,
            "completed_edges": [m.to_dict() for m in self.completed],
            "mode": self.mode,
        }
        if self.ou_params:
            d["ou_params"] = self.ou_params
        if self.ou_manager is not None:
            d["ou_convergence"] = self.ou_manager.convergence_fraction()
        return d


class ReplacementService:
    """Stateful replacement workflow: deploy replacement and migrate edges one-by-one."""

    def __init__(
        self,
        graph: nx.DiGraph | None = None,
        engine: SimulationEngine | None = None,
    ) -> None:
        self.graph = graph.copy() if graph is not None else get_connectome_graph()
        self.engine = engine
        self.sessions: dict[str, ReplacementSession] = {}
        self._session_counter = 0
        self._replacement_counter = 0

    def reset(self) -> None:
        self.graph = get_connectome_graph()
        self.sessions.clear()
        self._session_counter = 0
        self._replacement_counter = 0

    def random_faulty_neurons(self, count: int, seed: int | None = None) -> list[str]:
        detector = FaultDetectionService()
        return detector.detect_faulty_neurons(
            graph=self.graph,
            count=count,
            seed=seed,
        )

    def start_replacement(
        self,
        faulty_neuron: str,
        edge_order: str = "random",
        seed: int | None = None,
        mode: Literal["instant", "ou"] = "instant",
        theta: float | None = None,
        sigma: float | None = None,
        integration: str = "mirror",
        integration_params: dict | None = None,
    ) -> ReplacementSession:
        if faulty_neuron not in self.graph:
            raise ValueError(f"Neuron '{faulty_neuron}' does not exist in graph.")

        attrs = self.graph.nodes[faulty_neuron]
        if attrs.get("is_replacement", False):
            raise ValueError("Cannot replace a replacement neuron.")
        if attrs.get("is_ghosted", False):
            raise ValueError("Neuron is already ghosted.")

        replacement_neuron = self._new_replacement_name(faulty_neuron)
        replacement_attrs = dict(attrs)
        replacement_attrs["is_replacement"] = True
        replacement_attrs["replacement_for"] = faulty_neuron
        replacement_attrs["is_ghosted"] = False
        replacement_attrs["name"] = replacement_neuron
        replacement_index = max(0, self._replacement_counter - 1)
        replacement_pos_x, replacement_pos_y = self._offset_replacement_position(
            base_x=float(attrs.get("pos_x", 0.0)),
            base_y=float(attrs.get("pos_y", 0.0)),
            replacement_index=replacement_index,
        )
        replacement_attrs["pos_x"] = replacement_pos_x
        replacement_attrs["pos_y"] = replacement_pos_y
        self.graph.add_node(replacement_neuron, **replacement_attrs)

        if self.engine is not None:
            self.engine.add_neuron_from_slot(
                replacement_name=replacement_neuron,
                copy_params_from=faulty_neuron,
            )

        migrations = self._build_edge_migrations(
            faulty_neuron=faulty_neuron,
            replacement_neuron=replacement_neuron,
        )

        # Apply integration strategy (transforms edge topology/weights)
        from app.interventions.integration_strategy import get_integration_strategy

        int_strategy = get_integration_strategy(
            integration, **(integration_params or {})
        )
        int_rng = stdlib_random.Random(seed)
        migrations = int_strategy.transform(
            graph=self.graph,
            faulty=faulty_neuron,
            replacement=replacement_neuron,
            migrations=migrations,
            rng=int_rng,
        )

        if edge_order == "random":
            rng = stdlib_random.Random(seed)
            rng.shuffle(migrations)
        elif edge_order != "deterministic":
            raise ValueError("edge_order must be 'random' or 'deterministic'.")

        session_id = f"session_{self._session_counter:04d}"
        self._session_counter += 1

        if mode == "ou":
            from app.interventions.ou_process import DEFAULT_THETA, DEFAULT_SIGMA

            ou_theta = theta if theta is not None else DEFAULT_THETA
            ou_sigma = sigma if sigma is not None else DEFAULT_SIGMA
            ou_mgr = OUReplacementManager(
                theta=ou_theta,
                sigma=ou_sigma,
                rng_seed=seed,
            )
            ou_mgr.add_edges(migrations)

            # Create new edges in graph at zero weight; keep old edges alive
            # (OU manager handles the gradual crossover in the engine)
            for m in migrations:
                self.graph.add_edge(
                    m.new_source,
                    m.new_target,
                    chemical_weight=0.0,
                    gap_weight=0.0,
                    weight=0.0,
                )

            session = ReplacementSession(
                session_id=session_id,
                faulty_neuron=faulty_neuron,
                replacement_neuron=replacement_neuron,
                pending=[],
                completed=migrations,
                mode="ou",
                ou_manager=ou_mgr,
                ou_params={"theta": ou_theta, "sigma": ou_sigma},
            )
        else:
            session = ReplacementSession(
                session_id=session_id,
                faulty_neuron=faulty_neuron,
                replacement_neuron=replacement_neuron,
                pending=migrations,
                mode="instant",
            )

        self.sessions[session_id] = session
        return session

    def step_replacement(self, session_id: str, edges_to_migrate: int = 1) -> ReplacementSession:
        if edges_to_migrate < 1:
            raise ValueError("edges_to_migrate must be >= 1")
        session = self._get_session(session_id)
        if session.status == "completed":
            return session

        for _ in range(min(edges_to_migrate, len(session.pending))):
            migration = session.pending.pop(0)
            self._apply_edge_migration(migration)
            session.completed.append(migration)

        if not session.pending:
            self._ghost_faulty_neuron(
                faulty_neuron=session.faulty_neuron,
                replacement_neuron=session.replacement_neuron,
            )
            session.status = "completed"

        return session

    def tick_ou(self, session_id: str) -> ReplacementSession:
        """Advance OU process one step. Ghosts the old neuron when converged."""
        session = self._get_session(session_id)
        if session.mode != "ou":
            raise ValueError("Session is not in OU mode.")
        if session.status == "completed":
            return session
        if session.ou_manager is None or self.engine is None:
            raise RuntimeError("OU manager or engine not available.")

        all_done = session.ou_manager.tick(self.engine)
        if all_done:
            # Snap new edge weights to exact target in graph
            for m in session.completed:
                self.graph.add_edge(
                    m.new_source,
                    m.new_target,
                    chemical_weight=m.chemical_weight,
                    gap_weight=m.gap_weight,
                    weight=m.chemical_weight + m.gap_weight,
                )
            # Remove old edges from graph
            for m in session.completed:
                if self.graph.has_edge(m.old_source, m.old_target):
                    self.graph.remove_edge(m.old_source, m.old_target)
            self._ghost_faulty_neuron(
                faulty_neuron=session.faulty_neuron,
                replacement_neuron=session.replacement_neuron,
            )
            session.status = "completed"
        return session

    def get_session(self, session_id: str) -> ReplacementSession:
        return self._get_session(session_id)

    def _get_session(self, session_id: str) -> ReplacementSession:
        if session_id not in self.sessions:
            raise ValueError(f"Unknown session '{session_id}'.")
        return self.sessions[session_id]

    def _new_replacement_name(self, faulty_neuron: str) -> str:
        name = f"{faulty_neuron}__rep_{self._replacement_counter:03d}"
        self._replacement_counter += 1
        return name

    def _offset_replacement_position(
        self,
        base_x: float,
        base_y: float,
        replacement_index: int,
    ) -> tuple[float, float]:
        # Golden-angle fan-out keeps multiple replacements visually separated.
        angle = replacement_index * 2.399963229728653
        radius = 2.4 + 0.5 * (replacement_index % 3)
        return (
            base_x + radius * math.cos(angle),
            base_y + radius * math.sin(angle),
        )

    def _build_edge_migrations(
        self,
        faulty_neuron: str,
        replacement_neuron: str,
    ) -> list[EdgeMigration]:
        unique_edges: dict[tuple[str, str], dict] = {}

        for src, tgt, payload in self.graph.in_edges(faulty_neuron, data=True):
            unique_edges[(src, tgt)] = dict(payload)
        for src, tgt, payload in self.graph.out_edges(faulty_neuron, data=True):
            unique_edges[(src, tgt)] = dict(payload)

        migrations: list[EdgeMigration] = []
        for edge_idx, ((old_src, old_tgt), payload) in enumerate(unique_edges.items()):
            chemical_weight = float(payload.get("chemical_weight", 0.0))
            gap_weight = float(payload.get("gap_weight", 0.0))
            if chemical_weight <= 0.0 and gap_weight <= 0.0:
                continue

            new_src = replacement_neuron if old_src == faulty_neuron else old_src
            new_tgt = replacement_neuron if old_tgt == faulty_neuron else old_tgt

            migrations.append(
                EdgeMigration(
                    migration_id=f"edge_{edge_idx:04d}",
                    old_source=old_src,
                    old_target=old_tgt,
                    new_source=new_src,
                    new_target=new_tgt,
                    chemical_weight=chemical_weight,
                    gap_weight=gap_weight,
                )
            )
        return migrations

    def _apply_edge_migration(self, migration: EdgeMigration) -> None:
        if self.graph.has_edge(migration.old_source, migration.old_target):
            self.graph.remove_edge(migration.old_source, migration.old_target)

        new_chemical = migration.chemical_weight
        new_gap = migration.gap_weight

        if self.graph.has_edge(migration.new_source, migration.new_target):
            existing = self.graph[migration.new_source][migration.new_target]
            new_chemical += float(existing.get("chemical_weight", 0.0))
            new_gap += float(existing.get("gap_weight", 0.0))

        self.graph.add_edge(
            migration.new_source,
            migration.new_target,
            chemical_weight=new_chemical,
            gap_weight=new_gap,
            weight=new_chemical + new_gap,
        )

        # Synchronise running engine weights
        if self.engine is not None:
            # Zero out old edge
            if migration.chemical_weight > 0:
                self.engine.set_weights(
                    [(migration.old_source, migration.old_target)], [0.0],
                )
            if migration.gap_weight > 0:
                self.engine.set_gap_weights(
                    [(migration.old_source, migration.old_target)], [0.0],
                )
            # Set accumulated new edge weights
            if new_chemical > 0:
                self.engine.set_weights(
                    [(migration.new_source, migration.new_target)], [new_chemical],
                )
            if new_gap > 0:
                self.engine.set_gap_weights(
                    [(migration.new_source, migration.new_target)], [new_gap],
                )

    def _ghost_faulty_neuron(self, faulty_neuron: str, replacement_neuron: str) -> None:
        if faulty_neuron not in self.graph:
            return
        if replacement_neuron in self.graph:
            # Once migration is complete, snap the replacement neuron onto the
            # original neuron's position so it fully takes over in the UI.
            self.graph.nodes[replacement_neuron]["pos_x"] = float(
                self.graph.nodes[faulty_neuron].get("pos_x", 0.0)
            )
            self.graph.nodes[replacement_neuron]["pos_y"] = float(
                self.graph.nodes[faulty_neuron].get("pos_y", 0.0)
            )
        self.graph.nodes[faulty_neuron]["is_ghosted"] = True
        self.graph.nodes[faulty_neuron]["is_active"] = False
        self.graph.nodes[faulty_neuron]["replaced_by"] = replacement_neuron

        in_edges = list(self.graph.in_edges(faulty_neuron))
        out_edges = list(self.graph.out_edges(faulty_neuron))
        self.graph.remove_edges_from(in_edges + out_edges)

        if self.engine is not None:
            self.engine.silence_neurons([faulty_neuron])
