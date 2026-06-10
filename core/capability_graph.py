from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from .config import ROOT_DIR
from .knowledge_metadata import normalize_metadata
from .tool_registry import ToolRegistry
from .skill_registry import SkillRegistry
from .user_knowledge_base import UserKnowledgeBaseService, AREAS

CAPABILITY_GRAPH_DIR = ROOT_DIR / "data" / "capability_graph"
GRAPH_FILE = CAPABILITY_GRAPH_DIR / "graph.json"
NODES_FILE = CAPABILITY_GRAPH_DIR / "nodes.json"
EDGES_FILE = CAPABILITY_GRAPH_DIR / "edges.json"

STOPWORDS = {
    "der", "die", "das", "und", "oder", "mit", "für", "eine", "einer", "ein", "the", "and", "with", "from", "this", "that",
    "tool", "skill", "pandora", "agent", "system", "datei", "daten", "info", "wissen", "notiz", "notes", "markdown",
}


@dataclass
class CapabilityNode:
    id: str
    label: str
    type: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "label": self.label, "type": self.type, "source": self.source, "metadata": self.metadata}


@dataclass
class CapabilityEdge:
    source: str
    target: str
    relation: str
    weight: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def key(self) -> tuple[str, str, str]:
        return self.source, self.target, self.relation

    def as_dict(self) -> dict[str, Any]:
        return {"source": self.source, "target": self.target, "relation": self.relation, "weight": self.weight, "metadata": self.metadata}


def slugify(value: str) -> str:
    text = (value or "").strip().lower()
    text = text.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or "unknown"


def capability_id(label: str) -> str:
    return f"cap:{slugify(label)}"


class CapabilityGraphService:
    """Build and persist a first capability graph from tools, skills and user knowledge.

    The first MVP deliberately stays deterministic and transparent: capabilities are
    derived from explicit tags, titles and registry metadata. It does not call an
    LLM and it does not mutate tools, skills or knowledge documents.
    """

    def __init__(self, graph_dir: Path = CAPABILITY_GRAPH_DIR):
        self.graph_dir = graph_dir
        self.graph_file = graph_dir / "graph.json"
        self.nodes_file = graph_dir / "nodes.json"
        self.edges_file = graph_dir / "edges.json"

    def status(self) -> dict[str, Any]:
        graph = self.load_graph()
        return {
            "kind": "capability_graph_status",
            "graph_dir": str(self.graph_dir),
            "exists": self.graph_file.exists(),
            "node_count": len(graph.get("nodes", [])),
            "edge_count": len(graph.get("edges", [])),
            "capability_count": sum(1 for node in graph.get("nodes", []) if node.get("type") == "capability"),
            "updated_at": graph.get("updated_at"),
            "read_only_sources": True,
        }

    def rebuild(self, *, write: bool = True) -> dict[str, Any]:
        nodes: dict[str, CapabilityNode] = {}
        edges: dict[tuple[str, str, str], CapabilityEdge] = {}

        def add_node(node: CapabilityNode) -> None:
            if node.id not in nodes:
                nodes[node.id] = node
            else:
                nodes[node.id].metadata.update({k: v for k, v in node.metadata.items() if v not in (None, "", [], {})})

        def add_edge(edge: CapabilityEdge) -> None:
            key = edge.key()
            if key in edges:
                edges[key].weight += edge.weight
                edges[key].metadata.update(edge.metadata)
            else:
                edges[key] = edge

        self._collect_tools(add_node, add_edge)
        self._collect_skills(add_node, add_edge)
        self._collect_knowledge(add_node, add_edge)
        self._collect_capability_gaps(add_node, add_edge)

        graph = {
            "kind": "capability_graph",
            "version": "mvp-23.0-capability-graph-foundation",
            "updated_at": datetime.now(UTC).isoformat(),
            "nodes": [node.as_dict() for node in sorted(nodes.values(), key=lambda item: (item.type, item.label.lower()))],
            "edges": [edge.as_dict() for edge in sorted(edges.values(), key=lambda item: (item.source, item.relation, item.target))],
            "summary": self._summary(nodes, edges),
        }
        if write:
            self._write(graph)
        return graph

    def load_graph(self) -> dict[str, Any]:
        if not self.graph_file.exists():
            return {"kind": "capability_graph", "nodes": [], "edges": [], "summary": {}}
        try:
            return json.loads(self.graph_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"kind": "capability_graph", "nodes": [], "edges": [], "summary": {}, "error": "graph file is invalid JSON"}

    def list_capabilities(self, *, query: str | None = None, limit: int = 200) -> dict[str, Any]:
        graph = self.load_graph()
        needle = (query or "").strip().lower()
        capabilities = [node for node in graph.get("nodes", []) if node.get("type") == "capability"]
        if needle:
            capabilities = [node for node in capabilities if needle in node.get("label", "").lower() or needle in node.get("id", "").lower()]
        capabilities.sort(key=lambda node: (node.get("metadata", {}).get("degree", 0), node.get("label", "")), reverse=True)
        return {"kind": "capability_list", "query": query or "", "count": min(len(capabilities), limit), "capabilities": capabilities[:limit]}

    def show_capability(self, capability: str) -> dict[str, Any]:
        graph = self.load_graph()
        cap_id = capability if capability.startswith("cap:") else capability_id(capability)
        nodes = {node.get("id"): node for node in graph.get("nodes", [])}
        node = nodes.get(cap_id)
        if not node:
            return {"found": False, "id": cap_id, "error": "capability not found"}
        connected_edges = [edge for edge in graph.get("edges", []) if edge.get("source") == cap_id or edge.get("target") == cap_id]
        related = []
        for edge in connected_edges:
            other_id = edge.get("target") if edge.get("source") == cap_id else edge.get("source")
            if other_id in nodes:
                related.append({"relation": edge.get("relation"), "direction": "out" if edge.get("source") == cap_id else "in", "edge": edge, "node": nodes[other_id]})
        return {"found": True, "capability": node, "related_count": len(related), "related": related}

    def _collect_tools(self, add_node, add_edge) -> None:
        registry = ToolRegistry()
        registry.discover()
        for tool in registry.list():
            tool_id = f"tool:{tool.id}"
            add_node(CapabilityNode(tool_id, tool.name or tool.id, "tool", "tool_registry", {
                "id": tool.id,
                "description": getattr(tool, "description", ""),
                "status": str(getattr(tool, "status", "")),
                "version": getattr(tool, "version", ""),
            }))
            candidates = self._keywords([tool.id, tool.name, getattr(tool, "description", ""), " ".join(getattr(tool, "aliases", []) or [])])
            for label in candidates[:8]:
                cap = capability_id(label)
                add_node(CapabilityNode(cap, label, "capability", "tool_registry"))
                add_edge(CapabilityEdge(cap, tool_id, "has_tool", weight=2))

    def _collect_skills(self, add_node, add_edge) -> None:
        registry = SkillRegistry()
        registry.discover()
        for skill in registry.list():
            skill_id = f"skill:{skill.id}"
            add_node(CapabilityNode(skill_id, getattr(skill, "name", skill.id) or skill.id, "skill", "skill_registry", {
                "id": skill.id,
                "description": getattr(skill, "description", ""),
                "version": getattr(skill, "version", ""),
            }))
            candidates = self._keywords([skill.id, getattr(skill, "name", ""), getattr(skill, "description", "")])
            for label in candidates[:8]:
                cap = capability_id(label)
                add_node(CapabilityNode(cap, label, "capability", "skill_registry"))
                add_edge(CapabilityEdge(cap, skill_id, "has_skill", weight=3))

    def _collect_knowledge(self, add_node, add_edge) -> None:
        kb = UserKnowledgeBaseService()
        kb.ensure_structure()
        for area in AREAS:
            area_root = kb.root_dir / area.directory
            for file in kb._scan_files(area_root, limit=2000):
                rel = file["relative_path"]
                node_id = f"knowledge:{area.name}:{rel}"
                metadata = file.get("metadata", {}) or {}
                title = metadata.get("title") or file.get("title") or Path(rel).stem
                add_node(CapabilityNode(node_id, title, "knowledge", "user_knowledge", {
                    "area": area.name,
                    "relative_path": rel,
                    "cloud_allowed": bool(metadata.get("cloud_allowed", area.cloud_allowed)),
                    "tags": metadata.get("tags", []),
                    "priority": metadata.get("priority", "normal"),
                }))
                labels = list(metadata.get("tags", []) or [])
                labels += self._path_labels(rel)
                labels += self._keywords([title, metadata.get("summary", "")])[:5]
                for label in self._dedupe_labels(labels)[:12]:
                    cap = capability_id(label)
                    add_node(CapabilityNode(cap, label, "capability", "user_knowledge"))
                    add_edge(CapabilityEdge(cap, node_id, "has_knowledge", weight=2 if label in (metadata.get("tags") or []) else 1))

    def _collect_capability_gaps(self, add_node, add_edge) -> None:
        gaps_dir = ROOT_DIR / "proposals" / "capability_gaps"
        if not gaps_dir.exists():
            return
        for path in sorted(gaps_dir.glob("*.json"))[:1000]:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            gap_label = str(data.get("capability") or data.get("name") or path.stem)
            gap_id = f"gap:{slugify(path.stem)}"
            add_node(CapabilityNode(gap_id, gap_label, "gap", "capability_gaps", {"path": str(path.relative_to(ROOT_DIR))}))
            cap = capability_id(gap_label)
            add_node(CapabilityNode(cap, gap_label, "capability", "capability_gaps"))
            add_edge(CapabilityEdge(cap, gap_id, "has_gap", weight=4))

    def _write(self, graph: dict[str, Any]) -> None:
        self.graph_dir.mkdir(parents=True, exist_ok=True)
        self.graph_file.write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")
        self.nodes_file.write_text(json.dumps(graph["nodes"], indent=2, ensure_ascii=False), encoding="utf-8")
        self.edges_file.write_text(json.dumps(graph["edges"], indent=2, ensure_ascii=False), encoding="utf-8")

    def _summary(self, nodes: dict[str, CapabilityNode], edges: dict[tuple[str, str, str], CapabilityEdge]) -> dict[str, Any]:
        degree: dict[str, int] = {}
        by_type: dict[str, int] = {}
        by_relation: dict[str, int] = {}
        for node in nodes.values():
            by_type[node.type] = by_type.get(node.type, 0) + 1
        for edge in edges.values():
            degree[edge.source] = degree.get(edge.source, 0) + edge.weight
            degree[edge.target] = degree.get(edge.target, 0) + edge.weight
            by_relation[edge.relation] = by_relation.get(edge.relation, 0) + 1
        for node_id, value in degree.items():
            if node_id in nodes:
                nodes[node_id].metadata["degree"] = value
        top = sorted((nodes[node_id] for node_id in degree if node_id in nodes and nodes[node_id].type == "capability"), key=lambda n: n.metadata.get("degree", 0), reverse=True)[:10]
        return {"node_count": len(nodes), "edge_count": len(edges), "by_type": by_type, "by_relation": by_relation, "top_capabilities": [node.as_dict() for node in top]}

    def _keywords(self, values: list[str]) -> list[str]:
        text = " ".join(str(value or "") for value in values).lower()
        raw = re.findall(r"[a-zA-ZäöüÄÖÜß0-9][a-zA-ZäöüÄÖÜß0-9_\-]{2,}", text)
        cleaned = []
        for item in raw:
            label = item.strip("_- ").lower()
            if len(label) < 3 or label in STOPWORDS or label.isdigit():
                continue
            cleaned.append(label.replace("_", " ").replace("-", " "))
        return self._dedupe_labels(cleaned)

    def _path_labels(self, relative_path: str) -> list[str]:
        parts = Path(relative_path).parts[:-1]
        return [part.replace("_", " ").replace("-", " ") for part in parts if part and part not in {".", ".."}]

    def _dedupe_labels(self, labels: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for label in labels:
            clean = str(label or "").strip().lower()
            if not clean or clean in STOPWORDS:
                continue
            if clean not in seen:
                seen.add(clean)
                result.append(clean)
        return result
