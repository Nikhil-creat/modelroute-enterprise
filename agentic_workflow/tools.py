"""Agent tools: knowledge-base RAG search, CNN visual diagnostic, SQL account/SLA lookup."""
from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Any

import config


@dataclass
class ToolResult:
    name: str
    arguments: dict[str, Any]
    output: dict[str, Any] | None
    ok: bool
    latency_ms: float
    error: str | None = None

    def summary(self) -> dict[str, Any]:
        return {"tool": self.name, "ok": self.ok, "latency_ms": round(self.latency_ms, 2)}


class Tool:
    name = "tool"
    description = ""

    def run(self, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError

    def __call__(self, **kwargs: Any) -> ToolResult:
        start = time.perf_counter()
        try:
            output = self.run(**kwargs)
            return ToolResult(self.name, kwargs, output, True, (time.perf_counter() - start) * 1000)
        except Exception as exc:  # tools must never crash the agent loop
            return ToolResult(self.name, kwargs, None, False, (time.perf_counter() - start) * 1000, str(exc))


class RAGKnowledgeSearch(Tool):
    name = "RAGKnowledgeSearch"
    description = "Hybrid (dense + BM25) search over internal knowledge-base articles."

    def __init__(self, retriever: Any) -> None:
        self.retriever = retriever

    def run(self, query: str, k: int = config.RAG_TOP_K) -> dict[str, Any]:
        result = self.retriever.search(query, k=k)
        return {
            "hits": [{"kb_id": h.kb_id, "title": h.title, "score": round(h.score, 4)} for h in result.hits],
            "kb_ids": result.kb_ids(),
            "relevance": result.relevance,
            "context": result.context(),
        }


class CNNVisualDiagnostic(Tool):
    name = "CNNVisualDiagnostic"
    description = "Classifies a ticket attachment as UI_ERROR, HARDWARE_FAULT, SPAM_IMAGE or VALID_DOCUMENT."

    def __init__(self, classifier: Any) -> None:
        self.classifier = classifier

    def run(self, image_path: str) -> dict[str, Any]:
        result = self.classifier.classify(image_path)
        return {"label": result.label, "confidence": result.confidence, "probs": result.probs}


class AccountStatusLookup(Tool):
    """Mock SQL tool (in-memory SQLite) returning tier details and the SLA agreement."""

    name = "AccountStatusLookup"
    description = "Looks up the customer tier, support channel and SLA response hours."

    _TIER_ROWS = {
        "enterprise": ("24x7 phone + priority queue", 1, 99.95),
        "pro": ("business-hours chat + email", 0, 99.9),
        "free": ("community email", 0, 99.0),
    }

    def __init__(self) -> None:
        self._db = sqlite3.connect(":memory:", check_same_thread=False)
        self._lock = threading.Lock()
        self._db.executescript(
            "CREATE TABLE tiers(tier TEXT PRIMARY KEY, support_channel TEXT, dedicated_csm INTEGER, uptime_sla REAL);"
            "CREATE TABLE sla_agreements(tier TEXT, priority TEXT, response_hours INTEGER, PRIMARY KEY(tier, priority));")
        self._db.executemany("INSERT INTO tiers VALUES (?, ?, ?, ?)",
                             [(t, *row) for t, row in self._TIER_ROWS.items()])
        self._db.executemany("INSERT INTO sla_agreements VALUES (?, ?, ?)",
                             [(t, p, h) for t, pr in config.SLA_HOURS.items() for p, h in pr.items()])

    def run(self, customer_tier: str, priority: str | None = None) -> dict[str, Any]:
        with self._lock:
            tier = self._db.execute("SELECT support_channel, dedicated_csm, uptime_sla FROM tiers WHERE tier = ?",
                                    (customer_tier,)).fetchone()
            if tier is None:
                return {"found": False, "tier": customer_tier}
            rows = self._db.execute("SELECT priority, response_hours FROM sla_agreements WHERE tier = ? "
                                    "ORDER BY priority", (customer_tier,)).fetchall()
        sla = {p: h for p, h in rows}
        return {"found": True, "tier": customer_tier, "support_channel": tier[0], "dedicated_csm": bool(tier[1]),
                "uptime_sla": tier[2], "sla_hours": sla,
                "response_hours": sla.get(priority) if priority else None}


def build_toolbox(retriever: Any | None = None, classifier: Any | None = None) -> dict[str, Tool]:
    tools: dict[str, Tool] = {}
    if retriever is not None:
        tools[RAGKnowledgeSearch.name] = RAGKnowledgeSearch(retriever)
    if classifier is not None:
        tools[CNNVisualDiagnostic.name] = CNNVisualDiagnostic(classifier)
    tools[AccountStatusLookup.name] = AccountStatusLookup()
    return tools


def tool_schemas() -> list[dict[str, Any]]:
    """OpenAI-style function schemas (usable for LLM-driven tool calling)."""
    return [
        {"type": "function", "function": {
            "name": RAGKnowledgeSearch.name, "description": RAGKnowledgeSearch.description,
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
        {"type": "function", "function": {
            "name": CNNVisualDiagnostic.name, "description": CNNVisualDiagnostic.description,
            "parameters": {"type": "object", "properties": {"image_path": {"type": "string"}},
                           "required": ["image_path"]}}},
        {"type": "function", "function": {
            "name": AccountStatusLookup.name, "description": AccountStatusLookup.description,
            "parameters": {"type": "object", "properties": {
                "customer_tier": {"type": "string", "enum": config.CUSTOMER_TIERS},
                "priority": {"type": "string", "enum": config.PRIORITIES}}, "required": ["customer_tier"]}}},
    ]
