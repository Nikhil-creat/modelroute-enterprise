"""Offline smoke test with fakes: no GPU, models, vector DB or network needed.

Run:  python -m tests.smoke_test
"""
from __future__ import annotations

import json
import math

import config
from agentic_workflow.agent_router import AgentResult, LLMResult, TriageAgent, parse_decision
from agentic_workflow.tools import AccountStatusLookup, build_toolbox
from evaluation.benchmark import exact_match, macro_f1
from rag_engine.retriever import RetrievalResult, RetrievedChunk
from router.confidence_engine import compute_confidence, text_confidence
from router.hybrid_router import FrontierLLM, HybridRouter, frontier_cost, local_cost


class FakeRetriever:
    def __init__(self, relevance: float) -> None:
        self.relevance = relevance

    def search(self, query: str, k: int = 4) -> RetrievalResult:
        hit = RetrievedChunk("KB-0005::c0", "KB-0005", "Device boot loop", "text", 0.7, 0.9, self.relevance)
        return RetrievalResult(query, [hit])


class FakeVision:
    class R:
        label, confidence, probs = "HARDWARE_FAULT", 0.97, {"HARDWARE_FAULT": 0.97}

    def classify(self, path: str):
        return self.R()


class FakeLLM:
    def __init__(self, text: str, lp: float) -> None:
        self.text, self.lp = text, lp

    def generate(self, messages):
        return LLMResult(self.text, [self.lp] * 20, 300, 20, 5.0)


class FakeFrontier(FrontierLLM):
    def __init__(self) -> None:
        self.provider, self.model = "openai", "gpt-4o"

    def generate(self, messages):
        text = json.dumps({"intent": "billing_issue", "priority": "P2", "target_department": "Billing & Finance"})
        return LLMResult(text, [], 400, 25, 800.0)


GOOD = json.dumps({"intent": "hardware_failure", "priority": "P1", "target_department": "Hardware Support"})


def build_router(text: str, logprob: float, relevance: float, threshold: float = 0.85) -> HybridRouter:
    tools = build_toolbox(FakeRetriever(relevance), FakeVision())
    return HybridRouter(TriageAgent(FakeLLM(text, logprob), tools), FakeFrontier(), threshold)


def main() -> None:
    assert parse_decision("noise " + GOOD + " tail")["priority"] == "P1"
    assert parse_decision('{"intent": "nope"}') is None
    assert math.isclose(text_confidence([math.log(0.9)] * 5), 0.9, rel_tol=1e-6)

    c = compute_confidence([math.log(0.9)] * 5, None, 0.8)
    assert set(c.weights) == {"text", "rag"} and abs(sum(c.weights.values()) - 1) < 1e-9
    assert compute_confidence([0.0], 1.0, 1.0, valid=False).c_total == 0.0

    sla = AccountStatusLookup().run(customer_tier="enterprise", priority="P1")
    assert sla["response_hours"] == 1 and AccountStatusLookup().run(customer_tier="x")["found"] is False

    confident = build_router(GOOD, math.log(0.99), 0.9).route("rack controller kernel panic", "enterprise", "img.png")
    assert confident.route == "local" and confident.decision["intent"] == "hardware_failure"
    assert [t["tool"] for t in confident.agent.tool_calls_made] == [
        "CNNVisualDiagnostic", "RAGKnowledgeSearch", "AccountStatusLookup"]
    assert confident.agent.sla["response_hours"] == 1

    unsure = build_router(GOOD, math.log(0.4), 0.2).route("rack controller kernel panic", "enterprise", "img.png")
    assert unsure.route == "frontier" and unsure.decision["intent"] == "billing_issue"
    assert unsure.telemetry["cost_usd"] > confident.telemetry["cost_usd"]

    broken = build_router("not json", math.log(0.99), 0.9).route("x ticket", "free")
    assert broken.confidence.c_total == 0.0 and broken.route == "frontier"

    reflect = build_router(GOOD, math.log(0.99), 0.1).route("help", "pro", "img.png")
    assert sum(t["tool"] == "RAGKnowledgeSearch" for t in reflect.agent.tool_calls_made) == 2

    direct = build_router(GOOD, math.log(0.99), 0.9).route("t", "pro", None, mode="direct")
    assert [t["tool"] for t in direct.agent.tool_calls_made] == []

    row = {"intent": "hardware_failure", "priority": "P1", "target_department": "Hardware Support"}
    assert exact_match(json.loads(GOOD), row) and not exact_match(None, row)
    assert macro_f1(["a", "b"], ["a", "b"], ["a", "b"]) == 1.0
    assert local_cost(300, 20) > 0 and frontier_cost("gpt-4o", 1000, 100) > 0
    assert isinstance(AgentResult.__name__, str) and config.load_calibrated_threshold() > 0
    print("smoke test passed")


if __name__ == "__main__":
    main()
