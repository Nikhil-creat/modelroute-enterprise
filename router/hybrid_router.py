"""Hybrid router: local agent first, frontier model when C_total < threshold."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import config
from agentic_workflow.agent_router import AgentResult, LLMResult, TriageAgent, parse_decision
from router.confidence_engine import ConfidenceBreakdown, compute_confidence

log = logging.getLogger("modelroute.router")


# --------------------------------------------------------------------------- #
# Cost model
# --------------------------------------------------------------------------- #
def local_cost(tokens_in: int, tokens_out: int) -> float:
    """GPU-seconds * hourly price. Prefill is assumed 20x faster than decoding."""
    seconds = tokens_out / config.LOCAL_TOKENS_PER_SECOND + tokens_in / (config.LOCAL_TOKENS_PER_SECOND * 20)
    return seconds * config.LOCAL_GPU_COST_PER_HOUR / 3600.0


def frontier_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    price = config.FRONTIER_PRICING.get(model, {"input": 0.0, "output": 0.0})
    return (tokens_in * price["input"] + tokens_out * price["output"]) / 1_000_000.0


# --------------------------------------------------------------------------- #
# Frontier client
# --------------------------------------------------------------------------- #
class FrontierLLM:
    """GPT-4o, Claude 3.5 Sonnet, or Groq (free tier) behind one interface."""

    def __init__(self, provider: str | None = None, model: str | None = None) -> None:
        self.provider = provider or config.FRONTIER_PROVIDER
        self.model = model or config.FRONTIER_MODELS[self.provider]
        self._client = None

    @classmethod
    def try_create(cls, provider: str | None = None) -> "FrontierLLM | None":
        provider = provider or config.FRONTIER_PROVIDER
        key = {"openai": config.OPENAI_API_KEY, "anthropic": config.ANTHROPIC_API_KEY,
               "groq": config.GROQ_API_KEY}.get(provider, "")
        if not key:
            log.warning("no API key for frontier provider '%s'; fallback disabled", provider)
            return None
        return cls(provider)

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        if self.provider == "anthropic":
            import anthropic

            self._client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        else:
            from openai import OpenAI

            if self.provider == "groq":
                self._client = OpenAI(api_key=config.GROQ_API_KEY, base_url=config.GROQ_BASE_URL)
            else:
                self._client = OpenAI(api_key=config.OPENAI_API_KEY)

    def generate(self, messages: list[dict[str, str]]) -> LLMResult:
        self._ensure_client()
        start = time.perf_counter()
        if self.provider == "anthropic":
            system = "\n".join(m["content"] for m in messages if m["role"] == "system")
            turns = [m for m in messages if m["role"] != "system"]
            resp = self._client.messages.create(model=self.model, system=system, messages=turns,
                                                max_tokens=config.MAX_NEW_TOKENS, temperature=0.0)
            text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            tin, tout = resp.usage.input_tokens, resp.usage.output_tokens
        else:
            resp = self._client.chat.completions.create(
                model=self.model, messages=messages, temperature=0.0, max_tokens=config.MAX_NEW_TOKENS,
                response_format={"type": "json_object"})
            text = resp.choices[0].message.content or ""
            tin = resp.usage.prompt_tokens if resp.usage else 0
            tout = resp.usage.completion_tokens if resp.usage else 0
        return LLMResult(text, [], tin, tout, (time.perf_counter() - start) * 1000)


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #
@dataclass
class RouteResult:
    route: str                                   # "local" | "frontier"
    decision: dict[str, str] | None
    confidence: ConfidenceBreakdown
    threshold: float
    agent: AgentResult
    frontier: LLMResult | None = None
    telemetry: dict[str, Any] = field(default_factory=dict)
    fallback_error: str | None = None


class HybridRouter:
    def __init__(self, agent: TriageAgent, frontier: FrontierLLM | None = None,
                 threshold: float | None = None) -> None:
        self.agent = agent
        self.frontier = frontier
        self.threshold = config.load_calibrated_threshold() if threshold is None else threshold

    @classmethod
    def from_config(cls, threshold: float | None = None, with_rag: bool = True,
                    with_vision: bool = True) -> "HybridRouter":
        """Wire the full production stack from config + environment."""
        from agentic_workflow.agent_router import LocalLLM
        from agentic_workflow.tools import build_toolbox

        retriever = classifier = None
        if with_rag:
            try:
                from rag_engine.retriever import HybridRetriever

                retriever = HybridRetriever()
            except Exception as exc:
                log.warning("RAG disabled: %s", exc)
        if with_vision and config.CNN_WEIGHTS_PATH.exists():
            try:
                from vision_cnn.classify_image import CNNClassifier

                classifier = CNNClassifier()
            except Exception as exc:
                log.warning("CNN disabled: %s", exc)
        agent = TriageAgent(LocalLLM(), build_toolbox(retriever, classifier))
        return cls(agent, FrontierLLM.try_create(), threshold)

    def route(self, ticket_text: str, customer_tier: str, image_path: str | None = None,
              mode: str = "agent", threshold: float | None = None, allow_fallback: bool = True) -> RouteResult:
        thr = self.threshold if threshold is None else threshold
        start = time.perf_counter()
        local = self.agent.run(ticket_text, customer_tier, image_path, mode)
        conf = compute_confidence(local.token_logprobs, local.vision["confidence"] if local.vision else None,
                                  local.rag_relevance, local.valid)

        route, decision, frontier_res, error = "local", local.decision, None, None
        if allow_fallback and conf.c_total < thr:
            if self.frontier is None:
                error = "frontier client not configured"
            else:
                try:
                    frontier_res = self.frontier.generate(local.messages)
                    parsed = parse_decision(frontier_res.text)
                    if parsed:
                        route, decision = "frontier", parsed
                    else:
                        error = "frontier returned invalid JSON"
                except Exception as exc:
                    error = f"frontier call failed: {exc}"
                    log.warning(error)

        tin = local.tokens_in + (frontier_res.tokens_in if frontier_res else 0)
        tout = local.tokens_out + (frontier_res.tokens_out if frontier_res else 0)
        cost = local_cost(local.tokens_in, local.tokens_out)
        if frontier_res:
            cost += frontier_cost(self.frontier.model, frontier_res.tokens_in, frontier_res.tokens_out)
        telemetry = {
            "route": route, "latency_ms": (time.perf_counter() - start) * 1000,
            "local_latency_ms": local.total_latency_ms,
            "frontier_latency_ms": frontier_res.latency_ms if frontier_res else 0.0,
            "tokens_in": tin, "tokens_out": tout, "cost_usd": cost,
        }
        return RouteResult(route, decision, conf, thr, local, frontier_res, telemetry, error)
