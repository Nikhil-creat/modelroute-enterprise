"""Stateful ReAct-style triage agent.

Flow (mode="agent"):  observe ticket -> CNNVisualDiagnostic (if attachment) -> RAGKnowledgeSearch
(re-queried with the visual hint when relevance is low) -> local fine-tuned LLM decision ->
AccountStatusLookup for the SLA of the chosen priority -> structured response.
Mode "direct" runs only the CNN + LLM (the "pure local model + CNN" baseline).
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

import config
from agentic_workflow.tools import Tool, ToolResult

log = logging.getLogger("modelroute.agent")

VISUAL_HINT = {
    "UI_ERROR": "user interface error screen",
    "HARDWARE_FAULT": "hardware crash fault device",
    "SPAM_IMAGE": "spam advertisement phishing",
    "VALID_DOCUMENT": "invoice receipt document log",
}


# --------------------------------------------------------------------------- #
# LLM clients
# --------------------------------------------------------------------------- #
@dataclass
class LLMResult:
    text: str
    logprobs: list[float] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0


class LocalLLM:
    """Merged FP16 fine-tuned model, via a vLLM OpenAI-compatible server or in-process transformers."""

    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        self.base_url = base_url if base_url is not None else config.VLLM_BASE_URL
        self.model = model or config.LOCAL_MODEL_NAME
        self._client = None
        self._tok = None
        self._hf = None

    def generate(self, messages: list[dict[str, str]]) -> LLMResult:
        start = time.perf_counter()
        result = self._generate_openai(messages) if self.base_url else self._generate_hf(messages)
        result.latency_ms = (time.perf_counter() - start) * 1000
        return result

    def _generate_openai(self, messages: list[dict[str, str]]) -> LLMResult:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(base_url=self.base_url, api_key="EMPTY")
        resp = self._client.chat.completions.create(
            model=self.model, messages=messages, temperature=0.0, max_tokens=config.MAX_NEW_TOKENS,
            logprobs=True, top_logprobs=1)
        choice = resp.choices[0]
        lps = [t.logprob for t in (choice.logprobs.content or [])] if choice.logprobs else []
        usage = resp.usage
        return LLMResult(choice.message.content or "", lps, usage.prompt_tokens if usage else 0,
                         usage.completion_tokens if usage else len(lps))

    def _generate_hf(self, messages: list[dict[str, str]]) -> LLMResult:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self._hf is None:
            self._tok = AutoTokenizer.from_pretrained(config.MERGED_MODEL_DIR)
            self._hf = AutoModelForCausalLM.from_pretrained(config.MERGED_MODEL_DIR, torch_dtype=torch.float16,
                                                            device_map="auto")
            self._hf.eval()
        ids = self._tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                            return_dict=True).to(self._hf.device)
        with torch.inference_mode():
            out = self._hf.generate(**ids, max_new_tokens=config.MAX_NEW_TOKENS, do_sample=False,
                                    output_scores=True, return_dict_in_generate=True,
                                    pad_token_id=self._tok.eos_token_id)
        prompt_len = ids["input_ids"].shape[1]
        scores = self._hf.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
        generated = out.sequences[0, prompt_len:]
        text = self._tok.decode(generated, skip_special_tokens=True)
        return LLMResult(text, scores[0].float().cpu().tolist(), prompt_len, int(generated.shape[0]))


# --------------------------------------------------------------------------- #
# Prompt + parsing helpers
# --------------------------------------------------------------------------- #
def build_messages(ticket_text: str, tier: str, visual: str, kb_context: str) -> list[dict[str, str]]:
    user = config.USER_PROMPT_TEMPLATE.format(tier=tier, visual=visual, kb_context=kb_context,
                                              ticket_text=ticket_text)
    return [{"role": "system", "content": config.SYSTEM_PROMPT}, {"role": "user", "content": user}]


_JSON_OBJECT = re.compile(r"\{.*?\}", re.S)


def parse_decision(text: str) -> dict[str, str] | None:
    """Extract and validate the first JSON object with intent / priority / target_department."""
    for match in _JSON_OBJECT.finditer(text or ""):
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if (data.get("intent") in config.INTENTS and data.get("priority") in config.PRIORITIES
                and data.get("target_department") in config.DEPARTMENTS):
            return {k: data[k] for k in ("intent", "priority", "target_department")}
    return None


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #
@dataclass
class AgentResult:
    decision: dict[str, str] | None
    valid: bool
    raw_text: str
    messages: list[dict[str, str]]
    mode: str
    rag_sources: list[dict[str, Any]] = field(default_factory=list)
    tool_calls_made: list[dict[str, Any]] = field(default_factory=list)
    vision: dict[str, Any] | None = None
    rag_relevance: float | None = None
    sla: dict[str, Any] | None = None
    token_logprobs: list[float] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    llm_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    trace: list[str] = field(default_factory=list)

    def to_response(self) -> dict[str, Any]:
        d = self.decision or {}
        return {"intent": d.get("intent"), "priority": d.get("priority"),
                "target_department": d.get("target_department"), "rag_sources": self.rag_sources,
                "tool_calls_made": self.tool_calls_made,
                "sla_response_hours": (self.sla or {}).get("response_hours")}


class TriageAgent:
    def __init__(self, llm: Any, tools: dict[str, Tool]) -> None:
        self.llm = llm
        self.tools = tools

    def _call(self, name: str, calls: list[dict[str, Any]], trace: list[str], **kwargs: Any) -> ToolResult | None:
        tool = self.tools.get(name)
        if tool is None:
            return None
        result = tool(**kwargs)
        calls.append(result.summary())
        trace.append(f"action {name} -> {'ok' if result.ok else 'error: ' + str(result.error)}")
        if not result.ok:
            log.warning("tool %s failed: %s", name, result.error)
        return result

    def run(self, ticket_text: str, customer_tier: str, image_path: str | None = None,
            mode: str = "agent") -> AgentResult:
        start = time.perf_counter()
        calls: list[dict[str, Any]] = []
        trace: list[str] = [f"observe ticket ({len(ticket_text)} chars, tier={customer_tier}, "
                            f"attachment={'yes' if image_path else 'no'}, mode={mode})"]
        vision: dict[str, Any] | None = None
        rag: dict[str, Any] | None = None

        if image_path:
            res = self._call("CNNVisualDiagnostic", calls, trace, image_path=image_path)
            if res and res.ok:
                vision = res.output
                trace.append(f"observe attachment={vision['label']} ({vision['confidence']:.2f})")

        if mode == "agent":
            res = self._call("RAGKnowledgeSearch", calls, trace, query=ticket_text)
            if res and res.ok:
                rag = res.output
            if vision and (rag is None or rag["relevance"] < config.RAG_REFLECT_THRESHOLD):
                trace.append("reflect: low retrieval relevance, re-querying with the visual hint")
                res = self._call("RAGKnowledgeSearch", calls, trace,
                                 query=f"{VISUAL_HINT.get(vision['label'], '')} {ticket_text}")
                if res and res.ok and (rag is None or res.output["relevance"] > rag["relevance"]):
                    rag = res.output

        visual = (f"{vision['label']} (confidence {vision['confidence']:.2f})" if vision
                  else "none (no attachment)")
        kb_context = rag["context"] if rag else "- (none retrieved)"
        messages = build_messages(ticket_text, customer_tier, visual, kb_context)
        llm = self.llm.generate(messages)
        decision = parse_decision(llm.text)
        trace.append(f"decide {decision}" if decision else "decide: model output was not valid JSON")

        sla = None
        if mode == "agent" and decision:
            res = self._call("AccountStatusLookup", calls, trace, customer_tier=customer_tier,
                             priority=decision["priority"])
            if res and res.ok:
                sla = res.output

        return AgentResult(
            decision=decision, valid=decision is not None, raw_text=llm.text, messages=messages, mode=mode,
            rag_sources=rag["hits"] if rag else [], tool_calls_made=calls, vision=vision,
            rag_relevance=rag["relevance"] if rag else None, sla=sla, token_logprobs=llm.logprobs,
            tokens_in=llm.tokens_in, tokens_out=llm.tokens_out, llm_latency_ms=llm.latency_ms,
            total_latency_ms=(time.perf_counter() - start) * 1000, trace=trace)
