"""Pydantic request/response models for the ModelRoute API."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class TicketRequest(BaseModel):
    ticket_text: str = Field(min_length=3, max_length=8000)
    customer_tier: Literal["free", "pro", "enterprise"] = "free"
    image_base64: str | None = Field(default=None, description="PNG/JPEG attachment, base64 encoded")
    image_path: str | None = Field(default=None, description="path under the data directory")
    threshold: float | None = Field(default=None, ge=0.0, le=1.0, description="override the router threshold")
    allow_fallback: bool = True


class ConfidenceInfo(BaseModel):
    c_total: float
    c_text: float
    c_vision: float | None
    c_rag: float | None
    threshold: float
    weights: dict[str, float]


class Telemetry(BaseModel):
    route: Literal["local", "frontier"]
    latency_ms: float
    local_latency_ms: float
    frontier_latency_ms: float
    tokens_in: int
    tokens_out: int
    cost_usd: float


class VisionInfo(BaseModel):
    label: str
    confidence: float


class PredictResponse(BaseModel):
    request_id: str
    intent: str
    priority: str
    target_department: str
    confidence: ConfidenceInfo
    telemetry: Telemetry
    vision: VisionInfo | None = None
    fallback_error: str | None = None


class RAGSource(BaseModel):
    kb_id: str
    title: str
    score: float


class ToolCall(BaseModel):
    tool: str
    ok: bool
    latency_ms: float


class AgentRouteResponse(PredictResponse):
    rag_sources: list[RAGSource]
    tool_calls_made: list[ToolCall]
    sla_response_hours: int | None = None
    trace: list[str] = []


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    author: str
    components: dict[str, bool]
    threshold: float


class MetricsResponse(BaseModel):
    requests: int
    fallbacks: int
    fallback_rate_pct: float
    avg_latency_ms: float
    total_cost_usd: float
    extra: dict[str, Any] = {}
