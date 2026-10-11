"""FastAPI service: /predict (CNN + local LLM) and /agent-route (full agentic RAG + hybrid router).

Run:  uvicorn serving.app:app --host 0.0.0.0 --port 8080
"""
from __future__ import annotations

import base64
import binascii
import io
import logging
import os
import tempfile
import threading
import uuid
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Iterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

import config
from router.hybrid_router import HybridRouter, RouteResult
from serving.schema import (AgentRouteResponse, ConfidenceInfo, HealthResponse, MetricsResponse,
                            PredictResponse, Telemetry, TicketRequest, VisionInfo)

log = logging.getLogger("modelroute.api")
MAX_IMAGE_BYTES = 6 * 1024 * 1024


class State:
    router: HybridRouter | None = None
    lock = threading.Lock()
    requests = 0
    fallbacks = 0
    latency_sum = 0.0
    cost_sum = 0.0


@asynccontextmanager
async def lifespan(_: FastAPI):
    logging.basicConfig(level=logging.INFO)
    State.router = HybridRouter.from_config()
    log.info("ModelRoute ready (threshold=%.3f)", State.router.threshold)
    yield


app = FastAPI(
    title=config.PROJECT_NAME, version=config.PROJECT_VERSION, lifespan=lifespan,
    description=f"Multimodal support-ticket routing. {config.AUTHOR_TAGLINE}.",
    contact={"name": config.AUTHOR, "url": config.AUTHOR_GITHUB})
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@contextmanager
def resolved_image(req: TicketRequest) -> Iterator[str | None]:
    """Yield a filesystem path for the attachment (decoding base64 to a temp file when needed)."""
    if req.image_base64:
        try:
            raw = base64.b64decode(req.image_base64.split(",")[-1], validate=True)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(422, "image_base64 is not valid base64") from exc
        if len(raw) > MAX_IMAGE_BYTES:
            raise HTTPException(413, "image too large (max 6 MB)")
        try:
            Image.open(io.BytesIO(raw)).verify()
        except Exception as exc:
            raise HTTPException(422, "attachment is not a valid image") from exc
        handle = tempfile.NamedTemporaryFile(suffix=".img", delete=False)
        try:
            handle.write(raw)
            handle.close()
            yield handle.name
        finally:
            os.unlink(handle.name)
    elif req.image_path:
        path = config.resolve_data_path(req.image_path).resolve()
        if not path.is_relative_to(config.DATA_DIR.resolve()) or not path.is_file():
            raise HTTPException(422, "image_path must be an existing file under the data directory")
        yield str(path)
    else:
        yield None


def _execute(req: TicketRequest, mode: str) -> tuple[RouteResult, str]:
    if State.router is None:
        raise HTTPException(503, "router not initialised")
    with resolved_image(req) as image:
        result = State.router.route(req.ticket_text, req.customer_tier, image, mode=mode,
                                    threshold=req.threshold, allow_fallback=req.allow_fallback)
    if result.decision is None:
        raise HTTPException(502, f"no valid decision produced ({result.fallback_error or 'local model invalid'})")
    tel = result.telemetry
    with State.lock:
        State.requests += 1
        State.fallbacks += int(result.route == "frontier")
        State.latency_sum += tel["latency_ms"]
        State.cost_sum += tel["cost_usd"]
    return result, uuid.uuid4().hex[:12]


def _base_payload(result: RouteResult, request_id: str) -> dict:
    c, v = result.confidence, result.agent.vision
    return dict(
        request_id=request_id, **result.decision,
        confidence=ConfidenceInfo(c_total=c.c_total, c_text=c.c_text, c_vision=c.c_vision, c_rag=c.c_rag,
                                  threshold=result.threshold, weights=c.weights),
        telemetry=Telemetry(**result.telemetry),
        vision=VisionInfo(label=v["label"], confidence=v["confidence"]) if v else None,
        fallback_error=result.fallback_error)


@app.get("/")
def root() -> dict:
    return {"service": config.PROJECT_NAME, "version": config.PROJECT_VERSION, "credit": config.AUTHOR_TAGLINE,
            "docs": "/docs"}


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    router = State.router
    agent = router.agent if router else None
    return HealthResponse(
        status="ok" if router else "starting", service=config.PROJECT_NAME, version=config.PROJECT_VERSION,
        author=config.AUTHOR,
        components={"rag": bool(agent and "RAGKnowledgeSearch" in agent.tools),
                    "cnn": bool(agent and "CNNVisualDiagnostic" in agent.tools),
                    "frontier_fallback": bool(router and router.frontier)},
        threshold=router.threshold if router else config.ROUTER_THRESHOLD)


@app.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    with State.lock:
        n = State.requests
        return MetricsResponse(
            requests=n, fallbacks=State.fallbacks,
            fallback_rate_pct=round(100.0 * State.fallbacks / n, 2) if n else 0.0,
            avg_latency_ms=round(State.latency_sum / n, 2) if n else 0.0,
            total_cost_usd=round(State.cost_sum, 6))


@app.post("/predict", response_model=PredictResponse)
def predict(req: TicketRequest) -> PredictResponse:
    """Direct mode: CNN + fine-tuned local LLM, with confidence-gated frontier fallback."""
    result, request_id = _execute(req, "direct")
    return PredictResponse(**_base_payload(result, request_id))


@app.post("/agent-route", response_model=AgentRouteResponse)
def agent_route(req: TicketRequest) -> AgentRouteResponse:
    """Full ModelRoute: agentic RAG + CNN + local LLM + dynamic hybrid router."""
    result, request_id = _execute(req, "agent")
    resp = result.agent.to_response()
    return AgentRouteResponse(**_base_payload(result, request_id), rag_sources=resp["rag_sources"],
                              tool_calls_made=resp["tool_calls_made"],
                              sla_response_hours=resp["sla_response_hours"], trace=result.agent.trace)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("serving.app:app", host=config.API_HOST, port=config.API_PORT)
