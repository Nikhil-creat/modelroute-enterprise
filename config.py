"""Central configuration for ModelRoute Enterprise.

Every module imports constants from here so that paths, label spaces, prompt
templates, model names, router weights and cost assumptions stay consistent
across the data pipeline, training, serving and evaluation code.

Run all scripts from the project root with ``python -m package.module``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("MR_DATA_DIR", BASE_DIR / "data"))
IMAGE_DIR = DATA_DIR / "images"
KB_DIR = DATA_DIR / "kb"
SPLIT_DIR = DATA_DIR / "splits"
VECTOR_DIR = DATA_DIR / "vector_store"

ARTIFACT_DIR = Path(os.getenv("MR_ARTIFACT_DIR", BASE_DIR / "artifacts"))
CNN_DIR = ARTIFACT_DIR / "cnn"
ADAPTER_DIR = ARTIFACT_DIR / "qlora_adapter"
MERGED_MODEL_DIR = ARTIFACT_DIR / "merged_fp16"
REPORT_DIR = ARTIFACT_DIR / "reports"

TICKETS_PATH = DATA_DIR / "tickets.jsonl"
KB_PATH = KB_DIR / "knowledge_base.jsonl"
CHUNKS_PATH = VECTOR_DIR / "chunks.jsonl"
FAISS_PATH = VECTOR_DIR / "faiss.index"
CNN_WEIGHTS_PATH = CNN_DIR / "cnn_best.pt"
CNN_META_PATH = CNN_DIR / "cnn_meta.json"

TRAIN_PATH = SPLIT_DIR / "train.jsonl"
VAL_PATH = SPLIT_DIR / "val.jsonl"
TEST_PATH = SPLIT_DIR / "test.jsonl"
SFT_TRAIN_PATH = SPLIT_DIR / "sft_train.jsonl"
SFT_VAL_PATH = SPLIT_DIR / "sft_val.jsonl"

# --------------------------------------------------------------------------- #
# Reproducibility / data generation
# --------------------------------------------------------------------------- #
SEED = 42
N_TICKETS = 3200
IMAGE_SIZE = 224
HARD_TICKET_RATE = 0.08          # tickets with a distractor sentence + vague target
SPLIT_RATIOS = (0.70, 0.15, 0.15)  # train / val / test

# --------------------------------------------------------------------------- #
# Label spaces
# --------------------------------------------------------------------------- #
VISUAL_CLASSES = ["UI_ERROR", "HARDWARE_FAULT", "SPAM_IMAGE", "VALID_DOCUMENT"]

INTENTS = [
    "ui_bug",
    "hardware_failure",
    "billing_issue",
    "account_access",
    "performance_degradation",
    "security_incident",
    "feature_request",
    "spam_or_irrelevant",
]

PRIORITIES = ["P1", "P2", "P3", "P4"]
PRIORITY_NAMES = {"P1": "critical", "P2": "high", "P3": "medium", "P4": "low"}

INTENT_TO_DEPARTMENT = {
    "ui_bug": "Frontend Engineering",
    "hardware_failure": "Hardware Support",
    "billing_issue": "Billing & Finance",
    "account_access": "Identity & Access",
    "performance_degradation": "Platform SRE",
    "security_incident": "Security Operations",
    "feature_request": "Product Management",
    "spam_or_irrelevant": "Trust & Safety",
}
DEPARTMENTS = sorted(set(INTENT_TO_DEPARTMENT.values()))

CUSTOMER_TIERS = ["free", "pro", "enterprise"]
TIER_WEIGHTS = [0.45, 0.35, 0.20]

# SLA response targets in hours: tier -> priority -> hours (used by AccountStatusLookup)
SLA_HOURS = {
    "enterprise": {"P1": 1, "P2": 4, "P3": 8, "P4": 24},
    "pro": {"P1": 4, "P2": 8, "P3": 24, "P4": 72},
    "free": {"P1": 24, "P2": 48, "P3": 72, "P4": 120},
}

# --------------------------------------------------------------------------- #
# Prompt templates (shared by SFT data, agent, local model and frontier fallback)
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = (
    "You are ModelRoute, an enterprise support triage engine. Read the support ticket, "
    "the image diagnosis and the knowledge base context, then respond with a single JSON "
    "object and nothing else. Keys: intent, priority, target_department. "
    f"intent must be one of {INTENTS}. priority must be one of {PRIORITIES} "
    "(P1 critical ... P4 low). target_department must be the department that owns the intent."
)

USER_PROMPT_TEMPLATE = (
    "Customer tier: {tier}\n"
    "Image diagnosis: {visual}\n"
    "Knowledge base context:\n{kb_context}\n\n"
    "Ticket:\n{ticket_text}\n\n"
    "Return the JSON triage decision."
)

# --------------------------------------------------------------------------- #
# RAG
# --------------------------------------------------------------------------- #
EMBEDDING_MODEL = os.getenv("MR_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_WORDS = 90
CHUNK_OVERLAP_WORDS = 20
CHROMA_COLLECTION = "modelroute_kb"
CHROMA_HOST = os.getenv("CHROMA_HOST", "")          # empty -> embedded persistent client
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
RAG_TOP_K = 4
HYBRID_ALPHA = 0.6               # weight of dense score; (1 - alpha) goes to BM25
RRF_K = 60

# --------------------------------------------------------------------------- #
# Vision CNN
# --------------------------------------------------------------------------- #
CNN_BACKBONE = "resnet18"        # "resnet18" | "mobilenet_v3_small" | "custom"
CNN_EPOCHS = 8
CNN_BATCH_SIZE = 64
CNN_LR = 1e-3
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# --------------------------------------------------------------------------- #
# Local LLM / QLoRA
# --------------------------------------------------------------------------- #
BASE_MODEL = os.getenv("MR_BASE_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
ALT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]
QLORA_QUANT_TYPE = "nf4"
QLORA_DOUBLE_QUANT = True
SFT_EPOCHS = 2
SFT_BATCH_SIZE = 4
SFT_GRAD_ACCUM = 4
SFT_LR = 2e-4
SFT_MAX_SEQ_LEN = 1024
MAX_NEW_TOKENS = 96

# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #
# C_total = w1*C_text_logprob + w2*C_vision + w3*C_rag_relevance
ROUTER_WEIGHTS = {"text": 0.50, "vision": 0.20, "rag": 0.30}
ROUTER_THRESHOLD = float(os.getenv("MR_ROUTER_THRESHOLD") or 0.85)
THRESHOLD_GRID = [round(0.50 + 0.025 * i, 3) for i in range(0, 20)]   # 0.50 ... 0.975

# --------------------------------------------------------------------------- #
# Frontier fallback + cost model (USD per 1M tokens; edit to match current pricing)
# --------------------------------------------------------------------------- #
FRONTIER_PROVIDER = os.getenv("MR_FRONTIER_PROVIDER", "openai")   # "openai" | "anthropic" | "groq"
FRONTIER_MODELS = {"openai": "gpt-4o", "anthropic": "claude-3-5-sonnet-latest",
                   "groq": "llama-3.3-70b-versatile"}
FRONTIER_PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "claude-3-5-sonnet-latest": {"input": 3.00, "output": 15.00},
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},   # Groq, assumed
}
LOCAL_GPU_COST_PER_HOUR = 1.10   # assumed on-demand 24 GB GPU
LOCAL_TOKENS_PER_SECOND = 55.0   # assumed single-stream throughput

# --------------------------------------------------------------------------- #
# Serving
# --------------------------------------------------------------------------- #
API_HOST = os.getenv("MR_API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("MR_API_PORT", "8080"))
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "")                     # OpenAI-compatible local server
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")                       # free tier key from console.groq.com
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


# --------------------------------------------------------------------------- #
# Branding, runtime wiring, calibrated threshold
# --------------------------------------------------------------------------- #
AUTHOR = "NIKHIL CHARY SRIRAMOJU"
AUTHOR_TAGLINE = "Designed and developed by NIKHIL CHARY SRIRAMOJU"
AUTHOR_GITHUB = "https://github.com/Nikhil-creat"
AUTHOR_LINKEDIN = "https://in.linkedin.com/in/nikhil-chary-sriramoju-95041b38a"
PROJECT_NAME = "ModelRoute Enterprise"
PROJECT_VERSION = "1.0.0"

LOCAL_MODEL_NAME = os.getenv("MR_LOCAL_MODEL_NAME", "modelroute-local")   # vLLM served-model-name
RAG_BACKEND = os.getenv("MR_RAG_BACKEND", "chroma")                       # "chroma" | "faiss"
RAG_CONTEXT_K = 3                 # KB articles injected into the prompt (matches SFT data)
RAG_REFLECT_THRESHOLD = 0.45      # agent re-queries with the visual hint below this relevance
THRESHOLD_FILE = REPORT_DIR / "threshold.json"
DOCS_DATA_DIR = BASE_DIR / "docs" / "data"


def load_calibrated_threshold(default: float = ROUTER_THRESHOLD) -> float:
    """Env override > calibrated file written by evaluation.calibrate_threshold > default."""
    if os.getenv("MR_ROUTER_THRESHOLD"):
        return float(os.environ["MR_ROUTER_THRESHOLD"])
    if THRESHOLD_FILE.exists():
        try:
            return float(json.loads(THRESHOLD_FILE.read_text(encoding="utf-8"))["threshold"])
        except (ValueError, KeyError, OSError):
            return default
    return default


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def ensure_dirs() -> None:
    """Create every directory the pipeline writes to."""
    for path in (
        DATA_DIR, IMAGE_DIR, KB_DIR, SPLIT_DIR, VECTOR_DIR,
        ARTIFACT_DIR, CNN_DIR, ADAPTER_DIR, MERGED_MODEL_DIR, REPORT_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def resolve_data_path(relative: str | os.PathLike[str]) -> Path:
    """Resolve an ``image_path`` stored relative to ``DATA_DIR``."""
    path = Path(relative)
    return path if path.is_absolute() else DATA_DIR / path


def read_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | os.PathLike[str], rows: Iterable[dict[str, Any]]) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count
