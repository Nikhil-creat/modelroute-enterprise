# ModelRoute Enterprise

**Designed and developed by NIKHIL CHARY SRIRAMOJU** · [GitHub](https://github.com/Nikhil-creat) · [LinkedIn](https://in.linkedin.com/in/nikhil-chary-sriramoju-95041b38a)

Multimodal support-ticket platform: a 4-bit **QLoRA** fine-tuned LLM, a PyTorch **CNN** for attachments, an **agentic hybrid RAG** workflow, and a **confidence router** that falls back to a frontier model (GPT-4o / Claude 3.5 Sonnet / Groq) only when the local stack is unsure.

```
ticket + image -> CNN diagnostic -> agent (hybrid RAG, SLA tool) -> QLoRA LLM (vLLM)
              -> C_total = w1*C_text + w2*C_vision + w3*C_rag -> C >= tau ? local : frontier
```

## Project tree
```
modelroute_enterprise/
├── config.py  requirements.txt  README.md  GITHUB_PAGES_GUIDE.md  .env.example  LICENSE
├── data_pipeline/    generate_synthetic_data.py  build_vector_store.py  split_data.py
├── vision_cnn/       train_cnn.py  classify_image.py
├── rag_engine/       vector_db.py  retriever.py
├── agentic_workflow/ tools.py  agent_router.py
├── fine_tuning/      train_qlora.py  merge_adapters.py
├── router/           confidence_engine.py  hybrid_router.py
├── serving/          app.py  schema.py
├── evaluation/       calibrate_threshold.py  benchmark.py
├── deployment/       Dockerfile  docker-compose.yml
├── tests/            smoke_test.py
└── docs/             index.html  data/benchmark.json   (GitHub Pages site)
```

## 1. Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add GROQ_API_KEY (free) or OPENAI/ANTHROPIC keys
set -a; source .env; set +a
python -m tests.smoke_test  # no GPU, models or network needed
```
Run every script from the project root with `python -m package.module`.

## 2. Data and indexes
```bash
python -m data_pipeline.generate_synthetic_data --n 3200   # tickets, KB, PNG attachments (~270 MB)
python -m data_pipeline.build_vector_store                 # Chroma (or --backend faiss|both)
python -m data_pipeline.split_data                         # 70/15/15 + chat-format SFT files
```

## 3. Train
```bash
python -m vision_cnn.train_cnn --backbone resnet18 --epochs 8       # CNN, saves calibrated weights
python -m fine_tuning.train_qlora --model meta-llama/Meta-Llama-3-8B-Instruct --epochs 2   # needs a CUDA GPU
python -m fine_tuning.merge_adapters --smoke-test                   # FP16 merged model
```
Llama-3 is gated on Hugging Face (accept the licence and `export HF_TOKEN=...`), or use `--model Qwen/Qwen2.5-7B-Instruct`.
QLoRA settings: NF4, double quantisation, r=16, alpha=32, targets `q_proj, k_proj, v_proj, o_proj`; loss on the assistant JSON only.

## 4. Calibrate, benchmark
```bash
python -m evaluation.calibrate_threshold --min-retention 98   # writes artifacts/reports/threshold.json
python -m evaluation.benchmark --limit 300                    # 3 modes, table + resume bullets
```
The router loads the calibrated threshold automatically (`MR_ROUTER_THRESHOLD` overrides it).

## 5. Serve
```bash
python -m serving.app            # http://localhost:8080/docs   (needs a vLLM server or the merged model locally)
```
- `POST /predict` : CNN + local LLM, confidence-gated fallback
- `POST /agent-route` : agentic RAG + CNN + LLM + hybrid router, returns `{intent, priority, target_department, rag_sources, tool_calls_made}` plus confidence, latency and cost telemetry
- `GET /health`, `GET /metrics`

```bash
curl -s localhost:8080/agent-route -H 'content-type: application/json' \
  -d '{"ticket_text":"Rack controller reboots with a kernel panic, production down","customer_tier":"enterprise"}'
```

## 6. Docker
```bash
docker compose -f deployment/docker-compose.yml --env-file .env up --build
docker compose -f deployment/docker-compose.yml run --rm api python -m data_pipeline.build_vector_store
```
Services: `api` (FastAPI), `chroma` (vector DB), `vllm` (serves `artifacts/merged_fp16`). Requires the NVIDIA container toolkit.

## 7. Website (GitHub Pages)
`docs/index.html` is a static showcase with a live tool-calling agent and a personal assistant chatbot powered by a **free Groq API key** entered in the browser. See `GITHUB_PAGES_GUIDE.md`.

## Design notes
- **Confidence:** `C_text = exp(mean token log-prob)`, `C_vision` = temperature-calibrated CNN softmax, `C_rag` = best fused (dense + BM25) score. Missing components are re-weighted; invalid JSON scores 0 and always falls back.
- **Agent:** deterministic ReAct loop (observe, act, reflect): CNN first, RAG next (re-queried with the visual hint if relevance is low), then the LLM decision, then an SLA lookup for the chosen priority.
- **Pure frontier baseline** sees ticket text only (no CNN, no KB), which is how the benchmark defines mode 1.
- **Synthetic data:** labels come from rules, so very high accuracy is expected; the "hard" tickets exist to exercise the router. Retrain on real tickets before production claims.

## Resume Bullet Block
`python -m evaluation.benchmark` writes `artifacts/reports/resume_bullets.md` with **your measured numbers**. Format:

```
ModelRoute Enterprise | Designed and developed by NIKHIL CHARY SRIRAMOJU
- Built a multimodal ticket-routing platform (4-bit QLoRA LLM + PyTorch CNN + agentic hybrid RAG) that cut
  cost per 1,000 requests by <X>% vs. a pure frontier model ($<A> -> $<B>).
- Designed a confidence router (text log-prob + CNN + RAG relevance, threshold <T>) that served <L>% of
  traffic locally, cut p95 latency by <Y>%.
- Retained <R>% of frontier accuracy (<a>% vs <b>%, macro-F1 <f>) on <N> held-out tickets.
- Shipped FastAPI + Docker (vLLM, ChromaDB) with per-request latency and cost telemetry.
```
Only publish figures the benchmark actually produced.

## License
MIT. Copyright (c) 2026 NIKHIL CHARY SRIRAMOJU.
