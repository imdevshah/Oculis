<div align="center">

# 🔍 Oculis
**Multimodal Agentic RAG · VLM Document Reasoning · Self-Correcting Hallucination Guardrails**
**Built for Verifiable Document Intelligence.**

## 🔍 Overview

LLMs hallucinate confidently. A wrong answer and a right answer look identical — same fluent tone, same certainty. For document intelligence use cases, that is not acceptable.

**Oculis** solves this. It is a multimodal agentic RAG system where every answer comes with a verifiable confidence score derived from three independent guardrail checks. It reads text AND images from PDFs, reasons over retrieved content using a ReAct agent, then verifies every answer before delivering it.

```
Upload PDF  →  Agent retrieves + reasons  →  3-layer guardrail check  →  Answer + confidence score
```

### What makes it different from standard RAG

| Feature | Standard RAG | Oculis |
|---|---|---|
| Text retrieval | ✅ | ✅ |
| Image / chart understanding | ❌ | ✅ via LLaVA |
| Agentic tool use | ❌ | ✅ ReAct loop |
| Hallucination detection | ❌ | ✅ 3-layer guardrails |
| Confidence score per answer | ❌ | ✅ 0.0 – 1.0 |
| Source citation | Partial | ✅ file + page |
| 100% free to run | ❌ | ✅ local + Groq free tier |

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINE                           │
│                                                                     │
│  PDF File                                                           │
│     │                                                               │
│     ▼                                                               │
│  pdf_parser.py ──────────────────────────────────┐                 │
│  (PyMuPDF)          pages[]          images[]     │                 │
│     │                  │                │         │                 │
│     ▼                  ▼               ▼          │                 │
│  chunker.py      chunk_pages()   vlm_processor.py │                 │
│  500-char            │           (LLaVA/Ollama)   │                 │
│  sliding window      │                │           │                 │
│                       └──────┬────────┘           │                 │
│                              ▼                    │                 │
│                     add_image_captions()          │                 │
│                     all chunks unified            │                 │
│                              │                    │                 │
│                              ▼                    │                 │
│                     embedder.py                   │                 │
│                     all-MiniLM-L6-v2              │                 │
│                     → 384-dim vectors             │                 │
│                              │                    │                 │
│                              ▼                    │                 │
│                     vector_store.py ──────► ChromaDB               │
│                     (abstraction layer)     (cosine index)          │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                         QUERY PIPELINE                              │
│                                                                     │
│  User Question                                                      │
│       │                                                             │
│       ▼                                                             │
│  agent.py  ◄──── ReAct loop (think → act → observe → repeat)       │
│  llama-3.1-70b                                                      │
│       │                                                             │
│       ├──► rag_search ──► retriever.py ──► ChromaDB                │
│       │    (always first)  embed_query()    cosine similarity       │
│       │                    same model!      top-K chunks            │
│       │                                                             │
│       ├──► web_search   (DuckDuckGo — fallback only)               │
│       │                                                             │
│       └──► calculate    (sandboxed eval — no builtins)             │
│                                                                     │
│  raw answer                                                         │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────────────┐               │
│  │              GUARDRAILS LAYER                   │               │
│  │                                                 │               │
│  │  checker.py orchestrates three checks:          │               │
│  │                                                 │               │
│  │  ① SelfCheck   ② NLI Scorer   ③ Faithfulness   │               │
│  │    3 samples     contradiction    grounding      │               │
│  │    consistency   detection       chain-of-thought│               │
│  │                                                 │               │
│  │  confidence = (consistency + nli + faith) / 3  │               │
│  │  flagged = confidence < 0.60                    │               │
│  └─────────────────────────────────────────────────┘               │
│       │                                                             │
│       ▼                                                             │
│  answer + confidence + flagged + warning                            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🛡 Hallucination Guardrails

Hallucinations come in different shapes. Three checks, each catching a different failure mode.

---

### Check 1 — SelfCheck Consistency (`checker.py`)

**The jury analogy:** Ask the same question to 3 independent witnesses at `temperature=0.7`. If the answer is grounded in real document content, all 3 will agree. If it's fabricated, each guess will drift.

```python
# 3 independent samples → fraction that agree with main answer
consistency_score = agreements / SELFCHECK_SAMPLES   # 0.0 to 1.0
```

### Check 2 — NLI Contradiction Detection (`nli_scorer.py`)

Catches direct factual conflicts between the answer and the retrieved context. If the document says `$4.2M` and the answer says `$9.8M`, NLI scores `0.0`.

```python
# "Does the ANSWER contradict any facts in the CONTEXT?"
# → CONSISTENT (1.0) or CONTRADICTION (0.0)
```

### Check 3 — Faithfulness Grounding (`faithfulness.py`)

Catches a different failure mode from NLI — fabricated claims that don't contradict the document because they simply aren't mentioned at all. Uses chain-of-thought prompting to enumerate each claim and check it independently.

```python
# "Is EVERY claim in the ANSWER explicitly supported by the CONTEXT?"
# → SUPPORTED (1.0) or UNSUPPORTED (0.0)
```

### Combined Score

```python
confidence = round((consistency + nli + faithfulness) / 3, 2)
flagged    = confidence < CONFIDENCE_THRESHOLD   # default 0.60
```

**Why average three independent checks?** Each catches what the others miss. NLI passes fabrications (no contradiction). Faithfulness can miss direct inversions. SelfCheck catches both but less precisely. Together, they have no single point of failure.

### Confidence Score Interpretation

| Score | Status | UI Colour | Meaning |
|---|---|---|---|
| 0.80 – 1.00 | ✅ Passed | 🟢 Green | All three checks passed |
| 0.60 – 0.79 | ✅ Passed | 🟡 Amber | Borderline — review source |
| 0.00 – 0.59 | ⚠️ Flagged | 🔴 Red + banner | Likely hallucination |

---

```
📊 Summary
  Total queries        : 8
  Passed               : 7  (87.5%)
  Flagged              : 1  (12.5%)
  Avg confidence       : 82%
  Avg top similarity   : 97.7%
  Hallucination caught : ✅ Yes — Q8 at 13% confidence
```

---

## ⚡ Quickstart

### Prerequisites

- Python 3.11+
- Docker Desktop
- Ollama (for LLaVA image captioning)
- Groq API key (free at [console.groq.com](https://console.groq.com))

### 1. Clone and install

```bash
git clone https://github.com/yourusername/oculis.git
cd oculis
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set up environment

```bash
cp .env.example .env
# Add your GROQ_API_KEY to .env
```

### 3. Pull the VLM model

```bash
ollama pull llava
```


### 4. Start with API

```bash
# Seed dummy data to test without a real PDF
python create_dummy_data.py

# Start the API
uvicorn api.main:app --reload --port 8000

# Open web/index.html in browser
```

---

## 📁 Project Structure

```
Oculis/
├── config.py                   # All settings — models, paths, thresholds
├── create_dummy_data.py        # Seed ChromaDB without a real PDF
├── agent/
│   ├── agent.py                # ReAct agent — answer() + answer_streaming()
│   └── tools.py                # rag_search, web_search, calculate
│
├── api/
│   └── main.py                 # FastAPI — POST /upload, POST /ask, GET /health
│
├── guardrails/
│   ├── checker.py              # Orchestrates all 3 checks → confidence score
│   ├── nli_scorer.py           # Contradiction detection
│   └── faithfulness.py         # Grounding check with chain-of-thought
│
├── ingestion/
│   ├── pipeline.py             # Orchestrator — run_pipeline(pdf_path)
│   ├── pdf_parser.py           # PyMuPDF — text pages + image bytes
│   ├── vlm_processor.py        # LLaVA captions every image
│   ├── chunker.py              # 500-char sliding window, 50-char overlap
│   ├── embedder.py             # all-MiniLM-L6-v2 → 384-dim vectors
│   └── vector_store.py         # ChromaDB abstraction layer
│
├── retrieval/
│   └── retriever.py            # embed_query() + cosine similarity search
│
├── eval/
│   ├── logger.py               # Logs every query to JSON for analysis
│   └── logs/
│       └── oculis_log.json     # Real query logs
│
└── web/
    ├── index.html              # Chat UI — upload, ask, confidence bar
    └── oculis_dashboard.html   # Analytics dashboard — 7 charts
```

---

## 🛠 Tech Stack

| Component | Technology | Why |
|---|---|---|
| Text embedding | `all-MiniLM-L6-v2` | Fast, 384-dim, fully local |
| Vector database | ChromaDB | No server needed, cosine index |
| Agent LLM | `llama-3.1-70b` via Groq | Fast inference, free tier |
| Guardrails LLM | `llama-3.1-8b` via Groq | Fast, cheap, binary decisions |
| VLM captioning | LLaVA via Ollama | Local, private, no API cost |
| PDF parsing | PyMuPDF (fitz) | Handles text + embedded images |
| Agent framework | LangChain ReAct | Tool routing, history, scratchpad |
| API | FastAPI + Uvicorn | Async, auto /docs, fast |
| Deployment | Docker Compose | One command, full system |

**Total running cost: $0.** Every model runs locally or on Groq's free tier.

---

## 🧠 Key Design Decisions

**Why the same embedding model for ingestion and retrieval?**
Embedding models create a coordinate space. Query vectors and document vectors must inhabit the same space for cosine similarity to mean anything. Mixing models produces meaningless comparisons.

**Why VLM + RAG instead of just RAG?**
Standard RAG is blind to images. A financial report's most important data is often in a chart. LLaVA converts every chart, diagram, and table into a plain-English caption stored alongside text chunks — making visual content fully searchable.

**Why three guardrail checks instead of one?**
NLI catches contradictions but passes fabrications. Faithfulness catches fabrications but can miss inversions. SelfCheck catches both but less precisely. No single check covers all failure modes. Together, they have no single point of failure.

**Why `VectorStore` as a wrapper class?**
If you swap ChromaDB for Pinecone or Qdrant, you change one file. Everything else stays identical.

**Why `temperature=0.7` for SelfCheck samples but `temperature=0` everywhere else?**
SelfCheck needs genuine variation between samples — 3 identical witnesses prove nothing. The variation forces each sample to independently anchor to the retrieved document. If there's a real fact there, all 3 find it.

---

## 🔮 Roadmap

- [ ] `eval/` — RAGAS-style evaluation suite (faithfulness, relevancy, context recall)
- [ ] Streaming responses — token-by-token delivery to frontend
- [ ] OCR integration — Tesseract for scanned PDFs
- [ ] Cross-encoder reranker — improve chunk precision after cosine retrieval
- [ ] Multi-document cross-referencing — resolve conflicts across PDFs

---

## 🤝 Contributing

Pull requests are welcome. For major changes, open an issue first.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.



Built **Dev** and **Rohit** · College project that became something real


