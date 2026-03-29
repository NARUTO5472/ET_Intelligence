# ET AI-Native News Intelligence Platform
### Economic Times GenAI Hackathon 2026 — Problem Statement 8

> *"Business news in 2026 is still delivered like it's 2005 — static text articles, one-size-fits-all homepage, same format for everyone."*

This platform transforms ET news consumption through three AI-powered modules, running entirely on **CPU-only hardware with 16 GB RAM** — no GPU, no VRAM, no cloud dependency.

---

## 🏗 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    ET AI Intelligence Platform                  │
│                    Streamlit Frontend (app.py)                  │
└──────────────┬──────────────────┬────────────────┬─────────────┘
               │                  │                │
    ┌──────────▼──────┐  ┌────────▼──────┐  ┌────▼──────────────┐
    │  My ET          │  │ News Navigator │  │ Story Arc Tracker  │
    │  Personalized   │  │ Intelligence   │  │ Knowledge Graph    │
    │  Newsroom       │  │ Briefings      │  │ Narrative Tracker  │
    └──────────┬──────┘  └────────┬──────┘  └────┬───────────────┘
               │                  │               │
    ┌──────────▼──────────────────▼───────────────▼───────────────┐
    │              Core Intelligence Layer                        │
    │  • LanceDB (disk-based IVF vector index)                    │
    │  • all-MiniLM-L6-v2 (384-dim embeddings, CPU-optimised)     │
    │  • Ollama llama3.2:3b q4_k_m (≈2.5 GB, 10-15 tok/s CPU)    │
    │  • GLiNER (zero-shot NER, CPU-optimised transformer)        │
    │  • NetworkX (in-memory knowledge graph)                     │
    └─────────────────────────────┬───────────────────────────────┘
                                  │
    ┌─────────────────────────────▼───────────────────────────────┐
    │           Data Ingestion Pipeline                           │
    │  feedparser → BeautifulSoup → process_article()            │
    │  Embed → NER → Sentiment → Chunk → LanceDB upsert          │
    └─────────────────────────────────────────────────────────────┘
```

---

## 💾 Memory Budget (16 GB RAM)

| Component                    | RAM Usage  | Notes                               |
|------------------------------|------------|-------------------------------------|
| Linux OS + background        | ~2.0 GB    | Headless recommended                |
| Ollama llama3.2:3b (q4_k_m)  | ~2.5 GB    | 4-bit quantised via llama.cpp AVX2  |
| all-MiniLM-L6-v2 embedder    | ~0.5 GB    | 384-dim, sentence-transformers      |
| LanceDB (memory-mapped)      | ~1.5 GB    | Disk-backed, streams from SSD       |
| GLiNER NER model             | ~1.0 GB    | CPU-optimised encoder               |
| FastAPI + Streamlit + NX     | ~1.5 GB    | App layer                           |
| Context window + OS cache    | ~4.0 GB    | Dynamic buffer, prevents OOM        |
| **Total**                    | **~13 GB** | 3 GB headroom for safety            |

---

## 📦 Project Structure

```
et_news_platform/
├── app.py                          # Main Streamlit application
├── config.py                       # All configuration (hardware-aware)
├── requirements.txt                # CPU-only Python dependencies
├── setup.sh                        # One-shot Ollama + pip setup
├── smoke_test.py                   # Pre-launch verification suite
│
├── ingestion/
│   ├── rss_fetcher.py              # feedparser + BeautifulSoup DOM extraction
│   └── article_processor.py       # Embeddings, NER, sentiment, chunking
│
├── vector_store/
│   └── lancedb_manager.py         # LanceDB schema, upsert, semantic search
│
├── modules/
│   ├── my_et.py                   # Dual-pass personalised newsroom
│   ├── news_navigator.py          # Map-Reduce RAG briefings + Q&A
│   └── story_arc.py               # Knowledge graph + narrative tracking
│
├── llm/
│   └── ollama_client.py           # Ollama wrapper, prompt templates, streaming
│
└── utils/
    ├── helpers.py                  # Formatting, colour mapping
    └── ingestion_orchestrator.py  # Ingest pipeline + APScheduler loop
```

---

## 🚀 Quick Start

### Prerequisites
- Linux (Ubuntu 20.04+ recommended)
- Python 3.10+
- 16 GB RAM (no VRAM required)
- ~6 GB free disk space

### 1. Clone & Install

```bash
cd et_news_platform
bash setup.sh
```

The setup script:
1. Installs **Ollama** system binary
2. Pulls `llama3.2:3b` (q4_k_m, ~2.2 GB)
3. Installs CPU-only PyTorch + all dependencies
4. Downloads SpaCy `en_core_web_sm` + NLTK data

### 2. Verify Installation

```bash
python smoke_test.py
```

All 20+ tests should pass. If Ollama is not running, AI features show a warning but the app still loads.

### 3. Launch

```bash
# Terminal 1: Start Ollama (if not already running)
ollama serve

# Terminal 2: Launch the platform
streamlit run app.py
```

Visit `http://localhost:8501`

---

## 🧠 Module Deep Dives

### Module 1: My ET — The Personalized Newsroom

**Architecture:** Dual-pass retrieval + persona-based re-ranking

```
Pass 1 (Vector Search):
  query = persona_keywords → embed → cosine similarity → top-50 candidates

Pass 2 (Re-ranking):
  final_score = 0.45 × cosine_sim + 0.30 × temporal_decay + 0.25 × persona_score
  
  temporal_decay = e^(-age_hours / 24)   ← breaking news prioritised
  persona_score  = keyword_match + entity_boost + readability_bonus
```

**Personas:**
| Persona | Focus | Boost Entities | Summary Style |
|---------|-------|----------------|---------------|
| Mutual Fund Investor | NAV, SIP, SEBI, portfolio | SEBI, AMC, NSE, BSE | Data-driven with numbers |
| Startup Founder | Funding, VC, IPO, competitors | DPIIT, Sequoia, Accel | Strategic & action-oriented |
| Student / Learner | Economy, policy, GDP | — | ELI5 with analogies |
| Executive / CXO | M&A, strategy, disruption | — | Executive implications |

---

### Module 2: News Navigator — Interactive Intelligence Briefings

**Architecture:** Advanced RAG with Parent-Child retrieval + Map-Reduce synthesis

```
Step 1: Retrieve top-8 articles via vector search
Step 2: MAP — extract key facts from each article (per-article LLM call)
Step 3: REDUCE — Chain-of-Thought synthesis across all facts

CoT Prompt Structure:
  1. CORE NARRATIVE
  2. SUPPORTING EVIDENCE  
  3. CONTRADICTIONS
  4. IMPLICATIONS (persona-adapted)
  5. WHAT TO WATCH
  + 3 follow-up questions
```

**Memory Safety:** `MAP_REDUCE_CHUNK_ARTICLES = 3` ensures max 3 articles are in LLM context simultaneously, preventing RAM overflow.

---

### Module 3: Story Arc Tracker

**Architecture:** NetworkX knowledge graph + composite arc clustering

```
Graph Schema:
  Nodes: Article (title, sentiment, vector) | Entity (name, mention_count)
  Edges: MENTIONS | MENTIONED_IN | SUBSEQUENT_TO

Arc Association Score:
  S = 0.25 × temporal_closeness + 0.50 × cosine_similarity + 0.25 × jaccard_entity_overlap
  
  Articles cluster into same arc if S ≥ 0.45

Analysis:
  Key Players    → PageRank on entity sub-graph
  Sentiment      → VADER compound score + rolling average
  Contrarians    → Pairs with |sentiment_A - sentiment_B| ≥ 0.4 and opposite signs
  What to Watch  → LLM prediction from arc trajectory
```

---

## ⚙️ Configuration Reference

Key settings in `config.py`:

```python
OLLAMA_MODEL = "llama3.2:3b"          # Swap to "phi4-mini" for faster inference
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384-dim, best speed/quality balance
MAX_CONTEXT_TOKENS = 3500             # Hard cap to prevent RAM thrashing
MAP_REDUCE_CHUNK_ARTICLES = 3         # Articles processed per LLM batch
ARC_ASSOCIATION_THRESHOLD = 0.45      # Lower = more articles per arc
RSS_POLL_INTERVAL_MINUTES = 15        # Background refresh rate
```

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| `Ollama is not running` | Run `ollama serve` in a separate terminal |
| `Model not found` | Run `ollama pull llama3.2:3b` |
| `Out of memory` | Reduce `LLM_CTX_WINDOW` and `MAX_CONTEXT_TOKENS` in config.py |
| `SpaCy model missing` | Run `python -m spacy download en_core_web_sm` |
| `Slow inference` | Normal on CPU — llama3.2:3b averages 10-15 tok/s |
| `Empty feed` | Click **Load Demo** or **Fetch Live** in the sidebar |
| `RSS fetch fails` | ET RSS may be geo-restricted; use **Load Demo** for offline testing |

---

## 📊 Performance Benchmarks (16 GB RAM / Intel i7 CPU)

| Operation | Time |
|-----------|------|
| Embed single article | ~80ms |
| Vector search (1000 docs) | ~15ms |
| Article ingestion (1 article) | ~2s |
| LLM single summary | ~45-90s |
| Navigator briefing (8 articles) | ~4-8 min |
| Story arc analysis | ~2-4 min |

---

## 🏆 Hackathon Alignment

| Requirement | Implementation |
|-------------|----------------|
| Personalized newsroom (mutual fund, startup, student) | Dual-pass ranking with 4 persona profiles |
| Interactive briefings (8 articles → single document) | Map-Reduce RAG with CoT synthesis |
| Follow-up questions | Extracted from briefing + injected into conversation |
| Story arc with key players | NetworkX PageRank centrality |
| Sentiment shifts tracked | VADER + rolling average timeline |
| Contrarian perspectives | Divergence detection (|sent_A - sent_B| ≥ 0.4) |
| "What to watch next" | LLM prediction from arc trajectory |
| CPU-only, 16 GB RAM | q4_k_m quantization, LanceDB disk-mapped, memory guards |
