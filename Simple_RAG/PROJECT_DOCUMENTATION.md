# Simple RAG — Complete Project Documentation

## What Is This?

This is a **Retrieval-Augmented Generation (RAG)** system built for analyzing podcast transcripts. It lets you upload `.srt` or `.txt` transcript files, then ask natural language questions and get answers grounded in the actual content.

The system combines **dense vector search** (semantic similarity), **sparse BM25 keyword search**, and a **cross-encoder reranker** to find the most relevant transcript excerpts before passing them to an LLM to generate an answer.

---

## Directory Structure

```
Simple_RAG/
├── app.py                    # Streamlit web UI
├── rag_pipeline.py           # All RAG logic: parsing, chunking, retrieval
├── requirements.txt          # Python dependencies
├── .env                      # API keys and default config
└── rag_store/                # Persistent data (auto-created)
    ├── chroma/               # ChromaDB vector database files
    │   └── chroma.sqlite3
    ├── rag_state.json         # List of ingested source names
    ├── <hash>_chunks.json    # Cached chunks per ingested source
    └── ...
```

---

## How the System Works — Full Pipeline

### Step 1: Ingestion

When you upload a file, `PodcastRAG.ingest()` in `rag_pipeline.py` runs this sequence:

```
Upload (.srt / .txt)
    ↓
Parse SRT → SRTBlock objects (index, start_time, end_time, text)
    ↓
Clean: remove transcription markers like [Music], (applause), HTML tags
    ↓
Reconstruct: group consecutive SRT blocks into logical sentences/paragraphs
    - Groups blocks until a sentence-ending punctuation (. ! ?) is hit
    ↓
Chunk: split into fixed-token windows using tiktoken (CL100K_BASE encoding)
    - Default 500 tokens per chunk, 80-token overlap
    - Each chunk stores source filename + start/end timestamps
    ↓
Embed chunks using BAAI/bge-small-en-v1.5 (384-dimensional vectors)
    ↓
Upsert into ChromaDB (persistent HNSW vector index on disk)
    ↓
Cache raw chunks to JSON file (keyed by MD5 hash of source name)
    ↓
Build BM25 index in memory over all accumulated chunks
    ↓
Persist source list to rag_state.json for session restore
```

### Step 2: Retrieval

```
User query
    ↓
Dense retrieval:
    - Embed query with BGE prefix ("Represent this question for searching...")
    - Cosine similarity search against ChromaDB → top-K chunks
    ↓
Sparse retrieval:
    - Tokenize query (lowercase, strip punctuation)
    - BM25 keyword scoring across all chunks → top-K results
    ↓
RRF (Reciprocal Rank Fusion):
    - Merge both lists using rank-based scoring
    - Dense weighted 60%, sparse weighted 40%
    - Deduplicated by composite key (source + chunk_id)
    ↓
Cross-encoder rerank:
    - ms-marco-MiniLM-L-6-v2 scores each (query, chunk) pair
    - Selects final top-N most relevant chunks
    ↓
Format context string with source filename + timestamps per chunk
```

### Step 3: Answer Generation

```
Top-N chunks formatted as context
    ↓
System prompt: "Answer using only the transcript excerpts below"
    ↓
Call LLM (OpenAI GPT-4o-mini or Groq Llama 3.3-70b)
    ↓
Return answer + visible source excerpts with timestamps
```

---

## Key Technical Components

### Embedding Model: `BAAI/bge-small-en-v1.5`

- 384-dimensional dense vectors
- Uses prefix prompting: `"Represent this sentence: "` for documents, `"query: "` for queries
- Runs locally via `sentence-transformers` — no API cost for embeddings

### Vector Database: ChromaDB

- Persistent storage in `rag_store/chroma/`
- HNSW (Hierarchical Navigable Small World) indexing for approximate nearest-neighbor search
- Cosine similarity metric
- Skips re-embedding if source is already indexed (dedup check on startup)

### BM25 Sparse Retrieval

- `BM25Okapi` from `rank-bm25`
- Tokenized via regex: lowercase, punctuation stripped
- Rebuilt in memory on session start from cached JSON chunks
- Catches exact keyword matches that dense embeddings can miss (e.g., proper nouns, acronyms)

### RRF Fusion

- Reciprocal Rank Fusion with configurable weights
- Formula: `score(d) = dense_weight / (k + rank_dense + 1) + sparse_weight / (k + rank_sparse + 1)`
- Default: 60% dense, 40% sparse, `k=60`
- Composite key `source_chunkId` prevents cross-transcript chunk ID collisions

### Cross-Encoder Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`

- Takes (query, passage) pairs and produces a fine-grained relevance score
- More accurate than cosine similarity, but can only score pairs (not pre-compute)
- Applied to the top-20 fused candidates to select the final top-N
- Runs fully locally

---

## Data Models

### `SRTBlock`
```python
@dataclass
class SRTBlock:
    index: int
    start_time: str    # "00:01:23.456"
    end_time: str
    text: str
```

### `TextChunk`
```python
@dataclass
class TextChunk:
    chunk_id: int
    text: str
    start_time: str
    end_time: str
    char_start: int
    char_end: int
    source: str         # original filename stem
    token_count: int
```

### Chunk Cache JSON Format
```json
{
  "chunk_id": 0,
  "text": "...",
  "start_time": "00:00:00.190",
  "end_time": "00:00:08.170",
  "char_start": 0,
  "char_end": 2273,
  "source": "Andrej Karpathy: Tesla AI...",
  "token_count": 500
}
```

---

## Configuration

Set in `.env` or adjustable in the Streamlit sidebar:

| Parameter | Default | What it does |
|-----------|---------|-------------|
| `RAG_CHUNK_SIZE` | 500 | Tokens per chunk. Larger = more context per chunk, fewer chunks. |
| `RAG_OVERLAP` | 80 | Token overlap between adjacent chunks. Prevents cutting sentences mid-thought. |
| `RAG_TOP_K` | 20 | How many candidates to pull from dense + sparse retrieval before reranking. |
| `RAG_TOP_N` | 5 | Final chunks kept after reranking. These are passed as context to the LLM. |

---

## UI (app.py)

The Streamlit interface has a single-page layout:

### Chat Interface
- **Chat input**: Send a question, get an answer grounded in the transcripts
- **Source badges**: Each retrieved chunk shows which podcast it came from
- **Expandable excerpts**: Click to view the exact transcript text and timestamps that were used
- **Chat history**: Conversation is preserved within the session

### Sidebar
- LLM provider selector (OpenAI / Groq)
- API key input
- Sliders for RAG parameters (chunk size, overlap, top-k, top-n)
- File upload (`.srt` / `.txt`)
- Ingestion progress and stats per source
- Reset state button

---

## State Persistence

The system survives page refresh:

1. **ChromaDB** — embeddings persist to disk automatically in `rag_store/chroma/`
2. **Chunk JSON files** — one file per source, named `<md5_hash>_chunks.json`
3. **`rag_state.json`** — stores the list of ingested source names so they can be restored on next load

On startup, `load_state()` reads `rag_state.json`, reloads chunk JSONs, rebuilds BM25 in memory, and skips re-embedding sources already in ChromaDB.

---

## LLM Providers

| Provider | Model | Notes |
|----------|-------|-------|
| OpenAI | `gpt-4o-mini` | Default. Requires `OPENAI_API_KEY`. |
| Groq | `llama-3.3-70b-versatile` | Fast inference. Requires `GROQ_API_KEY`. |

---

## Architecture Summary

```
┌──────────────────────────────────────────────────┐
│               STREAMLIT UI (app.py)               │
│  Chat | Upload | Source Badges | Expandable Refs  │
└──────────────────────┬───────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────┐
│          RAG_PIPELINE.PY — PodcastRAG class       │
│                                                    │
│  INGESTION:                                        │
│    SRT Parse → Clean → Chunk → Embed → Index      │
│    → Cache JSON → Build BM25                       │
│                                                    │
│  RETRIEVAL:                                        │
│    Dense (ChromaDB) ──┐                            │
│    Sparse (BM25)   ───┼→ RRF Fusion → Rerank      │
│                       └→ Top-N Chunks              │
│                                                    │
│  GENERATION:                                       │
│    Format context → Call LLM → Return answer      │
└──────────────────────┬───────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────┐
│              STORAGE LAYER                         │
│    ChromaDB (vectors) | JSON (chunks) | JSON (state)│
└──────────────────────────────────────────────────┘
```

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your API key in .env or export it
export OPENAI_API_KEY="sk-..."

# 3. Start the app
streamlit run app.py

# 4. In the browser:
#    - Upload .srt or .txt files in the sidebar
#    - Wait for ingestion (embedding + indexing)
#    - Ask questions in the chat
```

---

## Design Decisions & Trade-offs

| Decision | Why |
|----------|-----|
| BM25 + dense (hybrid retrieval) | BM25 catches exact keywords; embeddings catch semantics. Together they cover more cases than either alone. |
| Cross-encoder reranker | More accurate than cosine similarity for final selection. Slower but only applied to the small top-K shortlist. |
| tiktoken for chunking | Accurate token counting ensures chunks fit consistently in LLM context windows. |
| BGE-small embeddings | Small, fast, high quality. Runs on CPU without significant quality loss. |
| Sliding window overlap | 80-token overlap between chunks prevents important context from being split across chunk boundaries. |
| JSON for chunk cache | Human-readable, inspectable, avoids re-embedding on every restart. |
| Streamlit | Fast to build, single-file deployment, no frontend code required. |

---

## Known Limitations

- **Single user / local only** — no authentication, all state on one machine
- **BM25 is in-memory** — rebuilt from JSON cache on every session start
- **Flat retrieval** — no concept of entity relationships or document-level themes; every query retrieves independent chunks
- **No cross-document reasoning** — chunks from different podcasts are treated as equal, unrelated units with no structural connection
- **Thematic questions are weak** — broad questions like "what are the main themes across all interviews?" retrieve semantically close individual chunks rather than synthesizing a structured view
- **No streaming** — LLM responses wait for full completion before displaying
- **`.env` file** — API key stored in plain text; don't commit to git
