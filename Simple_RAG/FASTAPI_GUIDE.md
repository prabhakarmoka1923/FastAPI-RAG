# Podcast RAG — FastAPI Backend Guide

A complete walkthrough of every file added in this phase, covering **what** each piece does, **why** it exists, and **which FastAPI concept** it demonstrates.

---

## Table of Contents

1. [Project layout](#1-project-layout)
2. [How to run](#2-how-to-run)
3. [How to test](#3-how-to-test)
4. [File-by-file breakdown](#4-file-by-file-breakdown)
   - [config.py — pydantic-settings](#41-appconfigpy--pydantic-settings)
   - [dependencies.py — Dependency Injection](#42-appdependenciespy--dependency-injection)
   - [main.py — Lifespan & App creation](#43-appmainpy--lifespan--app-creation)
   - [services/rag_service.py — async wrappers](#44-appservicesrag_servicepy--async-wrappers)
   - [schemas/ — Request & Response models](#45-appschemas--request--response-models)
   - [routers/ingest.py — POST /ingest](#46-approutersingestpy--post-ingest)
   - [routers/query.py — POST /ask](#47-approutersquerypy--post-ask)
   - [routers/health.py — GET /health](#48-approutershealthpy--get-health)
   - [tests/ — TestClient + mocks](#49-apptests--testclient--mocks)
5. [Swagger UI usage](#5-swagger-ui-usage)
6. [Postman usage](#6-postman-usage)
7. [Key FastAPI concepts index](#7-key-fastapi-concepts-index)
8. [Data flow diagram](#8-data-flow-diagram)

---

## 1. Project layout

```
Simple_RAG/
├── rag_pipeline.py          # existing — do NOT modify
├── app.py                   # existing Streamlit UI — do NOT modify
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app + lifespan
│   ├── config.py            # all settings via pydantic-settings
│   ├── dependencies.py      # shared DI functions
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── ingest.py        # POST /ingest
│   │   ├── query.py         # POST /ask
│   │   └── health.py        # GET /health
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── ingest.py        # IngestResponse
│   │   └── query.py         # AskRequest, AskResponse, SourceChunk
│   ├── services/
│   │   ├── __init__.py
│   │   └── rag_service.py   # async wrappers (run_in_executor)
│   └── tests/
│       ├── __init__.py
│       ├── conftest.py      # TestClient + MockRAG fixtures
│       ├── test_ingest.py
│       └── test_query.py
├── .env.example             # copy to .env, edit if needed
├── requirements_api.txt     # FastAPI-only deps (no duplication)
└── postman_collection.json  # ready-to-import Postman collection
```

---

## 2. How to run

### Install dependencies

```bash
conda activate ragpcnow
pip install -r requirements_api.txt
```

### Start the server

```bash
# from the Simple_RAG/ directory
uvicorn app.main:app --reload
```

| URL | Purpose |
|-----|---------|
| `http://localhost:8000/docs` | Swagger UI (interactive) |
| `http://localhost:8000/redoc` | ReDoc (read-only docs) |
| `http://localhost:8000/health` | Quick liveness check |

### Environment variables (optional)

Copy `.env.example` to `.env` and edit if you want non-default values.  
The server works perfectly without a `.env` file using the built-in defaults.

---

## 3. How to test

```bash
conda activate ragpcnow
cd "/Users/soumil/Desktop/FastAPI for GenAI/Simple_RAG"
pytest app/tests/ -v
```

All tests run without loading any ML model — the RAG pipeline is mocked.

---

## 4. File-by-file breakdown

### 4.1 `app/config.py` — pydantic-settings

**Concept: centralised, validated configuration**

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    RAG_PERSIST_DIR: str = "./rag_store"
    API_KEY: str = "dev-key-123"
    ...
```

- `pydantic-settings` reads values from environment variables **and** a `.env` file automatically.
- Types are validated at startup — if you set `RAG_CHUNK_SIZE=abc` you get an error immediately, not a runtime crash later.
- `@lru_cache` on `get_settings()` means the file is only read once, not on every request.

**Why not `os.environ`?** — `os.environ` gives you raw strings with no validation and no defaults. pydantic-settings gives you typed, documented, testable config.

---

### 4.2 `app/dependencies.py` — Dependency Injection

**Concept: FastAPI's `Depends()` system**

```python
def get_rag(request: Request):
    return request.app.state.rag

def verify_api_key(x_api_key: Optional[str] = Header(None)):
    if x_api_key is None or x_api_key != settings.API_KEY:
        raise HTTPException(status_code=401, ...)
```

- `Depends(get_rag)` injects the shared `PodcastRAG` instance into any route that declares it — no global variables needed.
- `Depends(verify_api_key)` runs the auth check before the route body executes. If the key is wrong, the request is rejected with 401.
- The `Header(None)` declaration makes the `X-API-Key` header show up in Swagger UI automatically.

**Why `Optional[str] = Header(None)` not `str = Header(...)`?**  
Using `Header(...)` (required) means FastAPI returns **422 Unprocessable Entity** before your function even runs when the header is absent. Using `Optional` + `None` default lets the request reach `verify_api_key`, where you can return a proper **401**. This is the correct pattern for custom auth headers.

**Why 401 not 403?**  
- 401 = *unauthenticated* (the client didn't prove who they are)  
- 403 = *unauthorised* (client is known, but lacks permission)  
A missing/wrong API key is an authentication failure → 401.

---

### 4.3 `app/main.py` — Lifespan & App creation

**Concept: lifespan context manager (replaces deprecated `on_event`)**

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- startup ---
    rag = PodcastRAG(...)
    await loop.run_in_executor(executor, rag.load_state)
    app.state.rag = rag
    app.state.executor = executor

    yield   # <-- server is alive here

    # --- shutdown ---
    executor.shutdown(wait=True)
```

- Everything **before** `yield` runs at startup. Everything **after** runs at shutdown.
- `app.state` is a simple bag for storing application-wide objects (rag, executor).
- Using `ThreadPoolExecutor(max_workers=1)` means RAG operations run one at a time — correct because ChromaDB and BM25 are not thread-safe for writes.

---

### 4.4 `app/services/rag_service.py` — async wrappers

**Concept: `run_in_executor` — offloading blocking code to a thread pool**

```python
async def ingest_file(rag, executor, tmp_path: str) -> dict:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, rag.ingest, tmp_path)
```

**Why does this matter?**  
FastAPI runs on an `asyncio` event loop. If you call a slow, CPU-bound function directly in an `async def` route, you **block the entire event loop** — no other requests can be served until it finishes.

`run_in_executor` moves the blocking call to a thread pool and returns control to the event loop immediately. The route handler `await`s the result when it's ready.

**Rule:** every `rag.*` call goes through this service layer. Route handlers never call `rag.*` directly.

---

### 4.5 `app/schemas/` — Request & Response models

**Concept: Pydantic models for request validation and response serialisation**

```python
class AskRequest(BaseModel):
    query: str
    top_k: int = 5

class SourceChunk(BaseModel):
    text: str
    source: str
    start_time: str
    end_time: str
    score: float
```

- **Request validation:** if a client sends `{"top_k": "banana"}` FastAPI returns a 422 automatically — you write zero validation code.
- **Response serialisation:** `response_model=AskResponse` on the route tells FastAPI to filter the return value through the schema. Any extra fields are stripped; missing required fields raise an error.
- **`json_schema_extra`:** adds an example to the schema so Swagger UI pre-fills the request body for you.

---

### 4.6 `app/routers/ingest.py` — POST /ingest

**Concept: file uploads with `UploadFile`, temp file handling, `APIRouter`**

```python
@router.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest(file: UploadFile, ...):
```

Key details:
- `UploadFile` is FastAPI's async file upload type — `python-multipart` must be installed for it to work.
- The file is written to `/tmp/<uuid>_<filename>` because `rag.ingest()` expects a **file path string**, not a file object.
- `finally` block ensures the temp file is deleted even if ingestion throws an exception — important for disk hygiene.
- Extension check returns **400 Bad Request** (not 422) because the file type is a domain rule, not a schema validation rule.

---

### 4.7 `app/routers/query.py` — POST /ask

**Concept: JSON request body, response model, chained async calls**

```python
@router.post("/ask", response_model=AskResponse, tags=["Query"])
async def ask(request: AskRequest, ...):
    hits = await rag_service.retrieve_chunks(rag, executor, request.query)
    context = await rag_service.build_context(rag, executor, hits)
    ...
```

- Accepts a JSON body (`AskRequest`) — FastAPI knows it's JSON because it's a Pydantic model parameter (not `UploadFile` or `Form`).
- Two sequential `await` calls: first retrieve, then format. They are sequential because `build_context` depends on the result of `retrieve_chunks`.
- Maps the raw `(text, meta, score)` tuples from the pipeline into `SourceChunk` schema objects.

---

### 4.8 `app/routers/health.py` — GET /health

**Concept: public endpoint, accessing app state from a route**

```python
@router.get("/health", tags=["System"])
async def health(request: Request):
    rag = request.app.state.rag
    return {"status": "ok", "env": settings.ENV, "sources_loaded": len(rag._ingested_sources)}
```

- No `Depends(verify_api_key)` — health checks must always work, even in monitoring systems that don't know the API key.
- `request.app.state.rag` shows how to access app-level state from any route.
- Returns a plain dict — FastAPI serialises it to JSON automatically (no schema needed for simple responses).

---

### 4.9 `app/tests/` — TestClient + mocks

**Concept: `TestClient`, dependency overrides, mocking ML models**

```python
# In conftest.py
app.dependency_overrides[get_rag] = lambda: mock_rag_instance
app.dependency_overrides[verify_api_key] = lambda: None
```

- `TestClient` (from `httpx`) runs the ASGI app in-process — no real HTTP server needed.
- `dependency_overrides` replaces any dependency with a different function at test time. This is the canonical FastAPI pattern for swapping in mocks.
- `MockRAG` returns hardcoded data without loading any model — tests run in milliseconds.
- `verify_api_key` is overridden to a no-op lambda so tests that want to check 401 can opt in by **not** using the override.

---

## 5. Swagger UI usage

1. Start the server: `uvicorn app.main:app --reload`
2. Open `http://localhost:8000/docs`
3. Click the **Authorize** button (top right, lock icon)
4. Enter `dev-key-123` as the value
5. Click **Authorize**, then **Close**
6. All three endpoints are now unlocked — click any, hit **Try it out**, fill the form, **Execute**

The `/ingest` endpoint shows a file picker in Swagger UI because it uses `UploadFile`.  
The `/ask` endpoint shows a pre-filled JSON body from `json_schema_extra`.

---

## 6. Postman usage

1. Open Postman → **Import** → select `postman_collection.json`
2. The collection appears with three pre-configured requests
3. For **POST /ingest**: go to Body → form-data → click the file field → select a `.srt` or `.txt` file
4. For **POST /ask**: go to Body → the JSON example is already filled in
5. Send — the `X-API-Key: dev-key-123` header is pre-set on both authenticated requests

---

## 7. Key FastAPI concepts index

| Concept | Where demonstrated |
|---|---|
| `pydantic-settings` for config | `app/config.py` |
| `Depends()` dependency injection | `app/dependencies.py`, all routers |
| `Header(...)` parameter | `app/dependencies.py` → `verify_api_key` |
| `HTTPException` | `app/dependencies.py`, `app/routers/ingest.py` |
| `@asynccontextmanager` lifespan | `app/main.py` |
| `app.state` for shared objects | `app/main.py`, `app/routers/health.py` |
| `run_in_executor` for blocking code | `app/services/rag_service.py` |
| `UploadFile` for file uploads | `app/routers/ingest.py` |
| `response_model` | All routers |
| `tags=[...]` for Swagger grouping | All routers |
| `json_schema_extra` for examples | `app/schemas/query.py` |
| `APIRouter` + `include_router` | All routers + `app/main.py` |
| `dependency_overrides` in tests | `app/tests/conftest.py` |
| `TestClient` | `app/tests/conftest.py` |

---

## 8. Data flow diagram

```
Client (Swagger UI / Postman / Streamlit)
        │
        ▼
  FastAPI route handler  (async def)
        │
        │  Depends()
        ├─────────────── verify_api_key  → 401 if bad key
        ├─────────────── get_rag         → app.state.rag
        └─────────────── get_executor    → app.state.executor
        │
        ▼
  rag_service (async wrapper)
        │
        │  run_in_executor
        ▼
  ThreadPoolExecutor (1 worker)
        │
        ▼
  PodcastRAG  (synchronous, CPU-bound)
   ├── .ingest(file_path)   → ChromaDB + BM25 index
   ├── .retrieve(query)     → dense + sparse + rerank
   └── .format_context()   → LLM-ready string
        │
        ▼
  Response model (Pydantic)
        │
        ▼
  JSON response to client
```
