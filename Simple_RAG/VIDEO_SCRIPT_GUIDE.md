# What We Added to Simple RAG — Video Explanation Guide

> **How to use this doc:** Read each section top to bottom before filming. Every section has a "What to say" block you can speak naturally from, followed by the actual code so you can reference it on screen.

---

## The Big Picture — What Problem Are We Solving?

Before we added anything, Simple RAG was a **Streamlit app**. It worked, but only one person could use it, only through a browser, and only if they ran it locally on their machine.

**The question we asked:** *What if another developer, another app, or a mobile frontend wanted to use our RAG system?*

The answer is a **REST API**. We wrapped the exact same RAG pipeline inside a FastAPI backend. Now anyone can send an HTTP request and get answers back — Swagger UI, Postman, another Python script, a mobile app — it doesn't matter.

**Nothing in `rag_pipeline.py` was touched.** We just built a clean layer on top of it.

---

## The Folder Structure We Added

```
app/
  main.py              ← the front door of the API
  config.py            ← all settings in one place
  dependencies.py      ← shared building blocks for routes
  routers/
    ingest.py          ← POST /ingest
    query.py           ← POST /ask
    health.py          ← GET /health
  schemas/
    ingest.py          ← shape of the response for /ingest
    query.py           ← shape of request and response for /ask
  services/
    rag_service.py     ← the bridge between async FastAPI and sync RAG
  tests/
    conftest.py        ← test setup and mocks
    test_ingest.py     ← tests for /ingest
    test_query.py      ← tests for /ask
```

Think of it like a restaurant:
- `config.py` = the restaurant's rulebook (opening hours, prices, policies)
- `main.py` = the front entrance
- `routers/` = the different counters (ingest counter, query counter, health counter)
- `schemas/` = the menu — it defines exactly what you can order and what you'll receive
- `services/` = the kitchen — the actual cooking happens here
- `dependencies.py` = the staff ID check at the door

---

## File 1 — `app/config.py` — Centralised Settings

### What it does

Every single configurable value in the entire application lives in this one file. Nothing is hardcoded anywhere else.

```python
class Settings(BaseSettings):
    RAG_PERSIST_DIR: str = "./rag_store"
    RAG_CHUNK_SIZE: int = 500
    RAG_OVERLAP: int = 80
    RAG_TOP_K: int = 20
    RAG_TOP_N: int = 5
    API_KEY: str = "dev-key-123"
    ENV: str = "development"
```

### Why this matters

Before this, the RAG pipeline read settings directly from `os.getenv()` scattered throughout the code. That means:
- If you want to change `chunk_size`, you have to hunt through multiple files
- There's no validation — if someone types `RAG_CHUNK_SIZE=abc`, it silently breaks later

With `pydantic-settings`:
- All settings have **types** — if you pass a string where an int is expected, the app refuses to start and tells you exactly what's wrong
- All settings have **defaults** — the app works out of the box with zero configuration
- You can override any value by creating a `.env` file or setting environment variables — no code changes needed

### The `@lru_cache` trick

```python
@lru_cache
def get_settings() -> Settings:
    return Settings()
```

The `.env` file is read **once** at startup, not on every single request. `lru_cache` caches the result — subsequent calls just return the same object instantly.

---

## File 2 — `app/dependencies.py` — Shared Building Blocks

### What it does

This file contains three functions that every route can use. Instead of writing the same logic in every route handler, you declare it once here and FastAPI injects it automatically.

```python
def get_rag(request: Request):
    return request.app.state.rag

def get_executor(request: Request):
    return request.app.state.executor

def verify_api_key(
    x_api_key: Optional[str] = Header(None, description="API key for auth"),
    settings: Settings = Depends(get_settings),
):
    if x_api_key is None or x_api_key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
```
 
### Why three separate functions?

**`get_rag`** — The `PodcastRAG` object is expensive to create (it sets up ChromaDB and builds BM25 indexes). We create it **once** at startup and store it in `app.state`. This function just fetches it from there. Every route that needs the RAG engine declares `rag = Depends(get_rag)` and gets the same shared instance.

**`get_executor`** — Same idea for the thread pool. One pool, shared across all requests.

**`verify_api_key`** — This is our security guard. Any route that declares `_ = Depends(verify_api_key)` will automatically check for a valid `X-API-Key` header before executing. If the key is wrong or missing, the request is rejected with a **401 Unauthorized** before a single line of your route handler runs.

### The Optional[str] = Header(None) pattern — important gotcha

You might wonder: why not just write `x_api_key: str = Header(...)` (required)?

If you do that, when the header is missing FastAPI returns **422 Unprocessable Entity** — a validation error — before our function even runs. We lose control of the response code.

Using `Optional[str] = Header(None)` means: *"this header is optional at the framework level, but I'll enforce it myself."* This gives us the proper **401** response that clients expect for auth failures.

### The `Depends()` system — FastAPI's superpower

This is one of FastAPI's most powerful features. Instead of this in every route:

```python
# Without Depends — repeated in every single route
async def ingest(...):
    rag = request.app.state.rag          # copy-pasted
    if api_key != settings.API_KEY:      # copy-pasted
        raise HTTPException(...)         # copy-pasted
```

You write this:

```python
# With Depends — declared once, injected everywhere
async def ingest(
    rag = Depends(get_rag),
    _ = Depends(verify_api_key)
):
```

FastAPI calls those dependency functions, passes the results in, and runs `verify_api_key` before your handler even starts. If auth fails, your code never runs.

---

## File 3 — `app/main.py` — The Front Door

### What it does

This file creates the FastAPI app and defines what happens at **startup** and **shutdown**.

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP ---
    rag = PodcastRAG(
        persist_dir=settings.RAG_PERSIST_DIR,
        chunk_size=settings.RAG_CHUNK_SIZE,
        ...
    )
    await loop.run_in_executor(executor, rag.load_state)
    app.state.rag = rag
    app.state.executor = executor

    yield   # ← server is alive and taking requests here

    # --- SHUTDOWN ---
    executor.shutdown(wait=True)
```

### Why the `lifespan` pattern?

FastAPI used to have `@app.on_event("startup")` and `@app.on_event("shutdown")` decorators. These are now **deprecated**. The modern way is `lifespan` — a single async context manager that splits on `yield`. Everything before the yield runs at startup, everything after runs at shutdown.

This is important because:
- `PodcastRAG` initialization + `load_state` takes a few seconds (it rebuilds BM25 from disk)
- You want that to happen **once** when the server starts, not on every request
- At shutdown you want to cleanly drain the thread pool

### `app.state` — the app's memory

`app.state.rag` and `app.state.executor` are how we share objects across the entire application lifetime. Think of it as the app's global backpack — you put things in at startup, and any route can reach in and grab them via the `get_rag` / `get_executor` dependency functions.

### The `ThreadPoolExecutor(max_workers=1)`

We set `max_workers=1` intentionally. The RAG pipeline is not thread-safe for writes (ChromaDB, BM25 rebuild). With one worker, all RAG operations are queued and executed one at a time — no two operations can corrupt each other's state.

---

## File 4 — `app/services/rag_service.py` — The Bridge

### The core problem this solves

FastAPI is **asynchronous** — it runs on an `asyncio` event loop. The RAG pipeline (`rag_pipeline.py`) is **synchronous and CPU-bound** — it does heavy computation, embedding, and disk I/O.

If you call a slow synchronous function directly inside an `async def` route handler, you **freeze the entire event loop**. No other requests can be handled until your slow function finishes. With something like embedding 500 chunks, that could be 10-30 seconds of total freeze.

`run_in_executor` is the solution:

```python
async def ingest_file(rag, executor, tmp_path: str) -> dict:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, rag.ingest, tmp_path)
```

### What `run_in_executor` actually does

1. Takes the blocking `rag.ingest` function and hands it off to the `ThreadPoolExecutor`
2. Returns control to the event loop immediately with `await`
3. The event loop is now free to handle other incoming requests
4. When `rag.ingest` finishes in the thread, the `await` resumes and returns the result

It's like a restaurant taking your order and giving you a buzzer. They don't make you stand at the counter blocking everyone else — you sit down, and the buzzer goes off when your food is ready.

### The golden rule

**Every `rag.*` call in this entire project goes through `rag_service.py`.** Route handlers never call `rag.ingest()` or `rag.retrieve()` directly. This is enforced by design — you'd have to deliberately bypass `rag_service` to break the rule.

---

## File 5 — `app/schemas/` — The Contract

### What schemas do

Schemas define the **exact shape** of every request coming in and every response going out. They're written as Pydantic models.

**`IngestResponse`** — what `/ingest` sends back:
```python
class IngestResponse(BaseModel):
    source: str          # name of the transcript file
    chunks: int          # number of chunks created
    cached: bool         # was it loaded from disk cache?
    total_chunks: int    # total across all ingested files
    total_sources: int   # how many files are loaded right now
```

**`AskRequest`** — what `/ask` expects to receive:
```python
class AskRequest(BaseModel):
    query: str
    top_k: int = 5
```

**`AskResponse`** — what `/ask` sends back:
```python
class AskResponse(BaseModel):
    query: str
    sources: list[SourceChunk]   # the actual transcript excerpts
    context: str                 # pre-formatted string for an LLM prompt
    total_hits: int
```

**`SourceChunk`** — one individual result:
```python
class SourceChunk(BaseModel):
    text: str          # the transcript excerpt
    source: str        # which file it came from
    start_time: str    # timestamp in the podcast
    end_time: str      # timestamp in the podcast
    score: float       # relevance score from the reranker
```

### Why schemas matter — two reasons

**Reason 1 — Input validation for free.** If a client sends `{"top_k": "five"}` instead of `{"top_k": 5}`, FastAPI returns a detailed 422 error automatically. You write zero validation code.

**Reason 2 — `response_model` filtering.** When you declare `response_model=AskResponse` on a route, FastAPI runs the return value through the schema. Extra fields are stripped, missing required fields raise an error at the Python level — not silently returned as null. This prevents accidentally leaking internal data.

### The example in the schema

```python
model_config = ConfigDict(
    json_schema_extra={
        "example": {
            "query": "What did the guest say about AI?",
            "top_k": 5
        }
    }
)
```

This is what pre-fills the request body in Swagger UI. Without it, users see empty fields. With it, they see a working example they can just hit Execute on.

---

## File 6 — `app/routers/ingest.py` — POST /ingest

### What this endpoint does

Accepts a transcript file, saves it temporarily, runs it through the RAG pipeline, then cleans up.

```python
@router.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest(
    file: UploadFile,
    rag=Depends(get_rag),
    executor=Depends(get_executor),
    _=Depends(verify_api_key),
):
```

### The `UploadFile` type

`UploadFile` is FastAPI's built-in file upload type. It handles multipart form data automatically. The file is streamed in memory-efficiently. This is why `python-multipart` is in the requirements — FastAPI needs it to parse `multipart/form-data`.

### The temp file pattern

Here's a subtle but important point. Our `rag.ingest()` function expects a **file path string**, not a file object. So we have to:

1. Read the uploaded bytes from `UploadFile`
2. Write them to a temporary file at `/tmp/<uuid>_<filename>`
3. Pass that path to `rag.ingest()`
4. Delete the temp file

```python
tmp_path = f"/tmp/{uuid.uuid4().hex}_{file.filename}"
try:
    contents = await file.read()
    with open(tmp_path, "wb") as f:
        f.write(contents)
    stats = await rag_service.ingest_file(rag, executor, tmp_path)
finally:
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
```

The `uuid4()` prefix ensures two simultaneous uploads of `episode1.txt` don't collide on disk. The `finally` block is critical — it runs **even if ingestion throws an exception**. If we didn't have this, failed uploads would litter `/tmp` with orphaned files.

### File extension validation

```python
_, ext = os.path.splitext(file.filename or "")
if ext.lower() not in {".srt", ".txt"}:
    raise HTTPException(status_code=400, detail=f"Unsupported file type '{ext}'...")
```

This is a **domain rule** (only transcripts make sense for a podcast RAG), so it returns **400 Bad Request** — not a 422 schema error. The distinction matters: 422 = "your JSON was malformed", 400 = "your request was structurally valid but semantically wrong."

---

## File 7 — `app/routers/query.py` — POST /ask

### What this endpoint does

Takes a natural-language question, runs it through hybrid retrieval and reranking, and returns the top transcript excerpts plus a pre-formatted context string.

```python
@router.post("/ask", response_model=AskResponse, tags=["Query"])
async def ask(request: AskRequest, ...):
    hits = await rag_service.retrieve_chunks(rag, executor, request.query)

    if not hits:
        return AskResponse(
            query=request.query,
            sources=[],
            context="No relevant excerpts found.",
            total_hits=0,
        )

    hits = hits[:request.top_k]
    context = await rag_service.build_context(rag, executor, hits)

    sources = [
        SourceChunk(
            text=text,
            source=meta.get("source", "unknown"),
            start_time=meta.get("start_time", ""),
            end_time=meta.get("end_time", ""),
            score=round(score, 4),
        )
        for text, meta, score in hits
    ]
```

### JSON body vs. form data

Notice there's no `UploadFile` here — just `request: AskRequest`. FastAPI sees a Pydantic model parameter and automatically knows to expect a **JSON body**. You don't have to say `Body(...)` or anything. The type annotation is the entire declaration.

### This endpoint does retrieval only — not generation

The `context` field in the response is a pre-formatted string that looks like this:

```
[Excerpt 1 | Source: lex_fridman_ep123 | Timestamp: 00:12:34 → 00:13:01]
The guest explained that transformers work by computing attention scores...

---

[Excerpt 2 | Source: lex_fridman_ep123 | Timestamp: 00:45:12 → 00:45:58]
When asked about AGI timelines, he said...
```

You paste that string into a prompt alongside the user's question and send it to GPT-4, Claude, or any LLM. The API's job is **retrieval** — finding the right excerpts. The LLM's job is **generation** — synthesizing an answer. Clean separation.

---

## File 8 — `app/routers/health.py` — GET /health

### What this endpoint does

Returns a simple status check. No auth required.

```python
@router.get("/health", tags=["System"])
async def health(request: Request):
    rag = request.app.state.rag
    return {
        "status": "ok",
        "env": settings.ENV,
        "sources_loaded": len(rag._ingested_sources),
    }
```

When you hit this:
```json
{
  "status": "ok",
  "env": "development",
  "sources_loaded": 7
}
```

### Why no auth here?

Health checks need to work always — even in monitoring systems, load balancers, and uptime checkers that don't know your API key. If the health endpoint required auth, a broken key rotation could make your monitoring system think the API is down.

The `sources_loaded: 7` tells you how many transcripts were restored from the disk cache at startup — useful for verifying state without having to ingest anything.

---

## File 9 — `app/tests/` — How We Test Without Loading Any Model

### The problem with testing ML

Loading `BAAI/bge-small-en-v1.5` and `cross-encoder/ms-marco-MiniLM-L-6-v2` takes 10-30 seconds and requires a GPU/CPU with significant memory. Running that on every test run would be painful. And in CI/CD, you might not even have the models downloaded.

### The solution — `MockRAG`

```python
class MockRAG:
    def ingest(self, file_path: str) -> dict:
        return {"source": "test", "chunks": 42, "cached": False, ...}

    def retrieve(self, query: str):
        return [("Sample transcript text.", {"source": "test_podcast", ...}, 0.93)]

    def format_context(self, hits) -> str:
        return "[Excerpt 1 | Source: test_podcast]..."
```

`MockRAG` returns hardcoded fake data instantly. Same interface, zero ML.

### `dependency_overrides` — swapping out dependencies in tests

```python
app.dependency_overrides[get_rag] = lambda: mock_rag_instance
app.dependency_overrides[get_executor] = lambda: None
```

This is FastAPI's built-in testing pattern. You tell the app: *"when any route asks for `get_rag`, give them `mock_rag_instance` instead."* The routes don't know the difference — they just receive something that behaves like a `PodcastRAG`.

We do **not** override `verify_api_key` — we want the real auth check to run in tests. That's why the 401 tests work: they just don't send the header.

### `TestClient` — no real HTTP server needed

```python
with TestClient(app) as c:
    response = c.post("/ingest", files={"file": ...}, headers={"X-API-Key": "dev-key-123"})
    assert response.status_code == 200
```

`TestClient` (from `httpx`) runs the entire ASGI app in-process. There's no real network socket, no port binding, no server startup — it's just function calls that look like HTTP. Tests run in milliseconds.

### The 8 tests and what they prove

| Test | What it proves |
|------|---------------|
| `test_ingest_valid_txt` | `.txt` files are accepted and return the correct response shape |
| `test_ingest_valid_srt` | `.srt` files are also accepted |
| `test_ingest_invalid_extension` | `.pdf` returns 400, not 500 |
| `test_ingest_missing_api_key` | Missing header returns 401 |
| `test_ask_valid_query` | Query returns correct `AskResponse` shape with `SourceChunk` list |
| `test_ask_missing_api_key` | Missing header returns 401 |
| `test_ask_empty_query` | Missing `query` field in JSON returns 422 |
| `test_ask_default_top_k` | `top_k` is optional — omitting it still works |

---

## The Full Request Lifecycle — What Happens When You Call POST /ask

Let's trace one request all the way through:

```
1.  Client sends:
    POST /ask
    X-API-Key: dev-key-123
    {"query": "What did the guest say about AI?", "top_k": 5}

2.  FastAPI receives the request.

3.  Dependency resolution runs BEFORE the handler:
    - get_settings()     → returns cached Settings object
    - get_rag()          → returns app.state.rag (the PodcastRAG instance)
    - get_executor()     → returns app.state.executor (the thread pool)
    - verify_api_key()   → "dev-key-123" == settings.API_KEY ✓ (passes)

4.  FastAPI validates the JSON body against AskRequest:
    - query = "What did the guest say about AI?" ✓
    - top_k = 5 ✓

5.  The ask() handler runs:
    hits = await rag_service.retrieve_chunks(rag, executor, query)

6.  rag_service.retrieve_chunks() calls:
    loop.run_in_executor(executor, rag.retrieve, query)
    → rag.retrieve() runs in the ThreadPoolExecutor thread
    → event loop is FREE to handle other requests during this time
    → rag.retrieve() does:
         dense search (ChromaDB cosine similarity)
         + sparse search (BM25)
         + RRF fusion
         + cross-encoder reranking
    → returns List[Tuple[str, dict, float]]

7.  Back in ask():
    hits = hits[:5]   (trim to top_k)
    context = await rag_service.build_context(rag, executor, hits)
    → rag.format_context() runs in thread pool
    → returns formatted string

8.  sources list is built by mapping each (text, meta, score) tuple
    into a SourceChunk Pydantic object.

9.  FastAPI serialises AskResponse to JSON via the response_model.

10. Client receives:
    {
      "query": "What did the guest say about AI?",
      "sources": [
        {
          "text": "The guest explained...",
          "source": "lex_fridman_ep123",
          "start_time": "00:12:34.000",
          "end_time": "00:13:01.000",
          "score": 0.9321
        },
        ...
      ],
      "context": "[Excerpt 1 | Source: lex_fridman_ep123 | ...]...",
      "total_hits": 5
    }
```

---

## What We Did NOT Change

This is worth saying clearly in the video:

- `rag_pipeline.py` — **not touched**. Zero changes to the core RAG logic.
- `app.py` (Streamlit) — **not touched**. The Streamlit app still works exactly as before.

We only added a new layer **on top**. The RAG system gained an API without losing anything it already had.

---

## Summary — The 6 FastAPI Concepts This Project Teaches

| Concept | Where |
|---------|-------|
| `pydantic-settings` for typed configuration | `config.py` |
| Lifespan for startup/shutdown logic | `main.py` |
| `Depends()` for reusable, injectable logic | `dependencies.py`, all routers |
| `run_in_executor` for blocking code in async apps | `services/rag_service.py` |
| `UploadFile` + temp file pattern for file uploads | `routers/ingest.py` |
| `dependency_overrides` + `TestClient` for fast testing | `tests/conftest.py` |
