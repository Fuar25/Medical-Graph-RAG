# medgraphrag — Design Document

> **Audience**: AI agents and developers onboarding to this codebase. Read this before making changes.

## Purpose

`medgraphrag` is a three-layer medical knowledge graph construction pipeline. It reads medical documents (PDFs, plain text), extracts structured entities and relationships via LLM, stores them in Neo4j, and links subgraphs across three semantic layers using embedding-based cosine similarity.

---

## Architecture

### Pipeline Stages

```
Documents
   │
   ▼
ingestion/loader.py          load_high()
   │
   ▼
ingestion/chunking/          split_text_chunks()  ← default
   │                         run_chunk()          ← --grained_chunk flag
   ▼
extraction/entity_extractor.py
   │  LLM call (async)       _extract_entities_from_chunk()
   │  parse response         parse_extraction_response()
   │  write to Neo4j         _write_to_neo4j()
   │  embed entities         embedding/local.py
   ▼
graph/operations.py
   │  merge similar nodes    merge_similar_nodes()   ← --ingraphmerge
   │  create summary node    add_sum()
   ▼
Neo4j subgraph (tagged with gid)

[after all layers imported]
   │
   ▼
pipeline/three_layer.py      create_trinity_links()  ← --trinity
   │  cross-layer edges      ref_link()
   ▼
Final three-layer graph

[optional QA API]
   │
   ▼
qa.py                       GraphQA.answer()
   │  embed question        embedding/local.py
   │  retrieve evidence     Neo4j + gds.similarity.cosine
   │  generate answer       llm/client.py
```

### Module Responsibilities

| Module | Responsibility | Key exports |
|--------|---------------|-------------|
| `llm/config.py` | API key, base URL, model name resolution; supports GLM and OpenAI-compatible backends | `get_chat_model()`, `openai_client_kwargs()` |
| `llm/client.py` | Async LLM call with exponential-backoff retry (8 attempts for recoverable failures) | `openai_complete_if_cache()` |
| `llm/summarizer.py` | Splits text into token-limited chunks, calls LLM, joins responses | `process_chunks()` |
| `embedding/local.py` | Singleton Qwen3-Embedding-8B loaded on `cuda:0`; 1024-dim, L2-normalised | `embed(texts)` |
| `graph/store.py` | Thin Neo4j driver wrapper | `Neo4jGraph.query()` |
| `graph/operations.py` | All Neo4j graph operations that touch embeddings or structure | `merge_similar_nodes()`, `ref_link()`, `add_sum()`, `str_uuid()` |
| `ingestion/loader.py` | Reads `.pdf` (pypdf) or `.txt`; returns single string | `load_high()` |
| `ingestion/chunking/basic.py` | Fixed-size character-level chunking with overlap | `split_text_chunks()` |
| `ingestion/chunking/proposition.py` | LangChain proposition extraction → AgenticChunker | `run_chunk()` |
| `ingestion/chunking/agentic.py` | LLM-guided chunk grouping by semantic topic | `AgenticChunker` |
| `extraction/entity_extractor.py` | Core pipeline: chunk → LLM → parse → embed → Neo4j | `create_metagraph_with_description()` |
| `pipeline/three_layer.py` | Orchestrates multi-layer import and cross-layer linking | `ThreeLayerImporter` |
| `qa.py` | Python Graph RAG QA API: question embedding, node retrieval, one-hop evidence expansion, grounded answer generation | `GraphQA`, `QAResult`, `EvidenceItem` |

---

## Data Model (Neo4j)

### Node Labels

Node labels come directly from `DEFAULT_ENTITY_TYPES` in `extraction/entity_extractor.py`:

```
药品, 活性成分, 禁忌, 用法用量, 给药途径, 不良反应,
警告注意事项, 药物相互作用, 特殊人群, 检查监测,
解剖部位, 疾病, 症状, 剂量规格
```

Plus `Summary` nodes created by `add_sum()`.

### Node Properties

| Property | Type | Description |
|----------|------|-------------|
| `id` | string | Entity name (used as unique key within a gid) |
| `gid` | string | UUID identifying which document subgraph this node belongs to |
| `description` | string | LLM-generated description in Chinese |
| `embedding` | float[] | 1024-dim Qwen3 embedding of `"{id}: {description}"` |
| `source` | string | Always `"medgraphrag"` |

### Relationship Types

Defined in `ALLOWED_RELATION_TYPES` in `extraction/entity_extractor.py`:

```
含有成分, 适用于, 禁用于, 用法用量, 给药途径, 不良反应,
警告注意, 药物相互作用, 需要监测, 特殊人群注意, 作用部位, 表现症状
```

Cross-layer edges (created by `ref_link()`): `REFERENCE`
Summary edges (created by `add_sum()`): `SUMMARIZES`

### The `gid` Design

Every document generates a UUID (`gid`). All nodes and relationships from that document carry this `gid`. This enables:
- Querying a single document's subgraph: `MATCH (n {gid: $gid})`
- Cross-layer linking via `ref_link(gid1, gid2)` which creates `REFERENCE` edges between high-cosine-similarity nodes across two subgraphs

---

## Embedding

- **Model**: Qwen3-Embedding-8B (local, `~/.cache/huggingface/hub/Qwen3-Embedding-8B`)
- **Dimension**: 1024 (truncated via MRL from the model's native 4096)
- **Device**: `cuda:0` (singleton, loaded on first call)
- **Normalisation**: L2-normalised (`normalize_embeddings=True`)
- **Text format**: `"{entity_name}: {description}"` for entities

Embeddings are stored as Neo4j node properties and used by:
- `merge_similar_nodes()` — merges nodes with cosine similarity > 0.5 within a subgraph
- `ref_link()` — creates REFERENCE edges for nodes with cosine similarity > 0.6 across subgraphs
- `GraphQA.answer()` — embeds a user question, retrieves the highest cosine-similarity nodes, expands one-hop evidence, and asks the LLM to answer from that evidence

---

## LLM Integration

### Backend Selection (`llm/config.py`)

Priority: GPT env vars → GLM env vars → OpenAI env vars → defaults.

| Env var | Purpose |
|---------|---------|
| `GPT_API_KEY` | Triggers GPT mode (highest priority) |
| `GPT_API_BASE_URL` | GPT proxy base URL — **must end with `/v1`** |
| `GPT_CHAT_MODEL` | GPT model name (e.g. `gpt-5.5`) |
| `GLM_API_KEY` | Triggers GLM mode (if GPT not set) |
| `GLM_API_BASE_URL` | GLM base URL (default: `https://open.bigmodel.cn/api/paas/v4`) |
| `GLM_CHAT_MODEL` | GLM model name (default: `glm-4.5`) |
| `OPENAI_API_KEY` | OpenAI fallback |
| `OPENAI_API_BASE_URL` | Custom OpenAI-compatible base URL |
| `OPENAI_CHAT_MODEL` | OpenAI model name |

### Rate Limit Handling

`llm/client.py` retries up to 5 times with exponential backoff (5s, 10s, 20s, 40s, 60s cap) on HTTP 429 or Chinese rate-limit error messages.

---

## Chunking Strategy

Two modes, selected at runtime:

| Mode | Flag | Function | When to use |
|------|------|----------|-------------|
| Fixed-size | *(default)* | `split_text_chunks(chunk_size=3000, overlap=300)` | Fast, works for most documents |
| Proposition | `--grained_chunk` | `run_chunk()` via LangChain | Better semantic boundaries; slower, requires `langchain_community` |

LangChain imports in `chunking/proposition.py` and `chunking/agentic.py` are **lazy** (inside function bodies) so the package imports cleanly even without LangChain installed.

---

## Neo4j Plugin Dependencies

Pipeline and QA features require Neo4j GDS and APOC plugins as follows:

| Flag | Functions used | Plugins needed |
|------|---------------|----------------|
| `--ingraphmerge` | `gds.similarity.cosine`, `apoc.refactor.mergeNodes`, `apoc.coll.sort` | GDS + APOC |
| `--trinity` | `gds.similarity.cosine`, `apoc.coll.sort` | GDS + APOC |
| `GraphQA.answer()` | `gds.similarity.cosine` | GDS |

The `docker-compose.yml` at the repo root installs both plugins automatically via `NEO4J_PLUGINS: '["graph-data-science", "apoc"]'`.

---

## Key Invariants

1. **`gid` uniqueness** — each `load_high()` call generates a fresh UUID. Never reuse gids across documents.
2. **Embedding singleton** — `embedding/local.py` uses a module-level `_model` variable. Do not reinitialize in parallel threads.
3. **Entity deduplication** — `create_metagraph_with_description()` merges entities with the same `entity_name` *before* writing to Neo4j (in-memory dict merge of descriptions). Neo4j-level deduplication uses `MERGE` on `(id, gid)`.
4. **Relation type allowlist** — only relations in `ALLOWED_RELATION_TYPES` are written. Unknown relation types from the LLM are silently dropped.
5. **Async extraction** — `_extract_entities_from_chunk()` is `async`; the top-level `create_metagraph_with_description()` is sync and calls `asyncio.run()`. Do not call it from within an already-running event loop.
