# Medical-Graph-RAG

Three-layer medical knowledge graph pipeline built on Neo4j. Extracts entities, relationships, and descriptions from medical documents via LLM, stores them as a structured graph, and links across layers using embedding-based similarity.

## Package Structure

```
Medical-Graph-RAG/
├── main.py                          # CLI entry point
└── medgraphrag/                     # Core package
    ├── pipeline/
    │   └── three_layer.py           # ThreeLayerImporter orchestration class
    ├── ingestion/
    │   ├── loader.py                # PDF / plain-text loader
    │   └── chunking/
    │       ├── basic.py             # Fixed-size text splitting
    │       ├── proposition.py       # LangChain proposition-level chunking
    │       └── agentic.py          # AgenticChunker (LLM-guided grouping)
    ├── extraction/
    │   └── entity_extractor.py      # LLM entity/relation extraction → Neo4j
    ├── graph/
    │   ├── store.py                 # Lightweight Neo4j driver wrapper
    │   └── operations.py           # merge_similar_nodes, ref_link, add_sum, str_uuid
    ├── embedding/
    │   └── local.py                 # Qwen3-Embedding-8B (local, 1024-dim)
    ├── llm/
    │   ├── config.py                # GLM / OpenAI-compatible model config
    │   ├── client.py                # Async OpenAI client with retry
    │   └── summarizer.py            # Medical text summarization
    └── qa.py                        # Python Graph RAG QA API
```

For a full design overview see [medgraphrag/DESIGN.md](medgraphrag/DESIGN.md).

## Setup

```bash
conda env create -f medgraphrag.yml
conda activate medgraphrag
```

### Neo4j

**Option A — Local Docker (recommended, includes GDS + APOC):**

```bash
docker compose up -d
```

Then open `http://localhost:7474` in a browser to inspect the graph.

**Option B — Neo4j Aura cloud:** create a free instance at [console.neo4j.io](https://console.neo4j.io) and copy the connection URI.

### Environment

Copy and edit the env file:

```bash
cp .env.example .env   # or edit .env directly
```

```env
# LLM — GPT proxy (highest priority when GPT_API_KEY is set)
GPT_API_KEY=your-gpt-api-key
GPT_API_BASE_URL=https://your-proxy.trycloudflare.com/v1   # must end with /v1
GPT_CHAT_MODEL=gpt-5.5

# LLM — GLM fallback (used when GPT_API_KEY is absent)
GLM_API_KEY=your-glm-api-key
GLM_API_BASE_URL=https://open.bigmodel.cn/api/paas/v4
GLM_CHAT_MODEL=glm-5.1

# Neo4j — local Docker
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password
NEO4J_LOCAL_PASSWORD=your-password   # used by docker-compose.yml

# Neo4j — Aura cloud (comment out local lines above)
# NEO4J_URI=neo4j+s://xxxx.databases.neo4j.io
# NEO4J_USERNAME=xxxx
# NEO4J_PASSWORD=xxxx
```

LLM backend priority: **GPT** (`GPT_API_KEY`) → **GLM** (`GLM_API_KEY`) → **OpenAI** (`OPENAI_API_KEY`).

> **Note:** `GPT_API_BASE_URL` must include the `/v1` path suffix so the OpenAI SDK routes requests correctly.

OpenAI fallback variables `OPENAI_API_KEY`, `OPENAI_API_BASE_URL`, `OPENAI_CHAT_MODEL` are also supported.

### Embedding Model

The pipeline uses **Qwen3-Embedding-8B** locally (1024-dim, no API cost). Download once:

```bash
env -u http_proxy -u https_proxy \
  HF_ENDPOINT=https://hf-mirror.com \
  hf download Qwen/Qwen3-Embedding-8B \
  --local-dir ~/.cache/huggingface/hub/Qwen3-Embedding-8B
```

## Data Layout

```
data/
  bottom/   # medical dictionaries, terminology
  middle/   # guidelines, textbooks, drug labels
  top/      # patient reports, case records
```

Supported formats: `.txt`, `.pdf` (searchable).

## Run

**Single layer:**

```bash
python main.py --top ./data/top --clear
```

**Full three-layer import:**

```bash
python main.py \
  --bottom ./data/bottom \
  --middle ./data/middle \
  --top    ./data/top \
  --clear --trinity
```

**All flags:**

| Flag | Description |
|------|-------------|
| `--bottom/--middle/--top` | Data path per layer |
| `--clear` | Wipe the database before import |
| `--trinity` | Create cross-layer REFERENCE edges (needs GDS + APOC) |
| `--ingraphmerge` | Merge similar nodes within each subgraph (needs GDS + APOC) |
| `--grained_chunk` | Use LangChain proposition chunking instead of fixed-size |
| `--skip_summary` | Skip Summary node generation |

> `--trinity` and `--ingraphmerge` require the GDS and APOC plugins. These are included automatically when using the Docker Compose setup.

## Python QA API

After importing data into Neo4j, use `GraphQA` for one-shot question answering over the graph. It embeds the question, retrieves similar non-`Summary` nodes, expands one-hop graph evidence, and asks the configured LLM to answer only from that evidence.

```python
from medgraphrag.graph import Neo4jGraph
from medgraphrag.qa import GraphQA

graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="your-password",
)

qa = GraphQA(graph)
result = qa.answer("二甲双胍在严重肾功能损害患者中是否禁用？为什么？")

print(result.answer)
for item in result.evidence:
    print(item.node_id, item.labels, item.score)
```

Optional arguments:

| Argument | Description |
|----------|-------------|
| `gids` | Limit retrieval to specific document subgraph IDs. Defaults to all graph nodes. |
| `top_k` | Number of embedding-similar seed nodes to retrieve. Defaults to `8`. |
| `neighbor_limit` | Maximum one-hop relationship rows added as evidence. Defaults to `30`. |

The QA API uses `gds.similarity.cosine`, so Neo4j GDS must be available.

## Cite

```bibtex
@article{wu2024medical,
  title={Medical Graph RAG: Towards Safe Medical Large Language Model via Graph Retrieval-Augmented Generation},
  author={Wu, Junde and Zhu, Jiayuan and Qi, Yunli},
  journal={arXiv preprint arXiv:2408.04187},
  year={2024}
}
```
