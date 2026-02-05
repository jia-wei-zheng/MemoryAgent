<div align="center">
  <img src="https://raw.githubusercontent.com/jia-wei-zheng/MemoryAgent/refs/heads/master/memoryagent_logo.jpg" alt="MemoryAgent" width="500">
  <h1>MemoryAgent: An Open, Modular Memory Framework for Agents (Beta)</h1>
</div>

MemoryAgent is a reusable memory framework for LLM-based agent systems. It provides tiered memory (working, episodic, semantic, perceptual), hot/cold storage, archive indexing, confidence-based retrieval escalation, and optional local vector search via sqlite-vec.

## Highlights
- **Tiered memory**: working (TTL), episodic, semantic, perceptual
- **Storage tiers**: hot metadata (SQLite + sqlite-vec), cold archive (filesystem), archive index (vector index)
- **Memory retrieval pipeline**: hot -> archive -> cold hydration with rerank + context packaging
- **Local mode**: SQLite + sqlite-vec (optional) + filesystem
- **Async-friendly** with sync convenience methods

## Project Layout
```
memoryagent/
  config.py            # Default system settings and retrieval thresholds
  models.py            # Pydantic data models for memory items, queries, bundles
  system.py            # MemorySystem entry point and wiring
  retrieval.py         # Retrieval orchestration and reranking
  confidence.py        # Confidence scoring components
  policy.py            # Conversation + routing policies
  indexers.py          # Episodic/semantic/perceptual indexers
  workers.py           # Consolidation, archiving, rehydration, compaction
  storage/
    base.py            # Storage adapter interfaces
    in_memory.py       # Simple in-memory vector + graph stores
    local_disk.py      # SQLite metadata/features + sqlite-vec + file object store
  examples/
    minimal.py         # Basic usage example
    openai_agent.py    # CLI OpenAI agent with memory retrieval
    memory_api_server.py # Local API for memory + chat
    memory_viz.html    # Web UI for chat + memory visualization
```

## Installation
Python 3.10+ required.

Development (sync deps from `uv.lock`):
```bash
uv sync
```

Use as a dependency:
```bash
uv add memoryagent-lib
# or
pip install memoryagent-lib
```

Optional extras:
```bash
uv add openai sqlite-vec
# or
pip install openai sqlite-vec
```

## Quick Start
```python
from memoryagent import MemoryEvent, MemorySystem

memory = MemorySystem()
owner = "user-001"

memory.write(
    MemoryEvent(
        content="User prefers concise summaries about climate policy.",
        type="semantic",
        owner=owner,
        tags=["preference", "summary"],
        confidence=0.7,
        stability=0.8,
    )
)

bundle = memory.retrieve("What policy topics did we cover?", owner=owner)
print(bundle.confidence.total)
for block in bundle.blocks:
    print(block.text)

memory.flush(owner)
```

## Enable sqlite-vec (Local Vector Search)
```python
from memoryagent import MemorySystem, MemorySystemConfig

config = MemorySystemConfig(
    use_sqlite_vec=True,
    vector_dim=1536,  # match your embedding model
)

memory = MemorySystem(config=config)
```

If sqlite-vec cannot be auto-loaded, set an explicit path:
```python
from pathlib import Path
from memoryagent import MemorySystemConfig

config = MemorySystemConfig(
    use_sqlite_vec=True,
    vector_dim=1536,
    sqlite_vec_extension_path=Path("/path/to/sqlite_vec.dylib"),
)
```

## Examples
### OpenAI Agent (CLI)
```bash
python -m memoryagent.examples.openai_agent
```
- Uses OpenAI responses + embeddings.
- Stores session transcript as a single working memory item.

### Memory Visualization + API
Start the API server:
```bash
python -m memoryagent.examples.memory_api_server
```
Open in browser:
```
http://127.0.0.1:8000/memory_viz.html
```

An example (System records semantic memory and updating working memory): 

![Screenshot](https://raw.githubusercontent.com/jia-wei-zheng/MemoryAgent/457c3fdadc099727b337838634b9ca3a4ec89fe8/Memory%20Agent%20_%20Live%20Console.jpeg)


The page calls:
- `GET /api/memory?owner=user-001`
- `POST /api/chat`

## Policies
### Conversation storage policy
`HeuristicMemoryPolicy` decides whether a turn should be stored and whether it becomes episodic or semantic memory.

### Routing policy
`MemoryRoutingPolicy` decides where a memory should be written:
- **Hot** metadata store
- **Vector index**
- **Feature store** (perceptual)
- **Cold** archive (via workers)

## Background Workers
- `ConsolidationWorker`: working -> episodic/semantic
- `ArchiverWorker`: hot -> cold + archive index
- `RehydratorWorker`: cold -> hot (based on access)
- `Compactor`: cleanup/TTL


## Data Stores
- **Hot metadata**: `.memoryagent_hot.sqlite`
- **Vector index**: `.memoryagent_vectors.sqlite` (sqlite-vec)
- **Features**: `.memoryagent_features.sqlite`
- **Cold archive**: `.memoryagent_cold/records/<owner>/YYYY/MM/DD/daily_notes.json`

## Data Root (Installed Usage)
The system auto-detects a project root by walking up from the current working directory and looking for `pyproject.toml` or `.git`. If it can’t find one, it uses the current directory.


## Configuration
See `memoryagent/config.py` for defaults:
- `working_ttl_seconds`
- `retrieval_plan` thresholds and budgets
- `use_sqlite_vec`, `vector_dim`, `sqlite_vec_extension_path`

## Notes
- Working memory is stored as a single session transcript (updated each turn).
- Episodic/semantic memories are candidates for cold archive.

## License
MIT License
