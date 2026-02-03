# MemoryAgent

A reusable memory framework for LLM-based agent systems. It provides tiered memory (working, episodic, semantic, perceptual), hot/cold storage, archive indexing, confidence-based retrieval escalation, and optional local vector search via sqlite-vec.

## Highlights
- **Tiered memory**: working (TTL), episodic, semantic, perceptual
- **Storage tiers**: hot metadata (SQLite), cold archive (filesystem), archive index (vector index)
- **Confidence gating**: hot → archive → cold hydration with rerank + context packaging
- **Adapters**: MetadataStore, VectorIndex, GraphStore, ObjectStore, FeatureStore
- **Local mode**: SQLite + sqlite-vec (optional) + filesystem
- **Async-friendly** with sync convenience methods

## Project Layout
```
memoryagent/
  config.py
  models.py
  system.py
  retrieval.py
  confidence.py
  policy.py
  indexers.py
  workers.py
  storage/
    base.py
    in_memory.py
    local_disk.py
  examples/
    minimal.py
    openai_agent.py
    memory_api_server.py
    memory_viz.html
```

## Installation
Python 3.10+ required.

```bash
uv sync
```

Optional extras:
```bash
uv add openai sqlite-vec
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
- `ConsolidationWorker`: working → episodic/semantic
- `ArchiverWorker`: hot → cold + archive index
- `RehydratorWorker`: cold → hot (based on access)
- `Compactor`: cleanup/TTL

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
The page calls:
- `GET /api/memory?owner=user-001`
- `POST /api/chat`

## Data Stores
- **Hot metadata**: `.memoryagent_hot.sqlite`
- **Vector index**: `.memoryagent_vectors.sqlite` (sqlite-vec)
- **Features**: `.memoryagent_features.sqlite`
- **Cold archive**: `.memoryagent_cold/records/<owner>/YYYY/MM/DD/daily_notes.json`

## Configuration
See `memoryagent/config.py` for defaults:
- `working_ttl_seconds`
- `retrieval_plan` thresholds and budgets
- `use_sqlite_vec`, `vector_dim`, `sqlite_vec_extension_path`

## Notes
- Working memory is stored as a single session transcript (updated each turn).
- Episodic/semantic memories are candidates for cold archive.

## License
Add your license here.
