from __future__ import annotations

from typing import List

from memoryagent.models import MemoryItem, MemoryType, StorageTier
from memoryagent.storage.base import FeatureStore, GraphStore, VectorIndex
from memoryagent.utils import tokenize


class EpisodicIndexer:
    """Indexes episodic content into the vector index."""

    def __init__(self, vector_index: VectorIndex) -> None:
        self.vector_index = vector_index

    async def index_hot(self, item: MemoryItem) -> None:
        await self.vector_index.upsert(
            item.id,
            text=item.text(),
            metadata={"owner": item.owner, "tier": StorageTier.HOT.value, "type": item.type.value, "item": item},
        )

    async def index_archive(self, item: MemoryItem) -> None:
        await self.vector_index.upsert(
            item.id,
            text=item.summary,
            metadata={"owner": item.owner, "tier": StorageTier.ARCHIVE_INDEX.value, "type": item.type.value, "item": item},
        )


class SemanticGraphIndexer:
    """Extracts simple fact-like triples from tags for demo use."""

    def __init__(self, graph_store: GraphStore) -> None:
        self.graph_store = graph_store

    async def index(self, item: MemoryItem) -> None:
        if item.type != MemoryType.SEMANTIC:
            return
        if len(item.tags) < 2:
            return
        subject = item.tags[0]
        for tag in item.tags[1:]:
            await self.graph_store.upsert_fact(item.owner, subject, "related_to", tag)


class PerceptualIndexer:
    """Summarizes perceptual inputs into feature store entries."""

    def __init__(self, feature_store: FeatureStore) -> None:
        self.feature_store = feature_store

    async def index(self, item: MemoryItem) -> None:
        if item.type != MemoryType.PERCEPTUAL:
            return
        payload = {
            "summary": item.summary,
            "tags": item.tags,
            "confidence": item.confidence,
        }
        await self.feature_store.write_feature(item.owner, payload)
