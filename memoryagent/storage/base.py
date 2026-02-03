from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List, Optional

from memoryagent.models import MemoryItem, MemoryQuery, ScoredMemory


class MetadataStore(ABC):
    """Stores canonical MemoryItem metadata."""

    @abstractmethod
    async def upsert(self, item: MemoryItem) -> None:
        raise NotImplementedError

    @abstractmethod
    async def get(self, item_id) -> Optional[MemoryItem]:
        raise NotImplementedError

    @abstractmethod
    async def delete(self, item_id) -> None:
        raise NotImplementedError

    @abstractmethod
    async def list_by_owner(self, owner: str) -> List[MemoryItem]:
        raise NotImplementedError

    @abstractmethod
    async def list_by_owner_and_type(self, owner: str, types: Iterable[str]) -> List[MemoryItem]:
        raise NotImplementedError

    @abstractmethod
    async def update_access(self, item_id) -> None:
        raise NotImplementedError


class VectorIndex(ABC):
    """Vector or lexical index supporting similarity search."""

    @abstractmethod
    async def upsert(self, item_id, text: str, metadata: dict) -> None:
        raise NotImplementedError

    @abstractmethod
    async def delete(self, item_id) -> None:
        raise NotImplementedError

    @abstractmethod
    async def query(self, query: MemoryQuery, filters: dict, limit: int) -> List[ScoredMemory]:
        raise NotImplementedError


class GraphStore(ABC):
    """Stores semantic relationships (facts, preferences, rules)."""

    @abstractmethod
    async def upsert_fact(self, owner: str, subject: str, predicate: str, obj: str) -> None:
        raise NotImplementedError

    @abstractmethod
    async def query_related(self, owner: str, subject: str, limit: int) -> List[str]:
        raise NotImplementedError


class ObjectStore(ABC):
    """Stores cold memory payloads."""

    @abstractmethod
    async def put(self, key: str, payload: dict) -> str:
        raise NotImplementedError

    @abstractmethod
    async def get(self, key: str) -> Optional[dict]:
        raise NotImplementedError


class FeatureStore(ABC):
    """Stores perceptual aggregates (time-series or feature logs)."""

    @abstractmethod
    async def write_feature(self, owner: str, payload: dict) -> None:
        raise NotImplementedError

    @abstractmethod
    async def query_features(self, owner: str, limit: int) -> List[dict]:
        raise NotImplementedError
