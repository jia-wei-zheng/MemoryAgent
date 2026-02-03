from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class MemoryType(str, Enum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PERCEPTUAL = "perceptual"


class StorageTier(str, Enum):
    HOT = "hot"
    COLD = "cold"
    ARCHIVE_INDEX = "archive_index"


class MemoryItem(BaseModel):
    """Canonical memory item stored in metadata storage."""

    id: UUID = Field(default_factory=uuid4)
    type: MemoryType
    owner: str
    summary: str
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    last_accessed: Optional[datetime] = None
    tier: StorageTier = StorageTier.HOT
    pointer: Dict[str, Any] = Field(default_factory=dict)
    content: Optional[Any] = None
    tags: List[str] = Field(default_factory=list)
    ttl_seconds: Optional[int] = None
    confidence: float = 0.5
    authority: float = 0.5
    stability: float = 0.5

    def is_expired(self, now: Optional[datetime] = None) -> bool:
        if self.ttl_seconds is None:
            return False
        now = now or utc_now()
        return self.created_at + timedelta(seconds=self.ttl_seconds) <= now

    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        return self.summary


class MemoryEvent(BaseModel):
    """Developer-facing input for memory writes."""

    content: Any
    type: MemoryType = MemoryType.WORKING
    owner: str
    summary: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    ttl_seconds: Optional[int] = None
    confidence: float = 0.5
    authority: float = 0.5
    stability: float = 0.5
    pointer: Dict[str, Any] = Field(default_factory=dict)

    def to_item(self) -> MemoryItem:
        summary = self.summary or (self.content if isinstance(self.content, str) else str(self.content))
        return MemoryItem(
            type=self.type,
            owner=self.owner,
            summary=summary,
            content=self.content,
            tags=self.tags,
            ttl_seconds=self.ttl_seconds,
            confidence=self.confidence,
            authority=self.authority,
            stability=self.stability,
            pointer=self.pointer,
        )


class MemoryQuery(BaseModel):
    text: str
    owner: str
    types: Optional[List[MemoryType]] = None
    top_k: int = 10
    time_range_seconds: Optional[int] = None


class RetrievalPlan(BaseModel):
    """Routing + budgets + thresholds for retrieval pipeline."""

    hot_top_k: int = 30
    archive_top_k: int = 30
    cold_fetch_limit: int = 20
    cold_fetch_min_score: float = 0.25
    hot_confidence: float = 0.62
    archive_confidence: float = 0.72
    cold_fetch_confidence: float = 0.58
    max_results: int = 50
    max_context_tokens: int = 600


class ScoredMemory(BaseModel):
    item: MemoryItem
    score: float
    tier: StorageTier
    explanation: Optional[str] = None


class ConfidenceReport(BaseModel):
    total: float
    semantic_relevance: float
    coverage: float
    temporal_fit: float
    authority: float
    consistency: float
    recommendation: str


class MemoryBlock(BaseModel):
    text: str
    item_id: UUID
    memory_type: MemoryType
    tier: StorageTier
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievalTrace(BaseModel):
    steps: List[str] = Field(default_factory=list)
    escalations: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)

    def add_step(self, text: str) -> None:
        self.steps.append(text)

    def add_escalation(self, text: str) -> None:
        self.escalations.append(text)


class MemoryBundle(BaseModel):
    query: str
    results: List[ScoredMemory]
    blocks: List[MemoryBlock]
    confidence: ConfidenceReport
    used_tiers: List[StorageTier]
    trace: RetrievalTrace
    warnings: List[str] = Field(default_factory=list)
