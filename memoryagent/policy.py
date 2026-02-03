from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from memoryagent.models import MemoryEvent, MemoryItem, MemoryType, StorageTier
from memoryagent.utils import tokenize


@dataclass
class MemoryDecision:
    store: bool
    memory_type: MemoryType
    summary: Optional[str] = None
    tags: Optional[List[str]] = None
    reasons: Optional[List[str]] = None


class ConversationMemoryPolicy:
    """Decides if a turn should be persisted to memory."""

    def should_store(
        self,
        owner: str,
        history: List[str],
        user_message: str,
        assistant_message: str,
    ) -> MemoryDecision:
        raise NotImplementedError

    def to_event(self, owner: str, decision: MemoryDecision) -> Optional[MemoryEvent]:
        if not decision.store or not decision.summary:
            return None
        return MemoryEvent(
            content=decision.summary,
            type=decision.memory_type,
            owner=owner,
            tags=decision.tags or [],
        )


class HeuristicMemoryPolicy(ConversationMemoryPolicy):
    """Default heuristic policy for deciding what to store."""

    def __init__(
        self,
        min_tokens: int = 24,
        novelty_threshold: float = 0.65,
        short_turn_min_novelty: float = 0.8,
        preference_keywords: Optional[Iterable[str]] = None,
    ) -> None:
        self.min_tokens = min_tokens
        self.novelty_threshold = novelty_threshold
        self.short_turn_min_novelty = short_turn_min_novelty
        self.preference_keywords = set(
            k.lower()
            for k in (preference_keywords or ["prefer", "always", "never", "likes", "dislikes"])
        )

    def should_store(
        self,
        owner: str,
        history: List[str],
        user_message: str,
        assistant_message: str,
    ) -> MemoryDecision:
        combined = f"{user_message} {assistant_message}"
        tokens = tokenize(combined)
        reasons: List[str] = []
        memory_type = MemoryType.EPISODIC

        is_preference = any(word in combined.lower() for word in self.preference_keywords)
        if len(tokens) < self.min_tokens:
            reasons.append("short_turn")
        if is_preference:
            memory_type = MemoryType.SEMANTIC
            reasons.append("preference_signal")

        if history:
            recent_entries = history[-3:]
            recent_text = " ".join(self._history_entry_text(entry) for entry in recent_entries)
            novelty = 1.0 - self._overlap_ratio(tokens, tokenize(recent_text))
            novelty_floor = self.short_turn_min_novelty if len(tokens) < self.min_tokens else self.novelty_threshold
            if novelty < novelty_floor:
                reasons.append("low_novelty")

        if is_preference:
            store = True
        else:
            store = "short_turn" not in reasons and "low_novelty" not in reasons
        summary = self._summarize(user_message, assistant_message, memory_type)
        tags = ["conversation", memory_type.value]
        return MemoryDecision(store=store, memory_type=memory_type, summary=summary, tags=tags, reasons=reasons)

    def _overlap_ratio(self, tokens_a: List[str], tokens_b: List[str]) -> float:
        if not tokens_a or not tokens_b:
            return 0.0
        set_a = set(tokens_a)
        set_b = set(tokens_b)
        return len(set_a & set_b) / max(1, len(set_a | set_b))

    def _summarize(self, user_message: str, assistant_message: str, memory_type: MemoryType) -> str:
        if memory_type == MemoryType.SEMANTIC:
            return f"User preference: {user_message.strip()}"
        return f"User asked: {user_message.strip()} | Assistant replied: {assistant_message.strip()}"

    def _history_entry_text(self, entry) -> str:
        if isinstance(entry, str):
            return entry
        if isinstance(entry, dict):
            if "user" in entry and "assistant" in entry:
                return f"User: {entry['user']} Assistant: {entry['assistant']}"
            if "role" in entry and "text" in entry:
                return f"{entry['role']}: {entry['text']}"
        return str(entry)


@dataclass
class RoutingDecision:
    write_hot: bool
    write_vector: bool
    write_features: bool
    archive_cold: bool
    reasons: List[str]


class MemoryRoutingPolicy:
    """Concrete policy for routing memory writes across tiers and indexes."""

    def __init__(
        self,
        hot_min_confidence: float = 0.4,
        cold_min_confidence: float = 0.55,
        vector_min_confidence: float = 0.5,
        feature_min_confidence: float = 0.45,
    ) -> None:
        self.hot_min_confidence = hot_min_confidence
        self.cold_min_confidence = cold_min_confidence
        self.vector_min_confidence = vector_min_confidence
        self.feature_min_confidence = feature_min_confidence

    def route(self, item: MemoryItem) -> RoutingDecision:
        reasons: List[str] = []
        confidence = item.confidence

        write_hot = confidence >= self.hot_min_confidence
        if not write_hot:
            reasons.append("low_confidence_hot")

        write_vector = confidence >= self.vector_min_confidence and item.type != MemoryType.WORKING
        if not write_vector:
            reasons.append("skip_vector")

        write_features = item.type == MemoryType.PERCEPTUAL and confidence >= self.feature_min_confidence
        if not write_features and item.type == MemoryType.PERCEPTUAL:
            reasons.append("skip_features")

        archive_cold = (
            item.type in {MemoryType.EPISODIC, MemoryType.SEMANTIC, MemoryType.PERCEPTUAL}
            and confidence >= self.cold_min_confidence
        )
        if not archive_cold:
            reasons.append("skip_cold")

        return RoutingDecision(
            write_hot=write_hot,
            write_vector=write_vector,
            write_features=write_features,
            archive_cold=archive_cold,
            reasons=reasons,
        )
