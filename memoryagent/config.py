from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from memoryagent.models import RetrievalPlan


class ConsolidationConfig(BaseModel):
    archive_on_flush: bool = True
    semantic_min_count: int = 2
    perceptual_summary_limit: int = 5


class MemorySystemConfig(BaseModel):
    """System-wide configuration with sane local defaults."""

    working_ttl_seconds: int = 3600
    retrieval_plan: RetrievalPlan = Field(default_factory=RetrievalPlan)
    consolidation: ConsolidationConfig = Field(default_factory=ConsolidationConfig)
    cold_store_path: Path = Field(default_factory=lambda: Path(".memoryagent_cold"))
    metadata_db_path: Path = Field(default_factory=lambda: Path(".memoryagent_hot.sqlite"))
    feature_db_path: Path = Field(default_factory=lambda: Path(".memoryagent_features.sqlite"))
    vector_db_path: Path = Field(default_factory=lambda: Path(".memoryagent_vectors.sqlite"))
    vector_dim: int = 384
    use_sqlite_vec: bool = False
    sqlite_vec_extension_path: Optional[Path] = None
    archive_index_path: Optional[Path] = None

    def resolved_archive_path(self) -> Path:
        if self.archive_index_path is not None:
            return self.archive_index_path
        return self.cold_store_path / "archive_index.json"
