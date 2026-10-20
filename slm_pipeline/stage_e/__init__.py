"""Stage E: Deduplication (exact + fuzzy + paragraph)."""

from .dedup import (
    DedupResult,
    ExactDuplicates,
    FuzzyDeduplicator,
    deduplicate_paragraphs,
)

__all__ = [
    "DedupResult",
    "ExactDuplicates",
    "FuzzyDeduplicator",
    "deduplicate_paragraphs",
]
