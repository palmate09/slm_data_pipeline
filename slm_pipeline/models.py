"""Core data model: a single curation Document and its structured blocks."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def new_id() -> str:
    return str(uuid.uuid4())


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Canonical block types.  Extraction produces these; cleaning operates on them
# before rendering a flat training text.
BLOCK_TYPES = {
    "heading",
    "paragraph",
    "list",
    "code",
    "table",
    "caption",
    "formula",
    "quote",
    "footnote",
    "navigation",
}


@dataclass
class Block:
    """One structural unit of a document.

    ``type`` is one of :data:`BLOCK_TYPES`.  ``level`` is the heading level
    (1-6) for headings.  ``meta`` keeps extractor-specific details such as the
    original tag name or the caption an image belonged to.
    """

    type: str = "paragraph"
    text: str = ""
    level: int | None = None
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        out: dict[str, Any] = {"type": self.type, "text": self.text}
        if self.level is not None:
            out["level"] = self.level
        if self.meta:
            out["meta"] = self.meta
        return out

    @classmethod
    def from_dict(cls, d: dict) -> Block:
        btype = d.get("type", "paragraph")
        if btype not in BLOCK_TYPES:
            btype = "paragraph"
        return cls(
            type=btype,
            text=d.get("text", "") or "",
            level=d.get("level"),
            meta=d.get("meta", {}) or {},
        )


@dataclass
class Document:
    """One text record flowing through the pipeline.

    Fields mirror the target output schema so the final JSONL line is a direct
    serialization of this object.  ``blocks`` is the structured representation
    used by extraction and cleaning; ``text`` is the rendered view that is
    eventually written for training.
    """

    id: str
    text: str
    url: str
    title: str = ""
    source: str = ""
    fetched_at: str = field(default_factory=now_iso)

    # Structure / provenance.
    blocks: list[Block] = field(default_factory=list)
    language: str | None = None
    license: str | None = None
    canonical_url: str | None = None
    content_hash: str | None = None
    extraction: dict = field(default_factory=dict)

    # Populated by later stages.
    quality_score: float | None = None
    stage: str = "raw"
    kept: bool = True
    drop_reason: str | None = None
    metadata: dict = field(default_factory=dict)

    # Curriculum ordering (base knowledge -> advanced), set in Stage F.
    difficulty: float | None = None
    curriculum_rank: int | None = None
    curriculum_level: str | None = None

    # ------------------------------------------------------------------ helpers
    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def is_empty(self) -> bool:
        return not self.text or not self.text.strip()

    @property
    def headings(self) -> list[str]:
        return [b.text for b in self.blocks if b.type == "heading" and b.text]

    def to_dict(self, include_blocks: bool = True) -> dict:
        """Serialize to a dict suitable for JSONL / Parquet output."""
        d = asdict(self)
        # Order output so `text` is prominent; drop internal drop_reason unless set.
        ordered = {
            "id": d["id"],
            "text": d["text"],
            "url": d["url"],
            "source": d["source"],
            "title": d["title"],
            "fetched_at": d["fetched_at"],
        }
        for key in ("language", "license", "canonical_url", "content_hash"):
            if d.get(key) is not None:
                ordered[key] = d[key]
        if include_blocks and d.get("blocks"):
            ordered["blocks"] = d["blocks"]
        if d.get("extraction"):
            ordered["extraction"] = d["extraction"]
        ordered["stage"] = d["stage"]
        ordered["kept"] = d["kept"]
        if d["quality_score"] is not None:
            ordered["quality_score"] = d["quality_score"]
        if d.get("difficulty") is not None:
            ordered["difficulty"] = d["difficulty"]
        if d.get("curriculum_rank") is not None:
            ordered["curriculum_rank"] = d["curriculum_rank"]
        if d.get("curriculum_level"):
            ordered["curriculum_level"] = d["curriculum_level"]
        if d["drop_reason"] is not None:
            ordered["drop_reason"] = d["drop_reason"]
        if d["metadata"]:
            ordered["metadata"] = d["metadata"]
        return ordered

    @classmethod
    def from_dict(cls, d: dict) -> Document:
        return cls(
            id=d.get("id") or new_id(),
            text=d.get("text", ""),
            url=d.get("url", ""),
            title=d.get("title", ""),
            source=d.get("source", ""),
            fetched_at=d.get("fetched_at") or now_iso(),
            blocks=[Block.from_dict(b) for b in (d.get("blocks") or [])],
            language=d.get("language"),
            license=d.get("license"),
            canonical_url=d.get("canonical_url"),
            content_hash=d.get("content_hash"),
            extraction=d.get("extraction", {}) or {},
            quality_score=d.get("quality_score"),
            stage=d.get("stage", "raw"),
            kept=d.get("kept", True),
            drop_reason=d.get("drop_reason"),
            metadata=d.get("metadata", {}) or {},
            difficulty=d.get("difficulty"),
            curriculum_rank=d.get("curriculum_rank"),
            curriculum_level=d.get("curriculum_level"),
        )

    def __repr__(self) -> str:
        return (
            f"Document(id={self.id[:8]}..., url={self.url!r}, "
            f"words={self.word_count}, blocks={len(self.blocks)}, "
            f"stage={self.stage}, kept={self.kept})"
        )
