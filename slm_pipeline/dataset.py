"""In-memory document collection with NeMo Curator-style DocumentDataset API.

This is a lightweight, CPU-friendly stand-in for NeMo Curator's
``DocumentDataset``. It holds ``Document`` objects and offers ``map`` /
``filter`` / ``to_json`` / ``to_parquet`` so the rest of the pipeline can be
written against a stable abstraction and later swapped onto Ray/cuDF without
changing stage logic.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Callable

try:  # optional Parquet support
    import pyarrow as pa
    import pyarrow.parquet as pq

    _HAS_PARQUET = True
except ImportError:  # pragma: no cover - optional dependency
    pa = None  # type: ignore[assignment]
    pq = None  # type: ignore[assignment]
    _HAS_PARQUET = False

from .models import Document


class DocumentDataset:
    """A sharded-aware collection of Documents."""

    def __init__(self, documents: Iterable[Document] | None = None):
        self._docs: list[Document] = list(documents or [])

    # ------------------------------------------------------------------ core
    def __len__(self) -> int:
        return len(self._docs)

    def __iter__(self) -> Iterator[Document]:
        return iter(self._docs)

    def __getitem__(self, idx):
        return self._docs[idx]

    @property
    def documents(self) -> list[Document]:
        return self._docs

    @classmethod
    def from_dicts(cls, dicts: Iterable[dict]) -> DocumentDataset:
        return cls([Document.from_dict(d) for d in dicts])

    def to_dicts(self, include_blocks: bool = True) -> list[dict]:
        return [d.to_dict(include_blocks=include_blocks) for d in self._docs]

    # ---------------------------------------------------------------- transforms
    def map(self, func: Callable[[Document], Document]) -> DocumentDataset:
        return DocumentDataset([func(d) for d in self._docs])

    def filter(self, pred: Callable[[Document], bool]) -> DocumentDataset:
        return DocumentDataset([d for d in self._docs if pred(d)])

    def flat_map(self, func: Callable[[Document], Iterable[Document]]) -> DocumentDataset:
        out: list[Document] = []
        for d in self._docs:
            out.extend(func(d))
        return DocumentDataset(out)

    # ---------------------------------------------------------------- persistence
    def to_json(
        self,
        output_path: str | Path,
        write_to_filename: bool = True,
        *,
        include_blocks: bool = True,
    ) -> str:
        """Write one JSON object per line (JSONL)."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for d in self._docs:
                f.write(json.dumps(d.to_dict(include_blocks=include_blocks), ensure_ascii=False))
                f.write("\n")
        return str(output_path)

    def to_jsonl(
        self,
        output_path: str | Path,
        write_to_filename: bool = True,
        *,
        include_blocks: bool = True,
    ) -> str:
        return self.to_json(
            output_path, write_to_filename=write_to_filename, include_blocks=include_blocks
        )

    def to_parquet(self, output_path: str | Path, *, include_blocks: bool = True) -> str:
        if not _HAS_PARQUET:
            raise RuntimeError(
                "Parquet output requires pyarrow. Install with: uv pip install pyarrow"
            )
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist(self.to_dicts(include_blocks=include_blocks))
        pq.write_table(table, output_path)
        return str(output_path)

    # ---------------------------------------------------------------- stats
    def stats(self) -> dict:
        total = len(self._docs)
        kept = sum(1 for d in self._docs if d.kept)
        dropped = total - kept
        words = sum(d.word_count for d in self._docs if d.kept)
        return {
            "total": total,
            "kept": kept,
            "dropped": dropped,
            "kept_rate": round(kept / total, 4) if total else 0.0,
            "total_words_kept": words,
            "avg_words_kept": round(words / kept, 1) if kept else 0,
        }
