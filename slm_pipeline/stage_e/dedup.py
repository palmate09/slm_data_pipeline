"""Deduplication: exact (hash), fuzzy (MinHash/LSH), and paragraph-level."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from datasketch import MinHash, MinHashLSH

from ..models import Block
from ..stage_a.html_extract import blocks_to_text


@dataclass
class DedupResult:
    kept_ids: list[str]
    dropped_ids: list[str]
    drop_reasons: dict[str, str] = field(default_factory=dict)
    exact_duplicates: int = 0
    fuzzy_duplicates: int = 0
    paragraph_duplicates: int = 0


def _normalize_for_hash(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).casefold()


def _sha256(text: str) -> str:
    return hashlib.sha256(_normalize_for_hash(text).encode("utf-8")).hexdigest()


def _doc_key(doc, index: int, id_field: str = "id") -> str:
    return getattr(doc, id_field, None) or f"__doc_{index}"


class ExactDuplicates:
    """Remove exact duplicates, keeping the best representative.

    ``kept_ids`` contains only documents that survived; the old implementation
    could lose track of cluster members.
    """

    def __init__(self, text_field: str = "text", id_field: str = "id") -> None:
        self.text_field = text_field
        self.id_field = id_field

    def __call__(self, documents: Iterable) -> DedupResult:
        docs = list(documents)
        groups: dict[str, list[tuple[int, object]]] = {}
        for index, d in enumerate(docs):
            if not getattr(d, "kept", True):
                continue
            text = getattr(d, self.text_field) or ""
            groups.setdefault(_sha256(text), []).append((index, d))

        kept: list[str] = []
        dropped: list[str] = []
        reasons: dict[str, str] = {}
        exact_dups = 0
        for members in groups.values():
            # Best quality, then longest, then earliest input order.
            members.sort(
                key=lambda pair: (
                    getattr(pair[1], "quality_score", None) or 0.0,
                    getattr(pair[1], "word_count", 0),
                    -pair[0],
                ),
                reverse=True,
            )
            rep_index, rep = members[0]
            rep_id = _doc_key(rep, rep_index, self.id_field)
            rep.kept = True
            kept.append(rep_id)
            for index, doc in members[1:]:
                did = _doc_key(doc, index, self.id_field)
                dropped.append(did)
                reasons[did] = f"exact duplicate of {rep_id}"
                doc.kept = False
                doc.drop_reason = reasons[did]
                exact_dups += 1
        return DedupResult(
            kept_ids=kept,
            dropped_ids=dropped,
            drop_reasons=reasons,
            exact_duplicates=exact_dups,
        )


class FuzzyDeduplicator:
    """Remove near-duplicates via MinHash signatures + LSH (Jaccard)."""

    def __init__(
        self,
        *,
        threshold: float = 0.85,
        num_perm: int = 128,
        shingle_size: int = 5,
        id_field: str = "id",
        text_field: str = "text",
    ) -> None:
        self.threshold = threshold
        self.num_perm = num_perm
        self.shingle_size = shingle_size
        self.id_field = id_field
        self.text_field = text_field

    def _shingles(self, text: str) -> list[str]:
        words = re.findall(r"\w+", (text or "").lower())
        if not words:
            return []
        size = min(self.shingle_size, max(1, len(words)))
        return [
            " ".join(words[i:i + size])
            for i in range(max(0, len(words) - size + 1))
        ]

    def __call__(self, documents: Iterable) -> DedupResult:
        docs = [d for d in documents if getattr(d, "kept", True)]
        if len(docs) < 2:
            return DedupResult(kept_ids=[], dropped_ids=[])

        keys: list[str] = []
        signatures: dict[str, MinHash] = {}
        doc_map: dict[str, object] = {}
        lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)

        for index, d in enumerate(docs):
            did = _doc_key(d, index, self.id_field)
            # Ensure unique keys even if input ids collide.
            while did in doc_map:
                did = did + "_"
            keys.append(did)
            doc_map[did] = d
            text = getattr(d, self.text_field) or ""
            mh = MinHash(num_perm=self.num_perm)
            for shingle in self._shingles(text):
                mh.update(shingle.encode("utf-8"))
            signatures[did] = mh
            try:
                lsh.insert(did, mh)
            except ValueError:
                # Empty/degenerate signature: treat as not indexable.
                pass

        # Keep the best document in each connected LSH neighbourhood.  Sorting
        # by quality/length makes the representative deterministic and avoids
        # the old bug where a dropped cluster member was reported as kept.
        order = sorted(
            keys,
            key=lambda k: (
                getattr(doc_map[k], "quality_score", None) or 0.0,
                getattr(doc_map[k], "word_count", 0),
                -keys.index(k),
            ),
            reverse=True,
        )
        kept: list[str] = []
        dropped: list[str] = []
        reasons: dict[str, str] = {}
        dropped_set: set[str] = set()
        for did in order:
            if did in dropped_set:
                continue
            try:
                cluster = set(lsh.query(signatures[did]))
            except Exception:  # noqa: BLE001
                cluster = {did}
            cluster.add(did)
            kept.append(did)
            for other in sorted(cluster):
                if other == did or other in dropped_set:
                    continue
                dropped_set.add(other)
                dropped.append(other)
                reasons[other] = f"near-duplicate (fuzzy) of {did}"
                doc = doc_map[other]
                doc.kept = False
                doc.drop_reason = reasons[other]

        return DedupResult(
            kept_ids=kept,
            dropped_ids=dropped,
            drop_reasons=reasons,
            fuzzy_duplicates=len(dropped),
        )


def deduplicate_paragraphs(
    documents: Iterable,
    *,
    min_words: int = 12,
    update_text: bool = True,
) -> DedupResult:
    """Remove exact repeated paragraphs across kept documents.

    Short paragraphs are ignored because boilerplate/definitions often repeat
    legitimately.  The first high-quality occurrence wins; if removing a
    paragraph would leave a document with almost no prose, the paragraph is
    retained.
    """
    docs = [d for d in documents if getattr(d, "kept", True)]
    order = sorted(
        docs,
        key=lambda d: (
            getattr(d, "quality_score", None) or 0.0,
            getattr(d, "word_count", 0),
        ),
        reverse=True,
    )
    seen: set[str] = set()
    duplicates = 0
    for d in order:
        blocks = getattr(d, "blocks", None) or []
        if not blocks:
            continue
        new_blocks: list[Block] = []
        removed = 0
        for block in blocks:
            if block.type != "paragraph" or len(block.text.split()) < min_words:
                new_blocks.append(block)
                continue
            key = _normalize_for_hash(block.text)
            if key in seen:
                removed += 1
                duplicates += 1
                continue
            seen.add(key)
            new_blocks.append(block)
        if not removed:
            continue
        prose_words = sum(
            len(b.text.split()) for b in new_blocks
            if b.type in {"paragraph", "quote", "list"}
        )
        if prose_words < max(30, min_words):
            continue
        d.blocks = new_blocks
        if update_text:
            d.text = blocks_to_text(new_blocks).strip()
        d.metadata["paragraph_dedup_removed"] = removed
    return DedupResult(
        kept_ids=[getattr(d, "id", str(i)) for i, d in enumerate(docs)],
        dropped_ids=[],
        paragraph_duplicates=duplicates,
    )
