"""Token-aware chunking and deterministic train/val/test splits."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from ..models import Block, Document
from ..stage_a.html_extract import blocks_to_text

try:
    import tiktoken
except Exception:  # noqa: BLE001
    tiktoken = None  # type: ignore[assignment]


class TokenCounter:
    """Count tokens with tiktoken, falling back to whitespace words."""

    def __init__(self, tokenizer: str = "tiktoken:cl100k_base") -> None:
        self.tokenizer = tokenizer or "words"
        self._encoding = None
        self._word_fallback = True
        if tiktoken is not None and self.tokenizer.startswith("tiktoken:"):
            name = self.tokenizer.split(":", 1)[1] or "cl100k_base"
            try:
                self._encoding = tiktoken.get_encoding(name)
                self._word_fallback = False
            except Exception:  # noqa: BLE001
                self._encoding = None
        elif tiktoken is not None and self.tokenizer not in {"words", "word"}:
            try:
                self._encoding = tiktoken.encoding_for_model(self.tokenizer)
                self._word_fallback = False
            except Exception:  # noqa: BLE001
                self._encoding = None

    def _encode(self, text: str) -> list[int]:
        """Encode text, treating special tokens as ordinary text.

        Corpus documents legitimately contain strings like ``<|endoftext|>``
        (code samples, tokenizer docs).  tiktoken raises ``ValueError`` for
        those unless ``disallowed_special`` is empty, which used to abort the
        whole chunking stage.
        """
        return self._encoding.encode(text, disallowed_special=())

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._encoding is not None:
            return len(self._encode(text))
        return len(text.split())

    def tail(self, text: str, tokens: int) -> str:
        if tokens <= 0 or not text:
            return ""
        if self._encoding is not None:
            ids = self._encode(text)
            return self._encoding.decode(ids[-tokens:])
        words = text.split()
        return " ".join(words[-tokens:])

    def trim_front(self, text: str, max_tokens: int) -> str:
        if self.count(text) <= max_tokens:
            return text
        if self._encoding is not None:
            ids = self._encode(text)
            return self._encoding.decode(ids[-max_tokens:])
        return " ".join(text.split()[-max_tokens:])

    def trim_back(self, text: str, max_tokens: int) -> str:
        """Keep the beginning of a chunk and trim only its tail."""
        if self.count(text) <= max_tokens:
            return text
        if self._encoding is not None:
            ids = self._encode(text)
            return self._encoding.decode(ids[:max_tokens])
        return " ".join(text.split()[:max_tokens])

    def split_parts(self, text: str, max_tokens: int) -> list[str]:
        """Split text into consecutive pieces of at most ``max_tokens`` each.

        All pieces concatenate back to the original text, so callers can use
        this as a lossless fallback for pathological blocks.
        """
        if not text:
            return []
        if max_tokens <= 0 or self.count(text) <= max_tokens:
            return [text]
        if self._encoding is not None:
            ids = self._encode(text)
            return [
                self._encoding.decode(ids[i:i + max_tokens])
                for i in range(0, len(ids), max_tokens)
            ]
        words = text.split()
        return [
            " ".join(words[i:i + max_tokens])
            for i in range(0, len(words), max_tokens)
        ]


@dataclass
class Chunk:
    id: str
    doc_id: str
    text: str
    title: str = ""
    url: str = ""
    source: str = ""
    language: str | None = None
    license: str | None = None
    fetched_at: str = ""
    heading_path: list[str] = field(default_factory=list)
    chunk_index: int = 0
    token_count: int = 0
    quality_score: float | None = None
    split: str = "train"
    content_hash: str = ""
    schema_version: str = "2.0"

    # Curriculum ordering (base knowledge -> advanced), copied from the
    # source document in Stage F.
    difficulty: float | None = None
    curriculum_rank: int | None = None
    curriculum_level: str | None = None

    def to_dict(self) -> dict:
        out = {
            "id": self.id,
            "doc_id": self.doc_id,
            "text": self.text,
            "title": self.title,
            "url": self.url,
            "source": self.source,
        }
        if self.language:
            out["language"] = self.language
        if self.license:
            out["license"] = self.license
        if self.fetched_at:
            out["fetched_at"] = self.fetched_at
        out["heading_path"] = self.heading_path
        out["chunk_index"] = self.chunk_index
        out["token_count"] = self.token_count
        if self.quality_score is not None:
            out["quality_score"] = self.quality_score
        out["split"] = self.split
        out["content_hash"] = self.content_hash
        out["schema_version"] = self.schema_version
        if self.difficulty is not None:
            out["difficulty"] = self.difficulty
        if self.curriculum_rank is not None:
            out["curriculum_rank"] = self.curriculum_rank
        if self.curriculum_level:
            out["curriculum_level"] = self.curriculum_level
        return out


def assign_split(document: Document, splits: dict | None) -> str:
    """Deterministic split assignment independent of processing order."""
    if not splits:
        return "train"
    keys = [k for k, v in splits.items() if float(v) > 0]
    if not keys:
        return "train"
    total = sum(float(splits[k]) for k in keys)
    if total <= 0:
        return "train"
    seed = (getattr(document, "id", "") or getattr(document, "url", "") or document.text[:128])
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    point = int(digest[:12], 16) / float(16 ** 12)
    cumulative = 0.0
    for key in keys:
        cumulative += float(splits[key]) / total
        if point <= cumulative:
            return key
    return keys[-1]


def assign_splits(
    documents: Iterable[Document],
    splits: dict | None,
    *,
    sizes: dict[str, int] | None = None,
) -> dict[str, str]:
    """Assign documents to splits deterministically, balancing by size.

    A document is never shared between splits, so train and evaluation data
    cannot leak into each other.  Documents are handed out by size: the split
    that is furthest below its target share receives the next document, so
    every split ends up close to its configured ratio instead of receiving one
    arbitrarily small document.  A stable hash breaks ties, which keeps the
    assignment reproducible.  Each non-zero split receives at least one
    document when the corpus is large enough.
    """
    docs = list(documents)
    if not docs or not splits:
        return {getattr(d, "id", str(i)): "train" for i, d in enumerate(docs)}
    keys = [k for k, v in splits.items() if float(v) > 0]
    if not keys:
        return {getattr(d, "id", str(i)): "train" for i, d in enumerate(docs)}
    ratio_total = sum(float(splits[k]) for k in keys) or 1.0

    def doc_id(doc: Document, index: int) -> str:
        return getattr(doc, "id", "") or str(index)

    def doc_size(doc: Document, index: int) -> int:
        if sizes is not None:
            return int(sizes.get(doc_id(doc, index), 0))
        return len((getattr(doc, "text", "") or "").split())

    def stable_hash(doc: Document) -> str:
        seed = (
            getattr(doc, "id", "")
            or getattr(doc, "url", "")
            or (getattr(doc, "text", "") or "")[:128]
        )
        return hashlib.sha256(seed.encode("utf-8")).hexdigest()

    entries = []
    for index, doc in enumerate(docs):
        entries.append((doc_id(doc, index), doc_size(doc, index), stable_hash(doc)))
    total_size = sum(entry[1] for entry in entries) or 1
    targets = {k: total_size * float(splits[k]) / ratio_total for k in keys}

    # Biggest documents first, hash for a stable order.
    ordered = sorted(entries, key=lambda e: (-e[1], e[2]))
    mapping: dict[str, str] = {}
    assigned_size = {k: 0 for k in keys}

    def relative_deficit(key: str) -> float:
        target = targets[key]
        if target <= 0:
            return float("-inf")
        return (target - assigned_size[key]) / target

    for entry in ordered:
        # Prefer a split that can still absorb the document without
        # overshooting its target; among those, serve the split that is
        # furthest below its target share.
        fitting = [
            key for key in keys
            if assigned_size[key] + entry[1] <= targets[key]
        ]
        candidates = fitting or list(keys)
        key = max(
            candidates,
            key=lambda k: (relative_deficit(k), -keys.index(k)),
        )
        mapping[entry[0]] = key
        assigned_size[key] += entry[1]
    return mapping


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")

# A chunk that opens with dots/ellipsis is a continuation splice ("...where
# the function is defined"), not a standalone training unit.
_LEADING_SPLICE_RE = re.compile(r"^(?:\.\s*){2,}|^\u2026+\s*")


def _clean_chunk_opening(text: str) -> str:
    """Make chunk text open without a splice marker.

    Sentence packing always starts a chunk at a sentence, but source
    fragments (a paragraph continuing an omitted formula) and hard splits can
    leave dots at the very start.  The marker is dropped and the continuation
    is capitalized so the chunk reads as a clean opening.
    """
    value = _LEADING_SPLICE_RE.sub("", (text or "").lstrip())
    if value != text.lstrip() and value[:1].islower():
        value = value[0].upper() + value[1:]
    return value.lstrip()


def _clean_chunk_boundaries(text: str) -> str:
    """Remove only obvious splice punctuation at a chunk boundary."""
    value = _clean_chunk_opening(text)
    # PDF/math extraction can leave a sentence-ending dot as the first token.
    # Do not strip ordinary decimal points or punctuation inside the chunk.
    value = re.sub(r"^\.\s+(?=[A-Za-z])", "", value)
    return value.lstrip()


def _render_blocks(blocks: list[Block], heading_markup: bool) -> str:
    return blocks_to_text(
        blocks,
        heading_markup=heading_markup,
        math_policy="text",
        table_policy="markdown",
        caption_policy="keep",
        code_policy="keep",
    ).strip()


def _render_block(block: Block, heading_markup: bool) -> str:
    return _render_blocks([block], heading_markup)


def _split_sentences(text: str) -> list[str]:
    """Split prose at real sentence boundaries, not inside ellipses.

    Requiring an uppercase/numeric start avoids splitting quoted continuations
    such as ``"...factors... different regions"`` in the middle.
    """
    protected = text.replace("...", "\u2026")
    parts = _SENTENCE_SPLIT_RE.split(protected)
    return [p.replace("\u2026", "...").strip() for p in parts if p.strip()]


def _hard_split_text(
    text: str,
    block: Block,
    max_tokens: int,
    counter: TokenCounter,
    heading_markup: bool,
) -> list[str]:
    """Last-resort split that preserves every word/token of ``text``."""
    if not text:
        return []
    rough = max(1, max_tokens - 8)
    pieces = counter.split_parts(text, rough)
    if len(pieces) <= 1:
        pieces = counter.split_parts(text, max(1, rough // 2))
    out: list[str] = []
    for piece in pieces:
        probe = Block(type=block.type, text=piece, level=block.level, meta=dict(block.meta))
        if counter.count(_render_block(probe, heading_markup)) <= max_tokens:
            out.append(piece)
            continue
        if len(pieces) == 1:
            # Avoid infinite recursion on pathological input.
            out.extend(counter.split_parts(piece, max(1, rough // 2)))
        else:
            out.extend(_hard_split_text(piece, block, max_tokens, counter, heading_markup))
    return [p for p in out if p]


def _split_oversized_block(
    block: Block,
    max_tokens: int,
    counter: TokenCounter,
    heading_markup: bool,
) -> list[str]:
    """Split one block into text parts whose rendered form fits the budget."""
    if block.type in {"code", "list"}:
        units = block.text.splitlines() or [block.text]
        joiner = "\n"
    else:
        units = _split_sentences(block.text)
        joiner = " "

    def fits(text: str) -> bool:
        probe = Block(type=block.type, text=text, level=block.level, meta=dict(block.meta))
        return counter.count(_render_block(probe, heading_markup)) <= max_tokens

    parts: list[str] = []
    current: list[str] = []
    for unit in units:
        candidate = joiner.join(current + [unit])
        if current and not fits(candidate):
            parts.append(joiner.join(current))
            current = [unit]
        else:
            current.append(unit)
    if current:
        parts.append(joiner.join(current))

    out: list[str] = []
    for part in parts:
        if fits(part):
            out.append(part)
        else:
            out.extend(_hard_split_text(part, block, max_tokens, counter, heading_markup))
    return out


def _split_entry(
    blocks: list[Block],
    path: list[str],
    max_tokens: int,
    counter: TokenCounter,
    heading_markup: bool,
) -> list[tuple[list[Block], list[str]]]:
    """Split one packed chunk until every rendered chunk fits the budget."""
    if not blocks:
        return []
    if counter.count(_render_blocks(blocks, heading_markup)) <= max_tokens:
        return [(blocks, path)]

    # Never split a heading away from the first content block.  If the prefix
    # contains headings, reserve their tokens and split the following content
    # under the remaining budget, then attach the headings to its first part.
    if len(blocks) > 1 and blocks[0].type == "heading":
        heading_count = 0
        while heading_count < len(blocks) and blocks[heading_count].type == "heading":
            heading_count += 1
        heading_blocks = blocks[:heading_count]
        content = blocks[heading_count:]
        if content:
            heading_tokens = counter.count(_render_blocks(heading_blocks, heading_markup))
            rest_budget = max_tokens - heading_tokens - 8
            while rest_budget > 0:
                rest_entries = _split_entry(
                    content, path, rest_budget, counter, heading_markup,
                )
                if rest_entries:
                    combined = heading_blocks + rest_entries[0][0]
                    if counter.count(_render_blocks(combined, heading_markup)) <= max_tokens:
                        return [(combined, path)] + rest_entries[1:]
                rest_budget = max(0, rest_budget - max(1, rest_budget // 10))
    if len(blocks) > 1:
        for k in range(len(blocks) - 1, 0, -1):
            if counter.count(_render_blocks(blocks[:k], heading_markup)) <= max_tokens:
                return _split_entry(blocks[:k], path, max_tokens, counter, heading_markup) + \
                    _split_entry(blocks[k:], path, max_tokens, counter, heading_markup)
    block = blocks[0]
    parts = _split_oversized_block(block, max_tokens, counter, heading_markup)
    out: list[tuple[list[Block], list[str]]] = []
    for part in parts:
        out.append(([Block(type=block.type, text=part, level=block.level, meta=dict(block.meta))], path))
    if len(blocks) > 1:
        out.extend(_split_entry(blocks[1:], path, max_tokens, counter, heading_markup))
    return out


def _carry_trailing_headings(
    entries: list[tuple[list[Block], list[str]]],
) -> list[tuple[list[Block], list[str]]]:
    """Move a heading that lands at the end of a chunk to the next chunk."""
    out: list[tuple[list[Block], list[str]]] = []
    carry: list[Block] = []
    for blocks, path in entries:
        blocks = carry + list(blocks)
        carry = []
        while blocks and blocks[-1].type == "heading":
            carry.insert(0, blocks.pop())
        if blocks:
            out.append((blocks, path))
    # Any headings left at the very end have no content to introduce.
    return out


def _stabilize_chunks(
    entries: list[tuple[list[Block], list[str]]],
    *,
    max_tokens: int,
    counter: TokenCounter,
    heading_markup: bool,
) -> list[tuple[list[Block], list[str]]]:
    """Enforce the token budget and keep headings attached to their content."""
    for _ in range(4):
        entries = _carry_trailing_headings(entries)
        entries = [
            item
            for blocks, path in entries
            for item in _split_entry(blocks, path, max_tokens, counter, heading_markup)
        ]
        if all(
            counter.count(_render_blocks(blocks, heading_markup)) <= max_tokens
            and (not blocks or blocks[-1].type != "heading")
            for blocks, _ in entries
        ):
            break
    return entries


def _chunk_blocks(
    document: Document,
    *,
    max_tokens: int,
    overlap_tokens: int,
    min_tokens: int,
    counter: TokenCounter,
    heading_markup: bool = True,
) -> list[tuple[str, list[str]]]:
    blocks = list(getattr(document, "blocks", None) or [])
    if not blocks:
        blocks = [Block(type="paragraph", text=document.text)]

    active_headings: dict[int, str] = {}
    chunks: list[tuple[list[Block], list[str]]] = []
    current: list[Block] = []
    current_tokens = 0
    current_start_path: list[str] = []

    def current_path() -> list[str]:
        return [active_headings[level] for level in sorted(active_headings) if active_headings[level]]

    def flush() -> None:
        nonlocal current, current_tokens, current_start_path
        if not current:
            return
        text = blocks_to_text(
            current,
            heading_markup=heading_markup,
            math_policy="text",
            table_policy="markdown",
            caption_policy="keep",
            code_policy="keep",
        ).strip()
        if text:
            chunks.append((current, current_start_path))
        current = []
        current_tokens = 0
        current_start_path = []

    for block in blocks:
        if block.type in {"footnote", "navigation"}:
            continue
        if block.type == "heading":
            level = block.level or 2
            active_headings[level] = block.text
            for deeper in [k for k in active_headings if k > level]:
                del active_headings[deeper]
            if current:
                # Keep a heading with content, but flush if it would overflow.
                heading_text = blocks_to_text([block], heading_markup=heading_markup)
                heading_tokens = counter.count(heading_text)
                if current_tokens + heading_tokens > max_tokens:
                    flush()
            if not current:
                current_start_path = current_path()
            current.append(block)
            current_tokens += counter.count(blocks_to_text([block], heading_markup=heading_markup))
            continue

        rendered = blocks_to_text(
            [block],
            heading_markup=heading_markup,
            math_policy="text",
            table_policy="markdown",
            caption_policy="keep",
            code_policy="keep",
        )
        if not rendered:
            continue
        block_tokens = counter.count(rendered)
        if block_tokens > max_tokens:
            parts = _split_oversized_block(block, max_tokens, counter, heading_markup)
        else:
            parts = [block.text]

        for part in parts:
            part_block = Block(type=block.type, text=part, level=block.level, meta=dict(block.meta))
            part_tokens = counter.count(_render_block(part_block, heading_markup))
            if current and current_tokens + part_tokens > max_tokens:
                flush()
            if not current:
                current_start_path = current_path()
            current.append(part_block)
            current_tokens += part_tokens

    flush()

    # Never drop text to satisfy the token budget: split the packed chunk at a
    # block/sentence/line boundary instead, and keep headings with content.
    chunks = _stabilize_chunks(
        chunks,
        max_tokens=max_tokens,
        counter=counter,
        heading_markup=heading_markup,
    )

    # Apply overlap and drop empty/duplicate base chunks.
    out: list[tuple[str, list[str]]] = []
    previous = ""
    for blocks_in_chunk, path in chunks:
        text = _render_blocks(blocks_in_chunk, heading_markup)
        if not text:
            continue
        if overlap_tokens > 0 and previous:
            allowed = max(0, max_tokens - counter.count(text))
            overlap = min(overlap_tokens, allowed)
            if overlap > 0:
                tail = counter.tail(previous, overlap).strip()
                # Overlap is context only: never begin with a code fence or a
                # heading, and only start at a real sentence boundary.
                if "```" in tail or re.search(r"(?m)^#", tail):
                    tail = ""
                elif boundary := re.search(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])", tail):
                    tail = tail[boundary.end():].strip()
                else:
                    tail = ""
                if tail:
                    candidate = tail + "\n\n" + text
                    if counter.count(candidate) <= max_tokens:
                        text = candidate
        if text == previous:
            continue
        out.append((text, path))
        previous = text

    # Merge tiny chunks into the previous chunk when possible.
    merged: list[tuple[str, list[str]]] = []
    for text, path in out:
        if merged and counter.count(text) < min_tokens:
            prev_text, prev_path = merged[-1]
            combined = prev_text + "\n\n" + text
            if counter.count(combined) <= max_tokens:
                merged[-1] = (combined, prev_path)
                continue
        merged.append((text, path))

    # If a tiny chunk cannot fit behind its predecessor, try folding it into
    # the following chunk.  This mostly helps short transitional paragraphs.
    balanced: list[tuple[str, list[str]]] = []
    index = 0
    while index < len(merged):
        text, path = merged[index]
        if (
            counter.count(text) < min_tokens
            and index + 1 < len(merged)
            and counter.count(text + "\n\n" + merged[index + 1][0]) <= max_tokens
        ):
            balanced.append((text + "\n\n" + merged[index + 1][0], path))
            index += 2
            continue
        balanced.append((text, path))
        index += 1
    return balanced


def chunk_document(
    document: Document,
    *,
    max_tokens: int = 1024,
    overlap_tokens: int = 64,
    min_tokens: int = 128,
    tokenizer: str = "tiktoken:cl100k_base",
    splits: dict | None = None,
    split: str | None = None,
    schema_version: str = "2.0",
    counter: TokenCounter | None = None,
) -> list[Chunk]:
    counter = counter or TokenCounter(tokenizer)
    assigned_split = split or assign_split(document, splits)
    parts = _chunk_blocks(
        document,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        min_tokens=min_tokens,
        counter=counter,
        heading_markup=True,
    )
    chunks: list[Chunk] = []
    for index, (text, path) in enumerate(parts):
        text = _clean_chunk_boundaries(text)
        chunk_id = f"{document.id}-c{index:04d}"
        chunks.append(Chunk(
            id=chunk_id,
            doc_id=document.id,
            text=text,
            title=document.title,
            url=document.url,
            source=document.source,
            language=document.language,
            license=document.license,
            fetched_at=document.fetched_at,
            heading_path=path,
            chunk_index=index,
            token_count=counter.count(text),
            quality_score=document.quality_score,
            split=assigned_split,
            content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            schema_version=schema_version,
            difficulty=getattr(document, "difficulty", None),
            curriculum_rank=getattr(document, "curriculum_rank", None),
            curriculum_level=getattr(document, "curriculum_level", None),
        ))
    return chunks


def chunk_documents(
    documents: Iterable[Document],
    *,
    max_tokens: int = 1024,
    overlap_tokens: int = 64,
    min_tokens: int = 128,
    tokenizer: str = "tiktoken:cl100k_base",
    splits: dict | None = None,
    schema_version: str = "2.0",
) -> list[Chunk]:
    counter = TokenCounter(tokenizer)
    kept_docs = [d for d in documents if getattr(d, "kept", True)]
    sizes = {
        getattr(d, "id", ""): counter.count(getattr(d, "text", "") or "")
        for d in kept_docs
    }
    split_map = assign_splits(kept_docs, splits, sizes=sizes)
    chunks: list[Chunk] = []
    for doc in kept_docs:
        chunks.extend(chunk_document(
            doc,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            min_tokens=min_tokens,
            tokenizer=tokenizer,
            splits=splits,
            split=split_map.get(getattr(doc, "id", "")),
            schema_version=schema_version,
            counter=counter,
        ))
    return chunks


def write_chunks(
    chunks: Iterable[Chunk],
    *,
    output_dir: str | Path,
    output_format: str = "jsonl",
    shard_size_records: int = 10_000,
) -> list[str]:
    """Write chunks grouped by split, then sharded."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Remove shards from an earlier run so a split that disappeared cannot
    # leave stale training data behind.
    for pattern in ("chunks_*.jsonl", "chunks_*.parquet"):
        for old in output_dir.glob(pattern):
            old.unlink()
    by_split: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        by_split.setdefault(chunk.split, []).append(chunk)

    written: list[str] = []
    for split, items in sorted(by_split.items()):
        shard_size = max(1, int(shard_size_records))
        for i in range(0, len(items), shard_size):
            shard = items[i:i + shard_size]
            if output_format == "parquet":
                try:
                    import pyarrow as pa
                    import pyarrow.parquet as pq
                except Exception as exc:
                    raise RuntimeError("Parquet output requires pyarrow") from exc
                path = output_dir / f"chunks_{split}_{i // shard_size:05d}.parquet"
                pq.write_table(pa.Table.from_pylist([c.to_dict() for c in shard]), path)
            else:
                path = output_dir / f"chunks_{split}_{i // shard_size:05d}.jsonl"
                with path.open("w", encoding="utf-8") as f:
                    for chunk in shard:
                        f.write(json.dumps(chunk.to_dict(), ensure_ascii=False))
                        f.write("\n")
            written.append(str(path))
    return written
