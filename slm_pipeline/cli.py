"""CLI entry point for the SLM data curation pipeline.

Legacy pipeline runs:
    python -m slm_pipeline.cli --config configs/machine_learning.yaml --stage all
    python -m slm_pipeline.cli --config configs/machine_learning.yaml --stage clean --resume-from gate

Inspection commands:
    python -m slm_pipeline.cli audit    --config configs/machine_learning.yaml
    python -m slm_pipeline.cli preview  --config configs/machine_learning.yaml --kind chunks
    python -m slm_pipeline.cli validate --config configs/machine_learning.yaml
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from .config import PipelineConfig, load_config
from .pipeline import STAGE_ORDER, run_pipeline
from .stage_c.artifacts import scan_artifacts
from .stage_f.chunking import TokenCounter

# --------------------------------------------------------------------------- helpers


def _output_dir(config: PipelineConfig, override: str | None = None) -> Path:
    return Path(override or config.resolved_output_dir)


def _read_records(paths: list[Path]) -> list[dict]:
    records: list[dict] = []
    for path in paths:
        if path.suffix == ".parquet":
            try:
                import pyarrow.parquet as pq
            except ImportError as exc:  # pragma: no cover - optional
                raise RuntimeError("Reading Parquet output requires pyarrow") from exc
            records.extend(pq.read_table(path).to_pylist())
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def _first_existing(output_dir: Path, stem: str) -> list[Path]:
    for pattern in (f"{stem}_*.jsonl", f"{stem}_*.parquet"):
        paths = sorted(output_dir.glob(pattern))
        if paths:
            return paths
    return []


def _read_documents(output_dir: Path) -> list[dict]:
    return _read_records(_first_existing(output_dir, "documents"))


def _read_chunks(output_dir: Path) -> list[dict]:
    return _read_records(_first_existing(output_dir / "chunks", "chunks"))


def _load_manifest(output_dir: Path) -> dict:
    path = output_dir / "manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _curriculum_is_ordered(docs: list[dict]) -> bool:
    """True when documents are written from base knowledge to advanced."""
    ranks = [d.get("curriculum_rank") for d in docs]
    if any(rank is None for rank in ranks):
        return False
    return all(a <= b for a, b in zip(ranks, ranks[1:]))


def _artifact_summary(records: list[dict]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        for name, value in scan_artifacts(record.get("text", "")).items():
            counts[name] += value
    return dict(counts)


def _load_config_or_error(path: str) -> PipelineConfig | None:
    try:
        return load_config(path)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return None


# --------------------------------------------------------------------------- structure


_HEADING_LINE_RE = re.compile(r"^\s{0,3}#{1,6}\s+\S")
_STRUCTURAL_START_RE = re.compile(
    r"^(?:```|~~~|#{1,6}\s|[-*+]\s|\d+[.)]\s|>\s)"
)
_CLOSING_PUNCTUATION = ",;:.)]}"


def _chunks_by_split(chunks: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.get("split", "unknown"), []).append(chunk)
    return grouped


def _structure_issues(text: str) -> list[str]:
    """Return structural problems in final document/chunk text.

    These are the failure modes that make a chunk look spliced even when the
    token count and schema are valid: an unbalanced fence, a chunk that ends on
    a heading, and a chunk that begins mid-sentence.
    """
    issues: list[str] = []
    stripped = text.strip()
    if not stripped:
        return ["empty text"]
    lines = [line for line in stripped.splitlines() if line.strip()]
    if not lines:
        return ["empty text"]

    fence_lines = [line for line in lines if line.lstrip().startswith(("```", "~~~"))]
    if len(fence_lines) % 2:
        issues.append("unbalanced code fence")
    if _HEADING_LINE_RE.match(lines[-1]):
        issues.append("ends with a dangling heading")

    first = lines[0].lstrip()
    if not _STRUCTURAL_START_RE.match(first):
        probe = first.lstrip("\"'([{")
        if probe[:1].islower():
            issues.append("starts with lowercase text")
        if first[:1] in _CLOSING_PUNCTUATION:
            issues.append(f"starts with closing punctuation {first[:1]!r}")
        elif probe[:1] and probe[:1] in _CLOSING_PUNCTUATION:
            issues.append(f"starts with closing punctuation {probe[:1]!r}")
    return issues


# --------------------------------------------------------------------------- audit


def _audit(args: argparse.Namespace) -> int:
    config = _load_config_or_error(args.config)
    if config is None:
        return 2
    out = _output_dir(config, args.output_dir)
    docs = _read_documents(out)
    chunks = _read_chunks(out)
    manifest = _load_manifest(out)

    report = {
        "output_dir": str(out),
        "documents": len(docs),
        "chunks": len(chunks),
        "document_words": sum(len(d.get("text", "").split()) for d in docs),
        "chunk_tokens": sum(int(c.get("token_count", 0)) for c in chunks),
        "splits": dict(Counter(c.get("split", "unknown") for c in chunks)),
        "document_artifacts": _artifact_summary(docs),
        "chunk_artifacts": _artifact_summary(chunks),
        "documents_with_blocks": sum("blocks" in d for d in docs),
        "chunks_over_budget": sum(
            int(c.get("token_count", 0)) > config.chunk_max_tokens for c in chunks
        ),
        "chunks_with_structure_issues": sum(
            bool(_structure_issues(c.get("text", ""))) for c in chunks
        ),
        "documents_with_structure_issues": sum(
            bool(_structure_issues(d.get("text", ""))) for d in docs
        ),
        "missing_chunk_fields": sum(
            any(k not in c for k in ("id", "doc_id", "text", "token_count", "split", "content_hash"))
            for c in chunks
        ),
        "manifest_documents": manifest.get("stats", {}).get("documents_written"),
        "manifest_chunks": manifest.get("stats", {}).get("chunks_written"),
        "drop_reasons": manifest.get("stats", {}).get("drop_reasons", {}),
        "curriculum_levels": dict(Counter(
            d.get("curriculum_level") for d in docs if d.get("curriculum_level")
        )),
        "documents_ordered_by_curriculum": _curriculum_is_ordered(docs),
    }
    if chunks:
        report["chunk_tokens_min"] = min(int(c.get("token_count", 0)) for c in chunks)
        report["chunk_tokens_max"] = max(int(c.get("token_count", 0)) for c in chunks)
        report["chunk_tokens_avg"] = round(
            report["chunk_tokens"] / len(chunks), 2
        )

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    print(f"=== Audit: {out} ===")
    print(f"Documents: {report['documents']}  ({report['document_words']} words)")
    print(f"Chunks:    {report['chunks']}  ({report['chunk_tokens']} tokens)")
    if chunks:
        print(f"Chunk size: min={report['chunk_tokens_min']}  "
              f"max={report['chunk_tokens_max']}  avg={report['chunk_tokens_avg']}")
        print(f"Splits: {report['splits']}")
    print(f"Artifacts (docs):   {report['document_artifacts'] or '{}'}")
    print(f"Artifacts (chunks): {report['chunk_artifacts'] or '{}'}")
    print(f"Documents with blocks: {report['documents_with_blocks']}")
    print(f"Chunks over token budget: {report['chunks_over_budget']}")
    print(f"Structural issues: docs={report['documents_with_structure_issues']} "
          f"chunks={report['chunks_with_structure_issues']}")
    print(f"Manifest counts: docs={report['manifest_documents']} chunks={report['manifest_chunks']}")
    if report["drop_reasons"]:
        print("Drop reasons:")
        for reason, count in sorted(report["drop_reasons"].items(), key=lambda kv: -kv[1]):
            print(f"  {count:>3}  {reason}")

    if report["curriculum_levels"]:
        print("\nCurriculum (base knowledge -> advanced):")
        for doc in sorted(
            (d for d in docs if d.get("curriculum_rank")),
            key=lambda d: d["curriculum_rank"],
        ):
            print(f"  {doc['curriculum_rank']:>3}. {doc.get('difficulty', 0):.2f} "
                  f"{doc.get('curriculum_level', ''):<14} {doc.get('title', '')[:60]}")
        print(f"  levels: {report['curriculum_levels']}"
              f"  ordered={report['documents_ordered_by_curriculum']}")

    print("\nTop documents:")
    for doc in sorted(docs, key=lambda d: d.get("quality_score") or 0, reverse=True)[:args.limit]:
        print(f"  {doc.get('quality_score', 0):.3f}  {len(doc.get('text', '').split()):>6}w  "
              f"{doc.get('title', '')[:80]}")
    if chunks:
        print("\nSample chunks:")
        for chunk in chunks[:args.limit]:
            print(f"  {chunk.get('token_count', 0):>5}t  {chunk.get('id', '')[:60]}  "
                  f"{chunk.get('heading_path', [])}")
    return 0


# --------------------------------------------------------------------------- preview


def _preview(args: argparse.Namespace) -> int:
    config = _load_config_or_error(args.config)
    if config is None:
        return 2
    out = _output_dir(config, args.output_dir)
    records = _read_documents(out) if args.kind == "documents" else _read_chunks(out)
    if args.index is not None:
        records = [records[args.index]] if 0 <= args.index < len(records) else []
    records = records[:args.limit]
    if args.json:
        print(json.dumps(records, indent=2, ensure_ascii=False))
        return 0
    if not records:
        print(f"No {args.kind} found in {out}", file=sys.stderr)
        return 1
    for index, record in enumerate(records):
        print("=" * 100)
        if args.kind == "documents":
            print(f"[{index}] {record.get('title', '')}")
            print(f"id={record.get('id')}  words={len(record.get('text', '').split())}  "
                  f"quality={record.get('quality_score')}  url={record.get('url')}")
        else:
            print(f"[{index}] {record.get('id')}  tokens={record.get('token_count')}  "
                  f"split={record.get('split')}  heading_path={record.get('heading_path')}")
            print(f"doc_id={record.get('doc_id')}  title={record.get('title')}")
        text = record.get("text", "")
        print(text[:args.chars])
        if len(text) > args.chars:
            print(f"\n... ({len(text) - args.chars} characters omitted)")
    return 0


# --------------------------------------------------------------------------- validate


def _validate(args: argparse.Namespace) -> int:
    config = _load_config_or_error(args.config)
    if config is None:
        return 2
    out = _output_dir(config, args.output_dir)
    docs = _read_documents(out)
    chunks = _read_chunks(out)
    manifest = _load_manifest(out)
    errors: list[str] = []
    warnings: list[str] = []

    if not docs:
        errors.append(f"no document shards found in {out}")
    if config.write_chunks and not chunks:
        errors.append(f"no chunk shards found in {out / 'chunks'}")
    if not manifest:
        warnings.append(f"manifest not found at {out / 'manifest.json'}")

    doc_ids: set[str] = set()
    required_doc = {"id", "text", "title", "url", "source"}
    hard_artifacts = {
        "displaystyle", "citation_tag", "html_tag", "wiki_template", "wiki_link",
        "control_chars", "mojibake",
    }
    for index, doc in enumerate(docs):
        label = f"document[{index}] {doc.get('id', '?')}"
        missing = required_doc - set(doc)
        if missing:
            errors.append(f"{label}: missing fields {sorted(missing)}")
        if "blocks" in doc:
            errors.append(f"{label}: intermediate 'blocks' field must not be in final output")
        if not doc.get("text", "").strip():
            errors.append(f"{label}: empty text")
        doc_id = doc.get("id")
        if doc_id:
            if doc_id in doc_ids:
                errors.append(f"{label}: duplicate document id")
            doc_ids.add(doc_id)
        artifacts = scan_artifacts(doc.get("text", ""))
        for name in hard_artifacts:
            if artifacts.get(name):
                errors.append(f"{label}: artifact {name}={artifacts[name]}")
        if artifacts.get("url"):
            warnings.append(f"{label}: {artifacts['url']} URL(s) in text")
        for issue in _structure_issues(doc.get("text", "")):
            errors.append(f"{label}: {issue}")

    counter = TokenCounter(config.tokenizer)
    chunk_sequences: dict[str, list[int]] = {}
    chunk_splits: dict[str, set[str]] = {}
    for index, chunk in enumerate(chunks):
        label = f"chunk[{index}] {chunk.get('id', '?')}"
        required_chunk = {
            "id", "doc_id", "text", "token_count", "split", "content_hash",
            "heading_path", "chunk_index",
        }
        missing = required_chunk - set(chunk)
        if missing:
            errors.append(f"{label}: missing fields {sorted(missing)}")
            continue
        text = chunk.get("text", "")
        if not text.strip():
            errors.append(f"{label}: empty text")
        if chunk.get("doc_id") not in doc_ids:
            errors.append(f"{label}: unknown doc_id {chunk.get('doc_id')}")
        actual_tokens = counter.count(text)
        if actual_tokens != int(chunk.get("token_count", -1)):
            errors.append(
                f"{label}: token_count {chunk.get('token_count')} != counted {actual_tokens}"
            )
        if actual_tokens > config.chunk_max_tokens:
            errors.append(
                f"{label}: {actual_tokens} tokens exceeds chunk_max_tokens={config.chunk_max_tokens}"
            )
        if actual_tokens < config.chunk_min_tokens:
            warnings.append(
                f"{label}: {actual_tokens} tokens below chunk_min_tokens={config.chunk_min_tokens}"
            )
        import hashlib

        expected_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if chunk.get("content_hash") != expected_hash:
            errors.append(f"{label}: content_hash mismatch")
        if not isinstance(chunk.get("heading_path"), list):
            errors.append(f"{label}: heading_path must be a list")
        split = chunk.get("split")
        if split not in (config.splits or {"train": 1.0}):
            errors.append(f"{label}: unknown split {split!r}")
        doc_id = chunk.get("doc_id", "")
        chunk_sequences.setdefault(doc_id, []).append(int(chunk.get("chunk_index", -1)))
        chunk_splits.setdefault(doc_id, set()).add(split)
        artifacts = scan_artifacts(text)
        for name in hard_artifacts:
            if artifacts.get(name):
                errors.append(f"{label}: artifact {name}={artifacts[name]}")
        if artifacts.get("url"):
            warnings.append(f"{label}: {artifacts['url']} URL(s) in text")
        for issue in _structure_issues(text):
            errors.append(f"{label}: {issue}")

    for split, items in _chunks_by_split(chunks).items():
        ranks = [c.get("curriculum_rank") for c in items]
        if any(rank is None for rank in ranks):
            continue
        if any(a > b for a, b in zip(ranks, ranks[1:])):
            errors.append(
                f"split {split}: chunks are not ordered from base knowledge to advanced"
            )

    for doc_id, sequence in chunk_sequences.items():
        if sequence != list(range(len(sequence))):
            errors.append(f"doc {doc_id}: chunk indexes are not consecutive: {sequence}")
    for doc_id, splits in chunk_splits.items():
        if len(splits) > 1:
            errors.append(f"doc {doc_id}: chunks assigned to multiple splits {sorted(splits)}")

    if manifest:
        stats = manifest.get("stats", {})
        if stats.get("documents_written") is not None and stats["documents_written"] != len(docs):
            errors.append(
                f"manifest documents_written={stats['documents_written']} != actual {len(docs)}"
            )
        if stats.get("chunks_written") is not None and stats["chunks_written"] != len(chunks):
            errors.append(
                f"manifest chunks_written={stats['chunks_written']} != actual {len(chunks)}"
            )
        for path in manifest.get("files", []) + manifest.get("chunk_files", []):
            if not Path(path).exists():
                errors.append(f"manifest references missing file: {path}")

    if args.strict:
        errors.extend(warnings)
        warnings = []

    print(f"=== Validate: {out} ===")
    print(f"Documents: {len(docs)}  Chunks: {len(chunks)}")
    if warnings:
        print(f"Warnings ({len(warnings)}):")
        for warning in warnings:
            print(f"  WARN  {warning}")
    if errors:
        print(f"Errors ({len(errors)}):")
        for error in errors:
            print(f"  FAIL  {error}")
        return 1
    print("OK: schema, token bounds, hashes, splits and artifacts are valid.")
    return 0


# --------------------------------------------------------------------------- legacy pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SLM Data Collection & Cleaning Pipeline",
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to topic_config.yaml",
    )
    parser.add_argument(
        "--stage", choices=["all"] + STAGE_ORDER, default="all",
        help="Pipeline stage to run (default: all)",
    )
    parser.add_argument(
        "--resume-from", choices=STAGE_ORDER, default=None,
        help="Resume pipeline from this stage (loads previous stage's intermediate)",
    )
    return parser


def _legacy_main(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = _load_config_or_error(args.config)
    if config is None:
        return 2

    result = run_pipeline(
        config,
        stage=args.stage,  # type: ignore[arg-type]
        resume_from=args.resume_from,
    )

    print("\n=== Pipeline complete ===")
    print(f"Topic: {config.topic}")
    print(f"Elapsed: {result.elapsed_seconds}s")
    print(f"Final records: {len(result.final_dataset) if result.final_dataset else 0}")
    print(f"Output files: {result.output_files}")
    if result.chunk_files:
        print(f"Chunk files: {len(result.chunk_files)}")
        print(f"Chunks: {len(result.chunks)} ({sum(c.token_count for c in result.chunks)} tokens)")
    return 0


# --------------------------------------------------------------------------- command parsing


def _audit_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="slm-pipeline audit", description="Audit final docs/chunks")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--json", action="store_true")
    return parser


def _preview_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="slm-pipeline preview", description="Preview docs/chunks")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--kind", choices=["documents", "chunks"], default="documents")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--chars", type=int, default=1200)
    parser.add_argument("--json", action="store_true")
    return parser


def _validate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="slm-pipeline validate", description="Validate final output")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--strict", action="store_true", help="Treat warnings as errors")
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        if argv and argv[0] in {"audit", "preview", "validate"}:
            command = argv.pop(0)
            parser = {
                "audit": _audit_parser,
                "preview": _preview_parser,
                "validate": _validate_parser,
            }[command]()
            args = parser.parse_args(argv)
            return {
                "audit": _audit,
                "preview": _preview,
                "validate": _validate,
            }[command](args)
        return _legacy_main(argv)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
