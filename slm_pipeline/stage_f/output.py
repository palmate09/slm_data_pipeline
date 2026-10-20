"""Stage F: output documents and tokenized chunks, with a manifest."""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from ..dataset import DocumentDataset
from .chunking import Chunk, write_chunks


def _write_records(
    records: list[dict],
    path: Path,
    output_format: str,
) -> None:
    if output_format == "parquet":
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except Exception as exc:
            raise RuntimeError("Parquet output requires pyarrow") from exc
        pq.write_table(pa.Table.from_pylist(records), path)
    else:
        with path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False))
                f.write("\n")


def write_dataset(
    dataset: DocumentDataset,
    *,
    output_dir: str | Path,
    output_format: Literal["jsonl", "parquet"] = "jsonl",
    shard_size_records: int = 10_000,
    prefix: str = "documents",
    schema_version: str = "2.0",
    include_blocks: bool = False,
) -> list[str]:
    """Write curated documents to sharded files.

    Final output intentionally excludes the intermediate ``blocks`` column by
    default; it is large and not needed for training.  Pass
    ``include_blocks=True`` for a debugging export.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ext = "jsonl" if output_format == "jsonl" else "parquet"
    # Drop old document shards (including legacy names) so a shorter run does
    # not leave unreferenced files in the curated output directory.
    for pattern in (f"{prefix}_*.{ext}", "shard_*.jsonl", "shard_*.parquet"):
        for old in output_dir.glob(pattern):
            old.unlink()

    docs = [d for d in dataset if d.kept]
    if not docs:
        return []

    shard_size = max(1, int(shard_size_records))
    written: list[str] = []
    for i in range(0, len(docs), shard_size):
        shard = docs[i:i + shard_size]
        records = []
        for doc in shard:
            record = doc.to_dict(include_blocks=include_blocks)
            record["schema_version"] = schema_version
            records.append(record)
        path = output_dir / f"{prefix}_{i // shard_size:05d}.{ext}"
        _write_records(records, path, output_format)
        written.append(str(path))
    return written


def write_chunk_files(
    chunks: Iterable[Chunk],
    *,
    output_dir: str | Path,
    output_format: str = "jsonl",
    shard_size_records: int = 10_000,
) -> list[str]:
    return write_chunks(
        chunks,
        output_dir=output_dir,
        output_format=output_format,
        shard_size_records=shard_size_records,
    )


def write_manifest(
    output_dir: str | Path,
    files: list[str],
    stats: dict,
    *,
    chunk_files: list[str] | None = None,
    schema_version: str = "2.0",
    config_hash: str | None = None,
) -> str:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": schema_version,
        "num_files": len(files) + len(chunk_files or []),
        "files": files,
        "chunk_files": chunk_files or [],
        "stats": stats,
    }
    if config_hash:
        manifest["config_hash"] = config_hash
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)
