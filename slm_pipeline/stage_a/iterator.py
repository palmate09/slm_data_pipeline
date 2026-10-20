"""TopicDocumentIterator: read downloaded files into raw records."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RawRecord:
    """Pre-extraction record: points at a downloaded file plus provenance."""

    id: str
    url: str
    source: str
    fetched_at: str
    content_type: str  # "html" | "pdf"
    file_path: str  # path on disk to the downloaded file


class TopicDocumentIterator:
    """Iterates over downloaded/local files and yields :class:`RawRecord`s.

    ``source_filter`` prevents a run for topic A from accidentally reading
    cached files downloaded for topic B.
    """

    def __init__(self, download_dir: str = "data/downloads", source: str | None = None) -> None:
        self.download_dir = Path(download_dir)
        self.source_filter = source

    def iter_files(self) -> Iterable[RawRecord]:
        if not self.download_dir.exists():
            return
        for path in sorted(self.download_dir.iterdir()):
            if not path.is_file():
                continue
            if path.suffix.lower() not in (".html", ".pdf"):
                continue
            record = self._file_to_record(path)
            if self.source_filter and record.source not in {"", "web", self.source_filter}:
                continue
            yield record

    def iter_paths(self, paths: Iterable[str | Path]) -> Iterable[RawRecord]:
        for value in paths:
            path = Path(value)
            if not path.is_file():
                continue
            if path.suffix.lower() not in (".html", ".pdf"):
                continue
            yield self._file_to_record(path)

    def _file_to_record(self, path: Path) -> RawRecord:
        content_type = "pdf" if path.suffix.lower() == ".pdf" else "html"
        meta = {}
        sidecar = path.with_suffix(path.suffix + ".meta.json")
        if sidecar.exists():
            try:
                meta = json.loads(sidecar.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                meta = {}
        return RawRecord(
            id=meta.get("id", ""),
            url=meta.get("url", ""),
            source=meta.get("source", self.source_filter or "web"),
            fetched_at=meta.get("fetched_at", ""),
            content_type=content_type,
            file_path=str(path),
        )
