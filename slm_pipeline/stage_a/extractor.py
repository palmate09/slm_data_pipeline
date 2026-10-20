"""TopicDocumentExtractor: HTML/PDF/text -> structured Document blocks."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ..models import Document, new_id, now_iso
from .html_extract import (
    HtmlExtraction,
    blocks_to_text,
    extract_html,
    extract_pdf,
    extract_plain_text,
)


@dataclass
class ExtractionResult:
    document: Document
    ok: bool
    error: str | None = None


class TopicDocumentExtractor:
    """Extracts title + structured main-content blocks from HTML, PDF or text.

    The extractor is intentionally conservative: it produces typed blocks and
    records which extraction method won, so later stages can reject a
    low-confidence extraction instead of silently training on navigation text.
    """

    def __init__(
        self,
        source: str = "web",
        *,
        extractor: str = "auto",
        min_confidence: float = 0.25,
        heading_markup: bool = True,
        math_policy: str = "text",
        table_policy: str = "drop",
        caption_policy: str = "drop",
        code_policy: str = "keep",
    ) -> None:
        self.source = source
        self.extractor = extractor
        self.min_confidence = min_confidence
        self.heading_markup = heading_markup
        self.math_policy = math_policy
        self.table_policy = table_policy
        self.caption_policy = caption_policy
        self.code_policy = code_policy

    # ------------------------------------------------------------------ public
    def extract_file(
        self,
        *,
        file_path: str,
        url: str,
        content_type: str,
        doc_id: str | None = None,
        title: str | None = None,
        fetched_at: str | None = None,
    ) -> ExtractionResult:
        path = Path(file_path)
        try:
            raw_bytes = path.read_bytes()
            raw_hash = hashlib.sha256(raw_bytes).hexdigest()
            if content_type == "pdf" or path.suffix.lower() == ".pdf":
                extraction = extract_pdf(path, url=url, title=title or "")
            else:
                html = raw_bytes.decode("utf-8", errors="replace")
                extraction = extract_html(
                    html,
                    url=url,
                    requested_extractor=self.extractor,
                    min_confidence=self.min_confidence,
                )
            return self._to_result(
                extraction,
                url=url,
                doc_id=doc_id,
                fallback_title=title,
                fetched_at=fetched_at,
                raw_hash=raw_hash,
            )
        except Exception as exc:  # noqa: BLE001
            return ExtractionResult(
                document=self._empty_document(url, doc_id, title, fetched_at),
                ok=False,
                error=str(exc),
            )

    def extract_records(self, records: Iterable) -> list[ExtractionResult]:
        results: list[ExtractionResult] = []
        for r in records:
            results.append(self.extract_file(
                file_path=r.file_path, url=r.url, content_type=r.content_type,
                doc_id=r.id, title=None, fetched_at=r.fetched_at,
            ))
        return results

    def extract_text(
        self,
        *,
        text: str,
        url: str,
        content_type: str = "html",
        doc_id: str | None = None,
        title: str | None = None,
        fetched_at: str | None = None,
    ) -> ExtractionResult:
        """Extract from an in-memory string (useful for tests / Stage A API)."""
        if content_type == "pdf" or content_type in {"text", "markdown"}:
            extraction = extract_plain_text(text, url=url, title=title or "")
        else:
            extraction = extract_html(
                text,
                url=url,
                requested_extractor=self.extractor,
                min_confidence=self.min_confidence,
            )
        return self._to_result(
            extraction,
            url=url,
            doc_id=doc_id,
            fallback_title=title,
            fetched_at=fetched_at,
            raw_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )

    # ------------------------------------------------------------------ internals
    def _to_result(
        self,
        extraction: HtmlExtraction,
        *,
        url: str,
        doc_id: str | None,
        fallback_title: str | None,
        fetched_at: str | None,
        raw_hash: str,
    ) -> ExtractionResult:
        title = extraction.title or fallback_title or ""
        rendered = blocks_to_text(
            extraction.blocks,
            heading_markup=self.heading_markup,
            math_policy=self.math_policy,
            table_policy=self.table_policy,
            caption_policy=self.caption_policy,
            code_policy=self.code_policy,
        )
        ok = bool(rendered.strip()) and extraction.error is None
        document = Document(
            id=doc_id or new_id(),
            text=rendered.strip(),
            url=url,
            title=title,
            source=self.source,
            fetched_at=fetched_at or now_iso(),
            blocks=extraction.blocks,
            language=extraction.language,
            license=extraction.license,
            canonical_url=extraction.canonical_url,
            content_hash=raw_hash,
            extraction={
                "extractor": extraction.extractor,
                "confidence": extraction.confidence,
                "candidates": extraction.candidates,
                "site_name": extraction.site_name,
                "block_count": len(extraction.blocks),
            },
            stage="extracted",
        )
        if not ok:
            document.kept = False
            document.drop_reason = extraction.error or "no extractable text"
        return ExtractionResult(document=document, ok=ok, error=None if ok else document.drop_reason)

    @staticmethod
    def _empty_document(url: str, doc_id: str | None, title: str | None,
                        fetched_at: str | None) -> Document:
        return Document(
            id=doc_id or new_id(),
            text="",
            url=url,
            title=title or "",
            source="web",
            fetched_at=fetched_at or now_iso(),
            stage="extracted", kept=False, drop_reason="no extractable text",
        )


def extract_document(*, file_path: str | None = None, text: str | None = None,
                     url: str = "", content_type: str = "html",
                     doc_id: str | None = None, title: str | None = None,
                     source: str = "web", fetched_at: str | None = None,
                     extractor: str = "auto",
                     min_confidence: float = 0.25,
                     ) -> ExtractionResult:
    extract = TopicDocumentExtractor(
        source=source, extractor=extractor, min_confidence=min_confidence,
    )
    if file_path is not None:
        return extract.extract_file(
            file_path=file_path, url=url, content_type=content_type,
            doc_id=doc_id, title=title, fetched_at=fetched_at,
        )
    return extract.extract_text(
        text=text or "", url=url, content_type=content_type,
        doc_id=doc_id, title=title, fetched_at=fetched_at,
    )
