"""TopicDocumentDownloader: fetch URLs to disk, respectfully."""

from __future__ import annotations

import json
import mimetypes
import re
import time
import urllib.parse
import urllib.robotparser
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import requests

from ..models import new_id, now_iso


@dataclass
class DownloadResult:
    local_path: str
    url: str
    source: str
    content_type: str  # "html" | "pdf" | "other"
    status: str  # "ok" | "skipped" | "error"
    error: str | None = None


class TopicDocumentDownloader:
    """Downloads HTML/PDF documents for a list of URLs.

    Features:
      * per-request delay (rate limiting)
      * retry with backoff
      * optional robots.txt honoring
      * content-type gating (HTML & PDF only)
      * filename derived from URL hash so re-runs are idempotent
    """

    EXT_BY_CONTENT_TYPE: ClassVar[dict[str, str]] = {
        "text/html": "html",
        "application/html": "html",
        "application/pdf": "pdf",
    }

    def __init__(
        self,
        *,
        download_dir: str = "data/downloads",
        request_delay_seconds: float = 1.0,
        max_retries: int = 3,
        request_timeout_seconds: int = 30,
        user_agent: str = "slm-data-pipeline/0.1",
        respect_robots_txt: bool = True,
    ) -> None:
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.request_delay_seconds = request_delay_seconds
        self.max_retries = max_retries
        self.request_timeout_seconds = request_timeout_seconds
        self.user_agent = user_agent
        self.respect_robots_txt = respect_robots_txt
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": user_agent})
        self._last_request_time = 0.0
        self._robots_cache: dict[str, urllib.robotparser.RobotFileParser] = {}

    # ------------------------------------------------------------------ public
    def download_urls(self, urls: Iterable[str], source: str = "web") -> list[DownloadResult]:
        results: list[DownloadResult] = []
        for url in urls:
            url = url.strip()
            if not url or url.startswith("#"):
                continue
            results.append(self.download_one(url, source=source))
        return results

    def download_one(self, url: str, source: str = "web") -> DownloadResult:
        if self.respect_robots_txt and not self._allowed_by_robots(url):
            return DownloadResult(
                local_path="", url=url, source=source, content_type="other",
                status="skipped", error="disallowed by robots.txt",
            )

        # Determine expected type from URL extension as a cheap pre-filter.
        guessed = mimetypes.guess_type(url)[0]
        url_ext = Path(urllib.parse.urlparse(url).path).suffix.lower()
        is_pdf = url_ext == ".pdf" or guessed == "application/pdf"
        is_html = not is_pdf and (guessed is None or "text" in (guessed or ""))

        local_path = self._cache_path(url, is_pdf)
        if local_path.exists():
            return DownloadResult(
                local_path=str(local_path), url=url, source=source,
                content_type="pdf" if is_pdf else "html", status="skipped",
                error="already cached",
            )

        content_type = "other"
        for attempt in range(1, self.max_retries + 1):
            try:
                self._rate_limit()
                resp = self._session.get(
                    url, timeout=self.request_timeout_seconds, allow_redirects=True
                )
                resp.raise_for_status()
                ct = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
                content_type = ct

                if "pdf" in ct or is_pdf:
                    data = resp.content
                    kind = "pdf"
                elif "html" in ct or is_html or ct in ("", "text/plain"):
                    data = resp.text.encode(resp.encoding or "utf-8", errors="replace")
                    kind = "html"
                else:
                    return DownloadResult(
                        local_path="", url=url, source=source, content_type=ct or "other",
                        status="skipped", error=f"unsupported content type: {ct}",
                    )

                local_path.write_bytes(data)
                # Write sidecar metadata so provenance survives to extraction.
                meta = {
                    "id": new_id(),
                    "url": url,
                    "source": source,
                    "fetched_at": now_iso(),
                    "content_type": kind,
                    "status": "ok",
                }
                local_path.with_suffix(local_path.suffix + ".meta.json").write_text(
                    json.dumps(meta, ensure_ascii=False), encoding="utf-8"
                )
                return DownloadResult(
                    local_path=str(local_path), url=url, source=source,
                    content_type=kind, status="ok",
                )
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                if attempt < self.max_retries:
                    backoff = 1.5 * attempt
                    time.sleep(backoff)
                else:
                    return DownloadResult(
                        local_path="", url=url, source=source,
                        content_type=content_type, status="error", error=last_error,
                    )
        return DownloadResult(
            local_path="", url=url, source=source, content_type=content_type,
            status="error", error="unknown download failure",
        )

    # ------------------------------------------------------------------ internals
    def _rate_limit(self) -> None:
        if self.request_delay_seconds <= 0:
            return
        elapsed = time.monotonic() - self._last_request_time
        wait = self.request_delay_seconds - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request_time = time.monotonic()

    def _cache_path(self, url: str, is_pdf: bool) -> Path:
        import hashlib

        digest = hashlib.md5(url.encode("utf-8")).hexdigest()[:16]
        ext = ".pdf" if is_pdf else ".html"
        # keep a readable prefix from the slug where possible
        slug = re.sub(r"[^a-z0-9]+", "-", urllib.parse.urlparse(url).netloc)[:40]
        return self.download_dir / f"{slug}-{digest}{ext}"

    def _allowed_by_robots(self, url: str) -> bool:
        parsed = urllib.parse.urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        rp = self._robots_cache.get(base)
        if rp is None:
            rp = urllib.robotparser.RobotFileParser()
            try:
                rp.set_url(base + "/robots.txt")
                rp.read()
            except Exception:  # noqa: BLE001 - robots unavailable => allow
                return True
            self._robots_cache[base] = rp
        return rp.can_fetch(self.user_agent, url)
