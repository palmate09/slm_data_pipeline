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

# Non-HTTP redirects render in the browser via JS/meta refresh; the raw HTML
# is a tiny stub page.  Docs sites (e.g. docs.pytorch.org, scikit-learn
# auto_examples) use these heavily, so they must be followed to the real page.
_SOFT_REDIRECT_RE = re.compile(
    "(?:" +
    r"""<meta[^>]+http-equiv=["']refresh["'][^>]+content=["'][^"']*url=([^"'>]+)""" +
    "|" +
    r"""location\.(?:replace|assign|href)\s*[=(]\s*["']([^"']+)["']""" +
    ")",
    re.IGNORECASE,
)
_FRAMESET_RE = re.compile(r"<frameset[\s>]", re.IGNORECASE)
_FRAME_SRC_RE = re.compile(r"""<frame[^>]+src=['"]([^'"]+)['"]""", re.IGNORECASE)
_FRAME_TARGET_RE = re.compile(
    r"""<a[^>]+(?:href=[^>]+target=['"]([^'"]+)['"]|target=['"]([^'"]+)['"][^>]+href=)""",
    re.IGNORECASE,
)


def soft_redirect_target(html: str, base_url: str) -> str | None:
    """Return the target of a meta-refresh/JS redirect stub, if any."""
    match = _SOFT_REDIRECT_RE.search(html)
    if not match:
        return None
    target = next((g for g in match.groups() if g), "").strip()
    if not target:
        return None
    resolved = urllib.parse.urljoin(base_url, target)
    if resolved.startswith(("http://", "https://")):
        return resolved
    return None


def is_soft_redirect_stub(head: str, is_pdf: bool) -> bool:
    """True if the bytes hold a redirect stub, a frameset shell, or nothing."""
    if is_pdf:
        return False
    if not head.strip():
        return True
    if _FRAMESET_RE.search(head):
        return True
    return bool(soft_redirect_target(head, "https://example.invalid/"))


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
        max_soft_redirects: int = 3,
    ) -> None:
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.request_delay_seconds = request_delay_seconds
        self.max_retries = max_retries
        self.request_timeout_seconds = request_timeout_seconds
        self.user_agent = user_agent
        self.respect_robots_txt = respect_robots_txt
        self.max_soft_redirects = max_soft_redirects
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
            # Invalidate a cache entry that actually holds a browser-level
            # redirect stub (or an empty/broken download): re-fetch instead.
            head = local_path.read_bytes()[:8192].decode("utf-8", errors="replace")
            if is_soft_redirect_stub(head, is_pdf):
                local_path.unlink()
                side = local_path.with_suffix(local_path.suffix + ".meta.json")
                if side.exists():
                    side.unlink()
            else:
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
                    data = self._decode_response(resp)
                    kind = "html"
                    # Follow browser-level (meta refresh / JS / frameset)
                    # redirects that HTTP redirect handling misses.
                    data, final_url = self._resolve_inline_redirects(
                        data.decode("utf-8", errors="replace"), url
                    )
                    data = data.encode("utf-8")
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
                    "final_url": final_url if kind == "html" else url,
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

    @staticmethod
    def _decode_response(resp) -> bytes:
        """Decode HTML bytes, preferring a server/meta charset over ISO-8859-1.

        ``requests`` claims ISO-8859-1 for any ``text/*`` response without an
        explicit charset, which silently mangles UTF-8 pages.
        """
        charset = resp.headers.get("Content-Type", "").lower()
        if "charset=" not in charset:
            meta_charset = re.search(
                rb'<meta[^>]+charset=["\']?([\w-]+)', resp.content[:8192], re.IGNORECASE
            )
            encoding = meta_charset.group(1).decode("ascii", "ignore") if meta_charset else None
            if not encoding:
                encoding = resp.apparent_encoding or "utf-8"
            return resp.content.decode(encoding, errors="replace").encode("utf-8", "replace")
        return resp.content

    def _resolve_inline_redirects(self, html: str, url: str, *, hops: int = 0) -> tuple[str, str]:
        """Follow meta-refresh / JS redirects and resolve framesets."""
        target = soft_redirect_target(html, url)
        if target is not None:
            if hops >= self.max_soft_redirects or self._same_page(target, url):
                return html, url
            self._rate_limit()
            resp = self._session.get(
                target, timeout=self.request_timeout_seconds, allow_redirects=True
            )
            resp.raise_for_status()
            page = self._decode_response(resp).decode("utf-8", errors="replace")
            return self._resolve_inline_redirects(page, target, hops=hops + 1)

        if _FRAMESET_RE.search(html):
            return self._resolve_frameset(html, url, hops=hops)
        return html, url

    def _resolve_frameset(self, html: str, url: str, *, hops: int = 0) -> tuple[str, str]:
        """Frameset pages: fetch the main content frame and follow its redirect.

        Frame targets are ranked by name/src ("main", "content", ...) and fall
        back to trying every frame in document order.  Menu/header frames are
        ignored when a main-looking frame resolves.
        """
        frames: list[tuple[str, str]] = []
        for match in re.finditer(r"<frame[^>]*>", html, re.IGNORECASE):
            tag = match.group(0)
            src_m = re.search(r'src=["\']([^"\']+)["\']', tag, re.IGNORECASE)
            if not src_m:
                continue
            name_m = re.search(r'name=["\']([^"\']+)["\']', tag, re.IGNORECASE)
            frames.append((src_m.group(1).strip(), (name_m.group(1) if name_m else "").lower()))
        if not frames:
            return html, url

        def rank(src: str, name: str) -> int:
            text = name + " " + src.lower()
            if re.search(r"(main|content|body|right)", text):
                return 0
            if re.search(r"(menu|nav|left|top|header)", text):
                return 2
            return 1

        ordered = sorted(
            frames,
            key=lambda item: (rank(item[1], item[0]), frames.index(item)),
        )
        for src, _name in ordered:
            frame_url = urllib.parse.urljoin(url, src)
            try:
                self._rate_limit()
                resp = self._session.get(
                    frame_url, timeout=self.request_timeout_seconds, allow_redirects=True
                )
                resp.raise_for_status()
                page = self._decode_response(resp).decode("utf-8", errors="replace")
                if soft_redirect_target(page, frame_url) is not None or _FRAMESET_RE.search(page):
                    page, resolved = self._resolve_inline_redirects(page, frame_url, hops=hops + 1)
                if len(page.strip()) > 500:
                    return page, frame_url
            except Exception:  # noqa: BLE001 - try the next frame
                continue
        return html, url

    @staticmethod
    def _same_page(a: str, b: str) -> bool:
        pa, pb = urllib.parse.urlparse(a), urllib.parse.urlparse(b)
        return (pa.netloc.lower(), pa.path.rstrip("/")) == (pb.netloc.lower(), pb.path.rstrip("/"))

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
