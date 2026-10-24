"""Recover useful high-value URLs using direct/API-friendly extraction.

This is deliberately separate from the normal crawl: it writes a JSONL input
that can be reviewed and fed through the normal cleaning/deduplication stages.
It prefers direct PDF and GitHub raw/README content over webpage shells.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

import requests
from lxml import html as lxml_html

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/126 Safari/537.36"
TIMEOUT = 30


def github_candidates(url: str) -> list[str]:
    p = urlsplit(url)
    bits = [x for x in p.path.split("/") if x]
    if len(bits) < 2:
        return []
    owner, repo = bits[:2]
    branch_path = "/".join(bits[2:])
    out = [f"https://raw.githubusercontent.com/{owner}/{repo}/HEAD/README.md"]
    if branch_path:
        out.insert(0, f"https://raw.githubusercontent.com/{owner}/{repo}/HEAD/{branch_path}")
    return out


def extract_html(data: bytes) -> str:
    root = lxml_html.fromstring(data)
    for node in root.xpath("//script|//style|//nav|//header|//footer|//aside|//form"):
        node.drop_tree()
    candidates = root.xpath("//article|//main|//body")
    root = candidates[0] if candidates else root
    return "\n\n".join(x.strip() for x in root.itertext() if x.strip() and len(x.strip()) > 1)


def extract_pdf(data: bytes) -> str:
    try:
        from pypdf import PdfReader
        import io
        return "\n\n".join((p.extract_text() or "").strip() for p in PdfReader(io.BytesIO(data)).pages)
    except Exception:
        return ""


def fetch(url: str) -> tuple[str, str]:
    urls = github_candidates(url) if "github.com" in urlsplit(url).netloc else [url]
    for candidate in urls:
        try:
            r = requests.get(candidate, headers={"User-Agent": UA, "Accept": "text/html,application/pdf,text/plain"}, timeout=TIMEOUT)
            if r.status_code >= 400:
                continue
            ct = r.headers.get("content-type", "").lower()
            if "pdf" in ct or candidate.lower().endswith(".pdf"):
                text = extract_pdf(r.content)
            elif "html" in ct:
                text = extract_html(r.content)
            else:
                text = r.content.decode("utf-8", "replace")
            if len(text.split()) >= 50:
                return text, candidate
        except requests.RequestException:
            continue
    return "", url


def main() -> int:
    source = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode/high_value_retry.txt")
    output = Path(sys.argv[2] if len(sys.argv) > 2 else ".pipeline_staging/machine_learning/recovered_high_value.jsonl")
    urls = list(dict.fromkeys(x.strip() for x in source.read_text().splitlines() if x.strip() and not x.startswith("#")))
    output.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if output.exists():
        for line in output.read_text(errors="replace").splitlines():
            try:
                rec = json.loads(line); done[rec["url"]] = rec
            except Exception:
                pass
    with output.open("a", encoding="utf-8") as f:
        for i, url in enumerate(urls, 1):
            if url in done:
                continue
            text, fetched = fetch(url)
            if text:
                rec = {
                    "id": hashlib.sha1(url.encode()).hexdigest(), "url": url,
                    "title": urlsplit(fetched).path.rsplit("/", 1)[-1] or urlsplit(url).netloc,
                    "text": text, "source": "machine_learning_recovery",
                    "fetched_url": fetched,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            if i % 25 == 0:
                print(f"{i}/{len(urls)} processed", flush=True)
            time.sleep(0.4)
    print("recovered records:", sum(1 for _ in output.open(encoding="utf-8")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
