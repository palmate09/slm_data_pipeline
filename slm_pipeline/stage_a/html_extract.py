"""Structure-aware HTML/PDF/text extraction.

The old extractor flattened pages to a string with jusText and kept only
paragraphs it labelled ``good``.  That silently dropped Wikipedia leads and
turned captions/references/math into training text.  This module produces typed
blocks instead, runs more than one extractor when needed, and scores the
candidates so a bad extraction is visible rather than silently emitted.
"""

from __future__ import annotations

import copy
import html as html_lib
import re
import urllib.parse
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from lxml import etree
from lxml import html as lxml_html

from ..models import Block

# --------------------------------------------------------------------------- data


@dataclass
class ExtractionCandidate:
    blocks: list[Block]
    extractor: str
    title: str = ""
    meta: dict = field(default_factory=dict)
    score: float = 0.0
    words: int = 0
    paragraphs: int = 0
    headings: int = 0
    lead_ok: bool = False
    boilerplate_ratio: float = 0.0

    def summary(self) -> dict:
        return {
            "extractor": self.extractor,
            "words": self.words,
            "paragraphs": self.paragraphs,
            "headings": self.headings,
            "lead_ok": self.lead_ok,
            "boilerplate_ratio": round(self.boilerplate_ratio, 4),
            "score": round(self.score, 4),
        }


@dataclass
class HtmlExtraction:
    blocks: list[Block]
    title: str = ""
    extractor: str = ""
    confidence: float = 0.0
    candidates: list[dict] = field(default_factory=list)
    error: str | None = None
    site_name: str | None = None
    canonical_url: str | None = None
    language: str | None = None
    license: str | None = None


# --------------------------------------------------------------------------- constants

REFERENCE_HEADINGS = {
    "references", "see also", "external links", "further reading", "sources",
    "notes", "bibliography", "citations", "footnotes", "works cited",
}

CAPTION_STARTS = re.compile(
    r"^(?:a|an|the)\s+(?:diagram|image|photo|photograph|screenshot|illustration|"
    r"chart|graph|figure|table|map|logo|picture|render|visuali[sz]ation)\b",
    re.IGNORECASE,
)
CAPTION_WORDS = re.compile(
    r"\b(showing|depicted|illustrated|screenshot|diagram|infographic|"
    r"here, each|from left to right|clockwise)\b",
    re.IGNORECASE,
)

SKIP_TAGS = {
    "script", "style", "noscript", "svg", "canvas", "iframe", "form", "input",
    "button", "select", "textarea", "nav", "header", "footer", "aside",
}

SKIP_CLASS_RE = re.compile(
    r"(?:^|\s)(?:navbox|vertical-navbox|infobox|sidebar|hatnote|metadata|ambox|"
    r"mbox|toc|reflist|references|mw-references-wrap|mw-editsection|"
    r"shortdescription|noprint|sistersitebox|side-box|gallery|navigation|"
    r"breadcrumb|pagination|share|social|comment|comments|advert|advertisement|"
    r"cookie|consent|newsletter|related|recommended|promo|subscribe)(?:\s|$)",
    re.IGNORECASE,
)

BOILERPLATE_LINE_RE = re.compile(
    r"(?i)^(?:cookie(?:s)?(?: policy)?|we use cookies|accept (?:all )?cookies|"
    r"manage (?:cookie )?preferences|subscribe(?: to)?(?: our)? newsletter|"
    r"sign in|log in|create (?:an )?account|share this(?: article| page)?|"
    r"related articles?|advertisement|privacy policy|terms of service|"
    r"skip to (?:main |content )?|jump to content|read more|learn more|"
    r"download(?: the)? (?:pdf|app)|print this page|email this page|"
    r"follow us|connect with us|all rights reserved|copyright ©)"
)

INLINE_CITATION_RE = re.compile(
    r"\[\s*(?:"
    r"(?:citation needed|needs update|clarification needed|non-primary source needed|"
    r"better source needed|failed verification|original research\?|"
    r"unreliable source\?|self-published source\?|verification needed|"
    r"according to whom\?|who\?|when\?|why\?|how\?|where\?|disputed[^]]*|"
    r"note \d+|nb \d+|a|b|c|d|e|f|g|h|i|j|k|l|m|n|o|p|q|r|s|t|u|v|w|x|y|z|"
    r"edit|update|tone|npov|neutrality[^]]*|importance\?|AI-generated\?|"
    r"UNK|unknown)"
    r"|(?:\d{1,3})(?:\s*,\s*\d{1,3})*"
    r")\s*\]",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------- helpers


def normalize_inline(text: str) -> str:
    """Collapse whitespace in prose; keep the result on one line."""
    if not text:
        return ""
    text = text.replace("\xa0", " ").replace("\u200b", "").replace("\ufeff", "")
    text = re.sub(r"[\t\r\f\v]+", " ", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def clean_inline(text: str) -> str:
    text = normalize_inline(html_lib.unescape(text))
    text = INLINE_CITATION_RE.sub("", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    text = re.sub(r"\s+([)\]])", r"\1", text)
    return text.strip()


def _class_string(el) -> str:
    try:
        return (el.get("class") or "").lower()
    except Exception:  # noqa: BLE001
        return ""


def _text_content(el) -> str:
    """``text_content`` for HtmlElement and plain etree elements."""
    try:
        return el.text_content()
    except AttributeError:
        return "".join(el.itertext())


def _should_skip(el) -> bool:
    tag = (el.tag or "").lower() if isinstance(el.tag, str) else ""
    if tag in SKIP_TAGS:
        return True
    cls = _class_string(el)
    if cls and SKIP_CLASS_RE.search(cls):
        return True
    # Wikipedia citation markers and edit links.
    if tag == "sup" and ("reference" in cls or "noprint" in cls):
        return True
    if tag == "span" and "mw-editsection" in cls:
        return True
    return tag == "img"


def _element_text(el) -> str:
    """Text of an element with citation/UI nodes removed and math kept readable."""
    clone = copy.deepcopy(el)
    for bad in clone.xpath(
        ".//sup[contains(@class,'reference')] | .//span[contains(@class,'mw-editsection')] | "
        ".//style | .//script | .//noscript"
    ):
        bad.drop_tree()
    return clean_inline(_text_content(clone))


def _is_caption_like(text: str) -> bool:
    if not text:
        return True
    if len(text.split()) < 4:
        return True
    if CAPTION_STARTS.search(text):
        return True
    return bool(CAPTION_WORDS.search(text) and len(text.split()) < 35)


def _math_text(el) -> str:
    """Extract a readable representation of a MathML/MediaWiki math node."""
    alt = el.get("alttext") or ""
    if not alt:
        math_el = el if (el.tag or "").lower() == "math" else el.find(".//math")
        if math_el is not None:
            alt = math_el.get("alttext") or ""
            if not alt:
                alt = math_el.findtext(".//annotation") or ""
    if not alt:
        alt = _text_content(el)
    return clean_inline(alt)


def _table_to_markdown(table) -> str:
    rows: list[list[str]] = []
    for tr in table.xpath(".//tr"):
        cells = []
        for cell in tr.xpath("./th|./td"):
            cells.append(normalize_inline(_text_content(cell)))
        if cells and any(cells):
            rows.append(cells)
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    header = "| " + " | ".join(rows[0]) + " |"
    sep = "| " + " | ".join(["---"] * width) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join([header, sep] + body)


def _list_text(el) -> str:
    items: list[str] = []
    for li in el.xpath("./li"):
        text = _element_text(li)
        if text:
            items.append(f"- {text}")
    return "\n".join(items)


def _walk_dom(el, blocks: list[Block], heading_level: int = 1) -> None:
    """Recursively convert a content container into typed blocks."""
    for child in list(el):
        if not isinstance(child.tag, str):
            continue
        if _should_skip(child):
            continue

        tag = child.tag.lower()
        cls = _class_string(child)

        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            text = _element_text(child)
            if text:
                blocks.append(Block(type="heading", text=text, level=int(tag[1])))
            continue

        if tag == "p":
            text = _element_text(child)
            if text and len(text.split()) >= 2:
                blocks.append(Block(type="paragraph", text=text))
            continue

        if tag in {"ul", "ol"}:
            text = _list_text(child)
            if text:
                blocks.append(Block(type="list", text=text))
            continue

        if tag == "pre":
            code = _text_content(child).replace("\r\n", "\n").strip("\n")
            if code.strip():
                blocks.append(Block(type="code", text=code))
            continue

        if tag == "table":
            # Skip navigation/infobox tables; keep real data tables for the
            # table policy stage.
            if any(k in cls for k in ("navbox", "infobox", "vertical-navbox", "sidebar")):
                continue
            md = _table_to_markdown(child)
            if md:
                blocks.append(Block(type="table", text=md))
            continue

        if tag == "figure":
            for cap in child.xpath(".//figcaption|.//div[contains(@class,'thumbcaption')]"):
                text = _element_text(cap)
                if text:
                    blocks.append(Block(type="caption", text=text))
            continue

        if tag == "blockquote":
            text = _element_text(child)
            if text:
                blocks.append(Block(type="quote", text=text))
            continue

        if tag in {"dl"}:
            text = _list_text(child)
            if text:
                blocks.append(Block(type="list", text=text))
            continue

        if tag == "math" or "mwe-math-element" in cls:
            text = _math_text(child)
            if text:
                blocks.append(Block(type="formula", text=text))
            continue

        if tag in {"div", "section", "article", "main", "span", "figure"}:
            # Recurse into layout containers, but treat captions as leaves.
            if "thumbcaption" in cls:
                text = _element_text(child)
                if text:
                    blocks.append(Block(type="caption", text=text))
                continue
            _walk_dom(child, blocks, heading_level)
            continue

        # Unknown inline/block tag: recurse if it has element children,
        # otherwise treat as prose if it has enough text.
        if len(child):
            _walk_dom(child, blocks, heading_level)
        else:
            text = _element_text(child)
            if text and len(text.split()) >= 4:
                blocks.append(Block(type="paragraph", text=text))


def _trim_reference_sections(blocks: list[Block]) -> list[Block]:
    """Drop 'References', 'See also', etc. sections when a heading is present."""
    out: list[Block] = []
    skip_level: int | None = None
    for block in blocks:
        if block.type == "heading":
            level = block.level or 1
            if skip_level is not None and level <= skip_level:
                skip_level = None
            if block.text.strip().lower().rstrip(":") in REFERENCE_HEADINGS:
                skip_level = level
                continue
        if skip_level is None:
            out.append(block)
    return out


def _dedupe_adjacent(blocks: list[Block]) -> list[Block]:
    out: list[Block] = []
    prev = ""
    for block in blocks:
        key = re.sub(r"\s+", " ", block.text).strip().casefold()
        if key and key == prev:
            continue
        prev = key or prev
        out.append(block)
    return out


def blocks_to_text(
    blocks: Iterable[Block],
    *,
    heading_markup: bool = True,
    math_policy: str = "text",
    table_policy: str = "drop",
    caption_policy: str = "drop",
    code_policy: str = "keep",
) -> str:
    """Render blocks into the training text representation."""
    parts: list[str] = []
    for block in blocks:
        btype = block.type
        text = block.text.strip()
        if not text:
            continue
        if btype == "heading":
            if heading_markup:
                level = min(max(block.level or 2, 1), 6)
                parts.append("#" * level + " " + text)
            else:
                parts.append(text)
        elif btype == "paragraph" or btype == "list":
            parts.append(text)
        elif btype == "code":
            if code_policy == "drop":
                continue
            parts.append("```\n" + text + "\n```")
        elif btype == "table":
            if table_policy == "drop":
                continue
            if table_policy == "csv":
                rows = []
                for line in text.splitlines():
                    cells = [c.strip() for c in line.strip("|").split("|")]
                    rows.append(",".join('"' + c.replace('"', '""') + '"' for c in cells))
                parts.append("\n".join(rows))
            else:
                parts.append(text)
        elif btype == "caption":
            if caption_policy == "drop":
                continue
            parts.append("Figure: " + text)
        elif btype == "formula":
            if math_policy == "drop":
                continue
            if math_policy == "latex":
                parts.append("$" + text.strip("$") + "$")
            else:
                parts.append(text)
        elif btype == "quote":
            parts.append("> " + text)
        elif btype == "footnote" or btype == "navigation":
            continue
        else:
            parts.append(text)
    return "\n\n".join(p for p in parts if p.strip())


# --------------------------------------------------------------------------- HTML metadata


def _meta(html: str, *, prop: str | None = None, name: str | None = None) -> str | None:
    if prop:
        m = re.search(
            rf'<meta[^>]+property=["\']{re.escape(prop)}["\'][^>]+content=["\']([^"\']*)',
            html, re.IGNORECASE,
        )
        if not m:
            m = re.search(
                rf'<meta[^>]+content=["\']([^"\']*)["\'][^>]+property=["\']{re.escape(prop)}["\']',
                html, re.IGNORECASE,
            )
        if m:
            return clean_inline(m.group(1))
    if name:
        m = re.search(
            rf'<meta[^>]+name=["\']{re.escape(name)}["\'][^>]+content=["\']([^"\']*)',
            html, re.IGNORECASE,
        )
        if m:
            return clean_inline(m.group(1))
    return None


def extract_title(html: str, site_name: str | None = None) -> str:
    m = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    if not m:
        return ""
    raw_title = m.group(1)
    title = clean_inline(_text_content(etree.HTML(raw_title)) if "<" in raw_title else raw_title)
    # Strip common site suffixes.  If a site name is available use it;
    # otherwise strip a short trailing site/brand segment after a separator.
    if site_name:
        for sep in (" - ", " | ", " — ", " – ", " :: "):
            if title.lower().endswith((sep + site_name).lower()):
                title = title[: -len(sep + site_name)].strip()
    parts = re.split(r"\s+[|]\s+|\s+[-–—]\s+", title)
    if len(parts) > 1:
        head, tail = parts[0].strip(), parts[-1].strip()
        if head and 1 <= len(tail.split()) <= 8:
            title = head
    return title.strip()


def infer_license(url: str, html: str) -> str | None:
    host = urllib.parse.urlparse(url).netloc.lower()
    if "wikipedia.org" in host or "wikimedia.org" in host:
        return "CC BY-SA 4.0"
    m = re.search(
        r'<a[^>]+rel=["\']license["\'][^>]*>(.*?)</a>|'
        r'<link[^>]+rel=["\']license["\'][^>]+href=["\']([^"\']+)',
        html, re.IGNORECASE | re.DOTALL,
    )
    if m:
        value = clean_inline(re.sub(r"<[^>]+>", "", m.group(1) or m.group(2) or ""))
        if value:
            return value
    if "creativecommons.org" in html.lower():
        return "Creative Commons (see page)"
    return None


# --------------------------------------------------------------------------- candidates


def _make_candidate(blocks: list[Block], extractor: str, title: str, **meta) -> ExtractionCandidate:
    blocks = _dedupe_adjacent(_trim_reference_sections(blocks))
    text = blocks_to_text(blocks)
    words = len(text.split())
    paragraphs = sum(1 for b in blocks if b.type in {"paragraph", "quote"})
    headings = sum(1 for b in blocks if b.type == "heading")
    first_prose = next((b for b in blocks if b.type in {"paragraph", "quote", "list"}), None)
    lead_ok = bool(first_prose and not _is_caption_like(first_prose.text)
                   and len(first_prose.text.split()) >= 8
                   and first_prose.text[:1].isupper())
    boilerplate = sum(1 for b in blocks if BOILERPLATE_LINE_RE.search(b.text)) / max(1, len(blocks))
    # A simple score that rewards content, structure and a real opening, and
    # penalizes boilerplate.  It is intentionally not a "quality score".
    length_score = min(1.0, words / 600.0)
    para_score = min(1.0, paragraphs / 8.0)
    heading_score = min(1.0, headings / 5.0)
    score = (0.45 * length_score + 0.20 * para_score + 0.15 * heading_score
             + 0.20 * (1.0 if lead_ok else 0.0) - 0.25 * boilerplate)
    score = max(0.0, min(1.0, score))
    if words < 50:
        score = min(score, 0.2)
    return ExtractionCandidate(
        blocks=blocks, extractor=extractor, title=title, meta=meta, score=score,
        words=words, paragraphs=paragraphs, headings=headings,
        lead_ok=lead_ok, boilerplate_ratio=boilerplate,
    )


def _wikipedia_candidate(html: str, url: str, title: str) -> ExtractionCandidate | None:
    try:
        doc = lxml_html.document_fromstring(html)
    except Exception:  # noqa: BLE001
        return None
    roots = doc.xpath(
        "//*[@id='mw-content-text']//div[contains(@class,'mw-parser-output')]"
    )
    if not roots:
        roots = doc.xpath("//*[@id='mw-content-text']")
    if not roots:
        return None
    blocks: list[Block] = []
    _walk_dom(roots[0], blocks)
    return _make_candidate(blocks, "wikipedia", title)


def _trafilatura_candidate(html: str, url: str, title: str, *, favor_recall: bool = False) -> ExtractionCandidate | None:
    try:
        import trafilatura
    except Exception:  # noqa: BLE001
        return None
    try:
        xml = trafilatura.extract(
            html,
            output_format="xml",
            include_comments=False,
            include_tables=True,
            include_links=False,
            include_formatting=True,
            favor_precision=not favor_recall,
            favor_recall=favor_recall,
            url=url,
        )
        if not xml:
            return None
        root = etree.fromstring(xml.encode("utf-8"))
    except Exception:  # noqa: BLE001
        return None
    blocks: list[Block] = []

    def walk(node):
        tag = node.tag.lower() if isinstance(node.tag, str) else ""
        if tag in {"head", "title"}:
            text = clean_inline(_text_content(node))
            if text:
                blocks.append(Block(type="heading", text=text, level=2))
        elif tag == "p":
            text = clean_inline(_text_content(node))
            if text and len(text.split()) >= 2:
                blocks.append(Block(type="paragraph", text=text))
        elif tag in {"list", "item"}:
            if tag == "list":
                items = []
                for item in node.xpath("./item"):
                    text = clean_inline(_text_content(item))
                    if text:
                        items.append(f"- {text}")
                if items:
                    blocks.append(Block(type="list", text="\n".join(items)))
            else:
                text = clean_inline(_text_content(node))
                if text:
                    blocks.append(Block(type="paragraph", text=text))
        elif tag == "table":
            rows = []
            for row in node.xpath("./row"):
                cells = [clean_inline(_text_content(c)) for c in row.xpath("./cell")]
                if any(cells):
                    rows.append(cells)
            if rows:
                width = max(len(r) for r in rows)
                rows = [r + [""] * (width - len(r)) for r in rows]
                md = "\n".join("| " + " | ".join(r) + " |" for r in rows)
                blocks.append(Block(type="table", text=md))
        elif tag == "code":
            text = _text_content(node).strip()
            if text:
                blocks.append(Block(type="code", text=text))
        elif tag == "quote":
            text = clean_inline(_text_content(node))
            if text:
                blocks.append(Block(type="quote", text=text))
        elif tag == "figure":
            for cap in node.xpath("./figcaption|./caption"):
                text = clean_inline(_text_content(cap))
                if text:
                    blocks.append(Block(type="caption", text=text))
        elif tag in {"main", "doc"}:
            for child in node:
                walk(child)
        elif tag in {"hi", "ref"}:
            # Inline formatting markers: do not create a block.
            return
        else:
            for child in node:
                walk(child)

    # Start at <main> if present, otherwise walk <doc>.
    main = root.find(".//main")
    if main is not None:
        for child in main:
            walk(child)
    else:
        for child in root:
            walk(child)
    return _make_candidate(blocks, "trafilatura" + ("-recall" if favor_recall else ""), title)


def _readability_candidate(html: str, url: str, title: str) -> ExtractionCandidate | None:
    try:
        from readability import Document as ReadabilityDocument
    except Exception:  # noqa: BLE001
        return None
    # Readability logs a full traceback for pages without any element, so
    # skip the obviously empty ones before handing them over.
    if not html or not re.search(r"[A-Za-z]{4}", html):
        return None
    try:
        rdoc = ReadabilityDocument(html)
        summary = rdoc.summary(html_partial=True)
        root = lxml_html.document_fromstring(summary)
        rtitle = clean_inline(rdoc.short_title() or "")
    except Exception:  # noqa: BLE001
        return None
    blocks: list[Block] = []
    _walk_dom(root, blocks)
    return _make_candidate(blocks, "readability", title or rtitle)


def _justext_candidate(html: str, url: str, title: str) -> ExtractionCandidate | None:
    try:
        import justext
    except Exception:  # noqa: BLE001
        return None
    try:
        paras = justext.justext(html, justext.get_stoplist("English"))
    except Exception:  # noqa: BLE001
        return None
    blocks: list[Block] = []
    # Keep good paragraphs, plus a leading long paragraph if jusText marked the
    # lead as boilerplate.  This is a fallback only.
    good = [p for p in paras if p.class_type == "good"]
    for p in good:
        text = clean_inline(p.text)
        if text and len(text.split()) >= 2:
            blocks.append(Block(type="paragraph", text=text))
    if not blocks:
        for p in paras:
            text = clean_inline(p.text)
            if text and len(text.split()) >= 20:
                blocks.append(Block(type="paragraph", text=text))
    return _make_candidate(blocks, "justext", title)


# --------------------------------------------------------------------------- public HTML


def extract_html(
    html: str,
    *,
    url: str = "",
    requested_extractor: str = "auto",
    min_confidence: float = 0.25,
) -> HtmlExtraction:
    """Extract a document from HTML using the best available candidate."""
    site_name = _meta(html, prop="og:site_name") or _meta(html, name="application-name")
    title = extract_title(html, site_name)
    canonical = None
    m = re.search(
        r'<link[^>]+rel=["\']canonical["\'][^>]+href=["\']([^"\']+)',
        html, re.IGNORECASE,
    )
    if m:
        canonical = m.group(1)
    lang = None
    m = re.search(r'<html[^>]+lang=["\']([^"\']+)', html, re.IGNORECASE)
    if m:
        lang = m.group(1).split("-")[0].lower()
        # zxx ("no linguistic content") / und (undetermined) / mul (mixed)
        # mean "unknown", not "a foreign language"; a later detection should
        # win over them instead of producing a language-mismatch drop.
        if lang in {"zxx", "und", "mul"}:
            lang = None
    license_value = infer_license(url, html)

    candidates: list[ExtractionCandidate] = []
    host = urllib.parse.urlparse(url).netloc.lower()
    requested = (requested_extractor or "auto").lower()

    def add(candidate: ExtractionCandidate | None) -> None:
        if candidate and candidate.blocks:
            candidates.append(candidate)

    if requested in {"auto", "wikipedia"} and ("wikipedia.org" in host or "wikimedia.org" in host):
        add(_wikipedia_candidate(html, url, title))
    if requested in {"auto", "trafilatura"}:
        add(_trafilatura_candidate(html, url, title))
    if requested in {"auto", "readability"}:
        current_best = max((c.score for c in candidates), default=0.0)
        if requested == "readability" or not candidates or current_best < 0.55:
            add(_readability_candidate(html, url, title))
    if requested in {"auto", "justext"}:
        current_best = max((c.score for c in candidates), default=0.0)
        if requested == "justext" or not candidates or current_best < 0.35:
            add(_justext_candidate(html, url, title))
    if requested == "auto":
        best_words = max((c.words for c in candidates), default=0)
        # If precision extraction found very little (or nothing), try a
        # recall-oriented pass.  trafilatura's favor_precision mode returns
        # no text at all on some outline/TOC-heavy pages (e.g. the
        # scikit-learn user guide), so this pass must also run when the
        # candidate list is empty.
        if not candidates or best_words < 150:
            add(_trafilatura_candidate(html, url, title, favor_recall=True))

    if not candidates:
        return HtmlExtraction(
            blocks=[], title=title, extractor="none", confidence=0.0,
            candidates=[], error="no extractable text", site_name=site_name,
            canonical_url=canonical, language=lang, license=license_value,
        )

    candidates.sort(key=lambda c: (c.score, c.words), reverse=True)
    best = candidates[0]
    confidence = best.score
    if confidence < min_confidence and best.words < 80:
        confidence = min(confidence, 0.2)
    return HtmlExtraction(
        blocks=best.blocks,
        title=best.title or title,
        extractor=best.extractor,
        confidence=round(confidence, 4),
        candidates=[c.summary() for c in candidates],
        error=None if best.words >= 20 else "too little text",
        site_name=site_name,
        canonical_url=canonical,
        language=lang,
        license=license_value,
    )


# --------------------------------------------------------------------------- Markdown / text


def markdown_to_blocks(text: str) -> list[Block]:
    """Parse the subset of Markdown produced by PDF/text extraction."""
    blocks: list[Block] = []
    lines = text.replace("\r\n", "\n").split("\n")
    i = 0
    para: list[str] = []

    def flush():
        nonlocal para
        if para:
            value = clean_inline(" ".join(para))
            if value:
                blocks.append(Block(type="paragraph", text=value))
            para = []

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            flush()
            i += 1
            continue
        if stripped.startswith("```"):
            flush()
            fence = stripped[:3]
            i += 1
            code: list[str] = []
            while i < len(lines) and not lines[i].strip().startswith(fence):
                code.append(lines[i])
                i += 1
            i += 1
            value = "\n".join(code).strip("\n")
            if value:
                blocks.append(Block(type="code", text=value))
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            flush()
            blocks.append(Block(type="heading", text=clean_inline(m.group(2)), level=len(m.group(1))))
            i += 1
            continue
        if stripped.startswith("|") and stripped.endswith("|"):
            flush()
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            blocks.append(Block(type="table", text="\n".join(table_lines)))
            continue
        if re.match(r"^(?:[-*+]|\d+\.)\s+", stripped):
            flush()
            items = []
            while i < len(lines):
                m2 = re.match(r"^(?:[-*+]|\d+\.)\s+(.*)$", lines[i].strip())
                if not m2:
                    break
                items.append("- " + clean_inline(m2.group(1)))
                i += 1
            if items:
                blocks.append(Block(type="list", text="\n".join(items)))
            continue
        if stripped.startswith(">"):
            flush()
            blocks.append(Block(type="quote", text=clean_inline(stripped[1:])))
            i += 1
            continue
        para.append(stripped)
        i += 1
    flush()
    return _dedupe_adjacent(_trim_reference_sections(blocks))


def text_to_blocks(text: str) -> list[Block]:
    """Convert plain text into paragraph/heading/code blocks."""
    if not text or not text.strip():
        return []
    # Markdown-ish text is common for PDFs and local inputs.
    if re.search(r"(?m)^(?:#{1,6}\s|```|\|.*\|$)", text):
        return markdown_to_blocks(text)
    blocks: list[Block] = []
    for chunk in re.split(r"\n\s*\n", text):
        value = chunk.strip()
        if not value:
            continue
        lines = [ln.strip() for ln in value.splitlines() if ln.strip()]
        if len(lines) == 1 and len(lines[0].split()) <= 12 and not lines[0].endswith((".", "!", "?", ":", ";")):
            blocks.append(Block(type="heading", text=clean_inline(lines[0]), level=2))
        else:
            blocks.append(Block(type="paragraph", text=clean_inline(" ".join(lines))))
    return _dedupe_adjacent(_trim_reference_sections(blocks))


def extract_pdf(path: str | Path, *, url: str = "", title: str = "") -> HtmlExtraction:
    """Extract a PDF into blocks, preferring PyMuPDF4LLM markdown."""
    path = Path(path)
    blocks: list[Block] = []
    pdf_title = title
    try:
        import pymupdf4llm  # type: ignore

        md = pymupdf4llm.to_markdown(str(path))
        blocks = markdown_to_blocks(md)
    except Exception:  # noqa: BLE001
        try:
            import pymupdf  # type: ignore

            doc = pymupdf.open(str(path))
            pdf_title = pdf_title or (doc.metadata or {}).get("title", "") or ""
            parts = []
            for page in doc:
                parts.append(page.get_text("text"))
            blocks = text_to_blocks("\n\n".join(parts))
            doc.close()
        except Exception:  # noqa: BLE001
            try:
                from pypdf import PdfReader

                reader = PdfReader(str(path))
                pdf_title = pdf_title or (reader.metadata or {}).get("/Title", "") or ""
                parts = [(page.extract_text() or "") for page in reader.pages]
                blocks = text_to_blocks("\n\n".join(parts))
            except Exception as exc:  # noqa: BLE001
                return HtmlExtraction(
                    blocks=[], title=pdf_title, extractor="pdf", confidence=0.0,
                    error=f"pdf extraction failed: {exc}",
                )
    candidate = _make_candidate(blocks, "pdf", pdf_title)
    if not candidate.blocks:
        return HtmlExtraction(
            blocks=[], title=pdf_title, extractor="pdf", confidence=0.0,
            error="no extractable text",
        )
    return HtmlExtraction(
        blocks=candidate.blocks, title=pdf_title or "",
        extractor="pdf", confidence=round(candidate.score, 4),
        candidates=[candidate.summary()], error=None,
    )


def extract_plain_text(text: str, *, url: str = "", title: str = "") -> HtmlExtraction:
    blocks = text_to_blocks(text)
    candidate = _make_candidate(blocks, "text", title)
    return HtmlExtraction(
        blocks=candidate.blocks, title=title, extractor="text",
        confidence=round(candidate.score, 4), candidates=[candidate.summary()],
        error=None if candidate.words >= 20 else "too little text",
    )
