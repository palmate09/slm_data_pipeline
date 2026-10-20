"""Reference/section removal helpers.

Reference lists are the single most common source of noisy training text in
Wikipedia-like pages.  We remove whole sections when a heading identifies them,
then fall back to line-level filters for pages with no usable headings.
"""

from __future__ import annotations

import re

from ..models import Block
from .artifacts import is_reference_line as _is_reference_line

REFERENCE_HEADINGS = {
    "references",
    "see also",
    "external links",
    "further reading",
    "sources",
    "notes",
    "bibliography",
    "citations",
    "footnotes",
    "works cited",
    "literature",
    "external links and references",
}

_REFERENCE_TAG_RE = re.compile(
    r"\[\s*(?:"
    r"citation needed|needs update|clarification needed|non-primary source needed|"
    r"better source needed|failed verification|original research\?|"
    r"unreliable source\?|self-published source\?|verification needed|"
    r"according to whom\?|who\?|when\?|why\?|how\?|where\?|disputed[^]]*|"
    r"note \d+|nb \d+|edit|update|tone|npov|importance\?|AI-generated\?|"
    r"dead link|permanent dead link|bare url|unreliable source|"
    r"[a-zA-Z]"
    r")\s*\]",
    re.IGNORECASE,
)

_NUMERIC_CITATION_RE = re.compile(r"\[\s*\d{1,3}(?:\s*,\s*\d{1,3}){0,4}\s*\]")

def is_reference_heading(text: str) -> bool:
    value = re.sub(r"\s+", " ", (text or "").strip().lower()).rstrip(":")
    return value in REFERENCE_HEADINGS


def strip_reference_sections(blocks: list[Block]) -> list[Block]:
    """Remove reference-like sections from a block list.

    A section starts at a matching heading and ends at the next heading with a
    level less than or equal to the reference heading.  If the heading has no
    level (plain-text inputs), treat it as level 2.
    """
    out: list[Block] = []
    skip_level: int | None = None
    for block in blocks:
        if block.type == "heading":
            level = block.level or 2
            if skip_level is not None and level <= skip_level:
                skip_level = None
            if is_reference_heading(block.text):
                skip_level = level
                continue
        if skip_level is None:
            out.append(block)
    return out


def strip_inline_citations(text: str) -> str:
    if not text:
        return text
    text = _REFERENCE_TAG_RE.sub("", text)

    def citation(match: re.Match) -> str:
        # Do not mistake a numeric vector after an equals sign for a citation.
        prefix = text[max(0, match.start() - 3):match.start()]
        return match.group(0) if re.search(r"[=≈]\s*$", prefix) else ""

    text = _NUMERIC_CITATION_RE.sub(citation, text)
    # Tidy double spaces/punctuation left behind by removals.
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    text = re.sub(r"\s+([)\]])", r"\1", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def is_reference_line(text: str) -> bool:
    """True for a bibliographic/reference line, not a prose mention.

    Delegates to the shared scanner so the cleaner, quality gate, and audit
    commands all agree on what counts as a reference.
    """
    return _is_reference_line(text)
