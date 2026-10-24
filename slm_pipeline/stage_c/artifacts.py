"""Shared artifact patterns and scanners.

These patterns are used by the cleaner and by the audit/validation commands so
the same definition of "dirty text" is applied everywhere.
"""

from __future__ import annotations

import re

# Real HTML element names.  A generic ``<something>`` pattern also matches
# placeholders (``<your-username>``) and Python reprs (``<module 'torch'>``)
# that show up in API documentation prose, so the tag list is explicit.
_HTML_TAGS = (
    "a|abbr|address|area|article|aside|audio|b|base|bdi|bdo|big|blockquote|"
    "body|br|button|canvas|caption|center|cite|code|col|colgroup|data|datalist|"
    "dd|del|details|dfn|dialog|div|dl|dt|em|embed|fieldset|figcaption|figure|"
    "font|footer|form|h1|h2|h3|h4|h5|h6|head|header|hgroup|hr|html|i|iframe|"
    "img|input|ins|kbd|label|legend|li|link|main|map|mark|menu|meta|meter|"
    "nav|noscript|object|ol|optgroup|option|output|p|param|picture|pre|progress|"
    "q|rp|rt|ruby|s|samp|script|section|select|slot|small|source|span|strong|"
    "style|sub|summary|sup|table|tbody|td|template|textarea|tfoot|th|thead|"
    "time|title|tr|track|u|ul|var|video|wbr"
)

ARTIFACT_PATTERNS = {
    "displaystyle": re.compile(r"\{\\displaystyle|\{\\textstyle"),
    "latex_command": re.compile(r"\\[a-zA-Z]+"),
    "uparrow_reference": re.compile(r"(?m)^\s*↑"),
    "citation_tag": re.compile(
        r"\[(?:citation needed|needs update|clarification needed|"
        r"non-primary source needed|better source needed|failed verification|"
        r"original research\?|unreliable source\?|self-published source\?|"
        r"verification needed|according to whom\?|who\?|when\?|why\?|how\?|"
        r"where\?|note \d+|edit|update|tone|npov|importance\?|AI-generated\?)\]",
        re.IGNORECASE,
    ),
    "redacted": re.compile(r"\[REDACTED\]"),
    "html_entity": re.compile(r"&[a-zA-Z]+;|&#\d+;"),
    "html_tag": re.compile(r"</?(?:" + _HTML_TAGS + r")\b[^>]*>", re.IGNORECASE),
    "wiki_template": re.compile(r"\{\{[^{}\n]{1,300}\}"),
    # A MediaWiki link is ``[[Target]]`` or ``[[Target|Label]]``.  Commas are
    # excluded from the target so Python type hints such as
    # ``Callable[[PreTrainedModel, torch.Tensor], torch.Tensor]`` (common in
    # API documentation prose) are not reported as wiki links.
    "wiki_link": re.compile(r"\[\[[^\[\]\n,]{1,120}(?:\|[^\[\]\n]{0,120})?\]\]"),
    "markdown_image": re.compile(r"!\[[^\]]*\]\([^)]*\)"),
    "url": re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE),
    "isbn": re.compile(r"\bISBN[\s:]*[\dXx-]{10,17}\b"),
    "doi": re.compile(r"\bdoi:\s*10\.\d{4,9}/\S+", re.IGNORECASE),
    "arxiv": re.compile(r"\barXiv:\s*\d{4}\.\d{4,5}\b", re.IGNORECASE),
    "reference_entry": re.compile(
        r"(?m)^(?:↑|\^|\s*\d+\.\s+.*(?:doi:|ISBN|Retrieved|pp\.))"
    ),
    "control_chars": re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"),
    # Real mojibake is a lead byte followed by a continuation byte ("Ã©") or
    # the "â€" family ("â€™").  A bare "â" is legitimate text such as "Română".
    "mojibake": re.compile(
        r"[\xc2-\xc3][\x80-\xbf]|\u00e2\u20ac[\u20ac\u2122\u2018\u2019\u201c\u201d\u0153\u017e\u2026]"
    ),
}

_CODE_FENCE_RE = re.compile(r"^\s*(?:```|~~~)")


def strip_code_blocks(text: str) -> str:
    """Return the prose part of a rendered document.

    Fenced code is preserved on purpose, and code legitimately contains
    URLs, HTML-like tags (``<eod>``), template braces and ``[[...]]``
    syntax.  Those must not be reported as document artifacts.
    """
    kept: list[str] = []
    in_code = False
    for line in (text or "").splitlines():
        if _CODE_FENCE_RE.match(line):
            in_code = not in_code
            continue
        if not in_code:
            kept.append(line)
    return "\n".join(kept)

BOILERPLATE_LINE_RE = re.compile(
    r"(?i)^(?:cookie(?:s)?(?: policy)?|we use cookies|accept (?:all )?cookies|"
    r"manage (?:cookie )?preferences|subscribe(?: to)?(?: our)? newsletter|"
    r"sign in|log in|create (?:an )?account|share this(?: article| page)?|"
    r"related articles?|advertisement|privacy policy|terms of service|"
    r"skip to (?:main |content )?|jump to content|read more|learn more|"
    r"download(?: the)? (?:pdf|app)|print this page|email this page|"
    r"follow us|connect with us|all rights reserved|copyright ©)"
)

NAV_LINE_RE = re.compile(
    r"(?i)^(?:home|about(?: us)?|contact(?: us)?|blog|docs?|documentation|"
    r"pricing|careers|community|support|resources|events|partners|"
    r"terms|privacy|cookies?|sitemap|search|menu|navigation|"
    r"previous|next|back to top|table of contents|contents)$"
)

REFERENCE_LINE_RE = re.compile(
    r"(?i)(?:"
    r"^\s*↑|^\s*\^|"
    r"\b(?:ISBN|ISSN)\s*[\dXx-]{8,}|"
    r"\bdoi:\s*10\.|"
    r"\barXiv:\s*\d{4}\.|"
    r"\b(?:Retrieved|Archived)\s+(?:on\s+)?\d|"
    r"\b(?:S2CID|OCLC|PMID|JSTOR|PMC)\s*:?\s*\d+|"
    r"\bpp?\.\s*\d+\s*[-–]\s*\d+"
    r")"
)

# A numbered bibliography often arrives as one long paragraph from PDF/HTML
# extractors.  Require both a numbered marker and bibliographic vocabulary so
# ordinary numbered tutorials are not removed.
NUMBERED_REFERENCE_RE = re.compile(
    r"(?i)^\s*(?:\d{1,3}[.)]\s*|\[\d{1,3}\]\s*)"
    r"[\"“']?.{0,800}?"
    r"(?:Journal|Proceedings|Conference|Vol\.|accessed|University|Press|"
    r"Springer|IEEE|ACM|arXiv|doi|ISBN)"
)

FRAGMENT_RE = re.compile(
    r"^(?:"
    r"(?:figure|fig\.|table|source|image|photo|credit|illustration)\b|"
    r"see also|read more|related|share|advertisement|"
    r"[A-Z][A-Za-z0-9 ,'()/-]{0,60}$"
    r")",
    re.IGNORECASE,
)


def scan_artifacts(text: str) -> dict[str, int]:
    """Count known artifact patterns in the prose part of a string.

    Fenced code blocks are excluded on purpose: sample code may contain
    URLs, HTML-like tags, LaTeX commands or template braces, and those are
    legitimate content rather than extraction debris.
    """
    counts: dict[str, int] = {}
    prose = strip_code_blocks(text)
    for name, pattern in ARTIFACT_PATTERNS.items():
        n = len(pattern.findall(prose))
        if n:
            counts[name] = n
    return counts


def is_boilerplate_line(text: str) -> bool:
    value = (text or "").strip()
    if not value:
        return False
    return bool(BOILERPLATE_LINE_RE.search(value) or NAV_LINE_RE.fullmatch(value))


def is_reference_line(text: str) -> bool:
    value = (text or "").strip()
    if not value:
        return False
    if REFERENCE_LINE_RE.search(value):
        return True
    # Numbered bibliographies can be long and arrive as one merged paragraph;
    # check this before the ordinary line-length guard.
    if NUMBERED_REFERENCE_RE.match(value):
        return True
    if len(value.split()) > 70:
        return False
    return False


def is_fragment_line(text: str) -> bool:
    value = (text or "").strip()
    if not value:
        return True
    if len(value.split()) > 18:
        return False
    if value.endswith((".", "!", "?", ":", ";", '"', ")")):
        return False
    return bool(FRAGMENT_RE.search(value))
