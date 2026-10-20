"""Block-aware cleaning and normalization.

The old cleaner operated on a flat string and had a broad ``cookie``/``menu``
line filter, which removed legitimate prose and could not tell a caption from a
lead.  This chain cleans typed blocks first, then renders the final training
text.  It also keeps the legacy modifier classes so existing callers (and the
public Stage C API) keep working.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterable
from dataclasses import dataclass

from ..models import Block, Document
from ..stage_a.html_extract import blocks_to_text
from .artifacts import (
    BOILERPLATE_LINE_RE,
    NAV_LINE_RE,
    is_boilerplate_line,
    is_reference_line,
    scan_artifacts,
)
from .language import detect_language
from .normalize import math_to_text, normalize_heading, normalize_prose
from .pii import PiiModifier, PiiRedactor
from .references import (
    is_reference_heading,
    strip_inline_citations,
    strip_reference_sections,
)

__all__ = [
    "BoilerplateStripper",
    "CleaningChain",
    "Modifier",
    "PiiModifier",
    "QuotationUnifier",
    "Sequential",
    "TechnicalArtifactCleaner",
    "UnicodeReformatter",
    "default_cleaning_chain",
]

REFERENCE_HEADINGS = {
    "references", "see also", "external links", "further reading", "sources",
    "notes", "bibliography", "citations", "footnotes", "works cited",
}

_MAINTENANCE_RE = re.compile(
    r"(?i)^(?:this article|this section|this page)\s+"
    r"(?:needs|may need|is|has been|contains|requires|does not|relies)\b.*$"
    r"|^please help (?:update|improve|expand) this (?:article|section|page)\b.*$"
    r"|^learn how and when to remove this (?:template|message|notice)\b.*$"
    r"|^the examples and perspective in this (?:article|section)\b.*$"
    r"|^(?:wikitext|short description)\b.*$"
)

# Product CTAs are common at the end of vendor articles.  Keep this narrow:
# a marketing verb plus a brand/product cue in a short sentence.
_MARKETING_RE = re.compile(
    r"(?i)^(?:easily|move|reinvent|reimagine|unlock|transform|accelerate|"
    r"streamline|simplify|automate|discover|build|start|scale)\b"
    r".{0,240}?"
    r"(?:IBM|watsonx|Microsoft|Azure|Google Cloud|AWS|NVIDIA|Salesforce|"
    r"Oracle|SAP|our AI|our (?:solutions?|platform|products?|services?)|"
    r"with the help of|business value)\b"
    r"|^(?:IBM|Microsoft|Google|Amazon|Meta|NVIDIA|OpenAI|Apple)\s+itself\b"
    r".{0,180}\b(?:maintains?|offers?|provides?|updates?)\b"
    r"|^stay up to date\b.{0,240}\b(?:newsletter|subscribe|delivered)\b"
)

# A related-link sentence at the end of a tutorial index.
_RELATED_LINK_RE = re.compile(
    r"(?i)^after\b.{0,240}\bfrom here:\s*.{0,140}"
    r"(?:tutorial|guide|course)\b"
)

# Sponsorship and credit CTAs that appear in READMEs and vendor pages.
_SPONSOR_RE = re.compile(
    r"(?i)^thank you\b.{0,160}\bfor sponsoring\b"
    r"|^all\b.{0,160}\bexperiments are powered by\b"
    r"|^get started with \$.{0,160}\b(?:free )?credits\b"
    r"|^for more questions\b.{0,160}\b(?:discord|slack|forum|mailing list)\b"
)

# The same CTAs, but hidden inside a longer paragraph, are removed per
# sentence instead of per line.
_SPONSOR_SENTENCE_RE = re.compile(
    r"(?i)\b(?:sign up (?:with this link )?to get \$\d+|get \$\d+ credits?\b|"
    r"free credits\b|we get a credit\b|"
    r"sponsor(?:ing|ship)?\b[^.]{0,40}\b(?:labs|cloud|provider)\b)"
)

# Markup that leaks in from HTML, wiki and markdown sources.  Fenced code is
# kept verbatim, so these rules only run on prose and list items.
_MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_WIKI_LINK_RE = re.compile(r"\[\[([^\]|]*)(?:\|([^\]]*))?\]\]")
_WIKI_TEMPLATE_RE = re.compile(r"\{\{[^{}]*\}\}")
_HTML_TAG_RE = re.compile(
    r"</?[A-Za-z][A-Za-z0-9-]*"
    r"(?:\s+[A-Za-z-]+(?:=[^<>]*)?|\s+[\"'][^<>]*[\"'])*"
    r"\s*/?>"
)
_HTML_ENTITY_RE = re.compile(r"&(?:[a-zA-Z]+|#\d+);")
_URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>\"']+")
_REDACTION_MARKER_RE = re.compile(r"\[(?:REDACTED|EMAIL|URL|PHONE|SSN)\]")

# A language switcher or a nav row: many short fragments separated by "|".
_PIPE_NAV_RE = re.compile(r"^[^.|]{0,200}(?:\|[^.|]{1,40}){3,}\.?$")

# A sentence may end with a closing quote or parenthesis ("... too.)"), so the
# splitter also looks one character past the terminator.
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=["\'(\[]?[A-Z0-9])')
_SENTENCE_SPLIT_CLOSED_RE = re.compile(r'(?<=[.!?][\)\]"\'”])\s+(?=["\'(\[]?[A-Z0-9])')


def _split_sentences(text: str) -> list[str]:
    """Split prose into sentences, honouring a closing quote/parenthesis."""
    parts: list[str] = []
    for chunk in _SENTENCE_SPLIT_CLOSED_RE.split(text or ""):
        parts.extend(part for part in _SENTENCE_SPLIT_RE.split(chunk) if part.strip())
    return [part for part in parts if part.strip()]


def _drop_sentences_with(text: str, pattern: re.Pattern) -> str:
    """Remove whole sentences that contain ``pattern``.

    Used for redaction markers and bare URLs: dropping the sentence keeps
    the surrounding prose balanced instead of leaving a broken fragment.
    """
    if not text:
        return ""
    parts = _split_sentences(text)
    kept = [part for part in parts if not pattern.search(part)]
    return " ".join(kept).strip()


def _strip_markup(text: str) -> str:
    """Remove HTML, wiki and markdown residue from a prose line.

    Links keep their label, images and template braces are removed, and
    raw HTML tags are dropped.  Fenced code never reaches this function.
    """
    value = text
    value = _MARKDOWN_IMAGE_RE.sub(" ", value)
    value = _MARKDOWN_LINK_RE.sub(r"\1", value)
    value = _WIKI_LINK_RE.sub(lambda m: (m.group(2) or m.group(1) or ""), value)
    value = _WIKI_TEMPLATE_RE.sub(" ", value)
    value = _HTML_TAG_RE.sub(" ", value)
    for entity, char in (
        ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
        ("&quot;", '"'), ("&apos;", "'"), ("&nbsp;", " "),
    ):
        value = value.replace(entity, char)
    value = _HTML_ENTITY_RE.sub(" ", value)
    return value

# Sections that are often just directories of names/links.  Only a trailing
# section made entirely of headings/lists is removed.
TRIM_TAIL_HEADINGS = {
    "software", "journals", "conferences", "publications", "gallery",
    "awards", "external links", "further reading", "see also",
}


@dataclass
class Modifier:
    name: str = "modifier"
    enabled: bool = True

    def __call__(self, document):
        raise NotImplementedError


class Sequential:
    """Chain modifiers; disabled ones are skipped."""

    def __init__(self, modifiers: Iterable[Modifier]) -> None:
        self.modifiers = list(modifiers)

    def __call__(self, document):
        for mod in self.modifiers:
            if not mod.enabled:
                continue
            document = mod(document)
        return document


# --------------------------------------------------------------------------- text -> blocks


def _looks_like_heading(line: str) -> bool:
    value = line.strip()
    if not value or len(value) > 120:
        return False
    if value.startswith("#"):
        return True
    if value.endswith((".", "!", "?", ";", ":", ",")):
        return False
    if len(value.split()) > 12:
        return False
    if value.startswith(("-", "*", ">", "|", "•")):
        return False
    # A short title-like line with no sentence punctuation.
    return bool(re.match(r"^[A-Z0-9][A-Za-z0-9 &/'()\-–—,]+$", value))


def text_to_cleaning_blocks(text: str) -> list[Block]:
    """Parse a flat text field into heading/paragraph blocks for cleaning."""
    blocks: list[Block] = []
    paragraph: list[str] = []

    def flush() -> None:
        nonlocal paragraph
        if paragraph:
            value = " ".join(paragraph).strip()
            if value:
                blocks.append(Block(type="paragraph", text=value))
            paragraph = []

    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            flush()
            continue
        heading = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading:
            flush()
            blocks.append(Block(type="heading", text=heading.group(2), level=len(heading.group(1))))
            continue
        if _looks_like_heading(line):
            flush()
            blocks.append(Block(type="heading", text=line, level=2))
            continue
        paragraph.append(line)
    flush()
    return blocks


# --------------------------------------------------------------------------- corpus fingerprints


def build_repeated_line_fingerprints(
    documents: Iterable[Document],
    *,
    min_docs: int = 3,
    min_words: int = 8,
) -> set[str]:
    """Find paragraph/list lines repeated across documents.

    This catches site footers and syndicated blurbs that survived DOM
    extraction.  It deliberately ignores short lines and headings, where
    repetition is often legitimate.
    """
    from collections import defaultdict

    seen: dict[str, set[str]] = defaultdict(set)
    for doc in documents:
        if not getattr(doc, "kept", True):
            continue
        blocks = getattr(doc, "blocks", None) or []
        if not blocks:
            continue
        for block in blocks:
            if block.type not in {"paragraph", "list"}:
                continue
            for line in block.text.splitlines():
                value = re.sub(r"\s+", " ", line).strip().casefold()
                if len(value.split()) < min_words:
                    continue
                seen[value].add(doc.id)
    return {line for line, ids in seen.items() if len(ids) >= min_docs}


def remove_repeated_lines(document: Document, fingerprints: set[str]) -> int:
    if not fingerprints or not document.blocks:
        return 0
    removed = 0
    kept: list[Block] = []
    for block in document.blocks:
        if block.type not in {"paragraph", "list"}:
            kept.append(block)
            continue
        lines = []
        for line in block.text.splitlines():
            value = re.sub(r"\s+", " ", line).strip().casefold()
            if value in fingerprints:
                removed += 1
                continue
            lines.append(line)
        if lines:
            kept.append(Block(type=block.type, text="\n".join(lines), meta=dict(block.meta)))
        else:
            removed += 1
    document.blocks = kept
    if removed:
        document.metadata["boilerplate_lines_removed"] = removed
    return removed


# --------------------------------------------------------------------------- chain


class CleaningChain:
    """Clean typed blocks, then render a stable training text."""

    def __init__(
        self,
        *,
        fix_unicode: bool = True,
        unify_quotes: bool = True,
        redact_pii: bool = True,
        pii_entities: Iterable[str] | None = None,
        pii_person: bool = False,
        use_presidio: bool = False,
        language: str = "en",
        min_language_confidence: float = 0.9,
        remove_references: bool = True,
        remove_captions: bool = True,
        remove_math: bool = True,
        math_policy: str = "text",
        table_policy: str = "drop",
        code_policy: str = "keep",
        caption_policy: str = "drop",
        repair_spacing: bool = True,
        boilerplate_extra: Iterable[str] | None = None,
        heading_markup: bool = True,
        keep_language_mismatch: bool = True,
    ) -> None:
        self.fix_unicode = fix_unicode
        self.unify_quotes = unify_quotes
        self.redact_pii = redact_pii
        self.language = language
        self.min_language_confidence = min_language_confidence
        self.remove_references = remove_references
        self.remove_captions = remove_captions
        self.remove_math = remove_math
        self.math_policy = math_policy
        self.table_policy = table_policy
        self.code_policy = code_policy
        self.caption_policy = caption_policy
        self.repair_spacing = repair_spacing
        self.heading_markup = heading_markup
        self.keep_language_mismatch = keep_language_mismatch
        entities = list(pii_entities or [])
        if pii_person and "PERSON" not in entities:
            entities.append("PERSON")
        self.redactor = PiiRedactor(
            entities=entities or None,
            use_presidio=use_presidio or pii_person,
        )
        self.extra_boilerplate = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in (boilerplate_extra or [])
        ]

    # ------------------------------------------------------------------ public
    def __call__(self, document: Document) -> Document:
        t0 = time.perf_counter()
        if not document.text and not document.blocks:
            document.kept = False
            document.drop_reason = document.drop_reason or "empty document"
            return document

        from_blocks = bool(document.blocks)
        blocks = list(document.blocks) if from_blocks else text_to_cleaning_blocks(document.text)
        input_blocks = len(blocks)
        if self.remove_references and from_blocks:
            blocks = strip_reference_sections(blocks)

        cleaned: list[Block] = []
        pii_counts: dict[str, int] = {}
        dropped = 0
        seen: set[tuple[str, str]] = set()
        previous_key = ""

        for block in blocks:
            btype = block.type if block.type in {
                "heading", "paragraph", "list", "code", "table", "caption",
                "formula", "quote", "footnote", "navigation",
            } else "paragraph"

            if btype in {"footnote", "navigation"}:
                dropped += 1
                continue

            if btype == "code":
                if self.code_policy == "drop":
                    dropped += 1
                    continue
                text = block.text.strip("\n")
                if not text.strip():
                    dropped += 1
                    continue
                value = Block(type="code", text=text, meta=dict(block.meta))
            elif btype == "table":
                if self.table_policy == "drop":
                    dropped += 1
                    continue
                value = Block(type="table", text=block.text.strip(), meta=dict(block.meta))
            elif btype == "formula":
                if self.remove_math or self.math_policy == "drop":
                    dropped += 1
                    continue
                text = block.text.strip()
                if self.math_policy != "latex":
                    text = math_to_text(text)
                if not text:
                    dropped += 1
                    continue
                value = Block(type="formula", text=text, meta=dict(block.meta))
            elif btype == "caption":
                if self.remove_captions or self.caption_policy == "drop":
                    dropped += 1
                    continue
                text = self._clean_prose(block.text)
                if not text:
                    dropped += 1
                    continue
                value = Block(type="caption", text=text, meta=dict(block.meta))
            elif btype == "heading":
                text = normalize_heading(
                    block.text,
                    repair_spacing=self.repair_spacing,
                    fix_unicode_text=self.fix_unicode,
                )
                text = strip_inline_citations(text)
                if not text or is_reference_heading(text) or self._is_droppable_line(text):
                    dropped += 1
                    continue
                value = Block(type="heading", text=text, level=block.level, meta=dict(block.meta))
            elif btype == "list":
                text = self._clean_list(block.text, pii_counts)
                if not text:
                    dropped += 1
                    continue
                value = Block(type="list", text=text, meta=dict(block.meta))
            else:  # paragraph / quote
                text = self._clean_prose(block.text, pii_counts)
                if not text:
                    dropped += 1
                    continue
                value = Block(type=btype, text=text, meta=dict(block.meta))

            key = (value.type, re.sub(r"\s+", " ", value.text).strip().casefold())
            if not key[1]:
                dropped += 1
                continue
            if key == previous_key:
                dropped += 1
                continue
            if key in seen and value.type != "heading":
                dropped += 1
                continue
            seen.add(key)
            previous_key = key
            cleaned.append(value)

        cleaned, repaired = self._repair_dangling_prose(cleaned)
        cleaned, broken = self._drop_broken_sentences(cleaned)
        dropped += repaired + broken

        # Drop a trailing software/journals/conferences directory, then any
        # headings left behind after their captions/lists were removed (e.g. a
        # final "Gallery" heading with no gallery).
        cleaned, tail_removed = self._trim_low_value_tail(cleaned)
        dropped += tail_removed
        while cleaned and cleaned[-1].type == "heading":
            cleaned.pop()
            dropped += 1

        # A document made only of headings is not training text.
        if not any(b.type in {"paragraph", "list", "quote", "code", "table"} for b in cleaned):
            document.blocks = cleaned
            document.text = ""
            document.kept = False
            document.drop_reason = document.drop_reason or "no prose after cleaning"
            document.metadata["cleaning_stats"] = {
                "input_blocks": input_blocks,
                "output_blocks": len(cleaned),
                "dropped_blocks": dropped,
                "pii_counts": pii_counts,
            }
            return document

        document.blocks = cleaned
        document.text = blocks_to_text(
            cleaned,
            heading_markup=self.heading_markup,
            math_policy=self.math_policy,
            table_policy=self.table_policy,
            caption_policy=self.caption_policy,
            code_policy=self.code_policy,
        ).strip()

        prose = "\n".join(
            b.text for b in cleaned if b.type in {"paragraph", "list", "heading", "quote"}
        )
        detected, confidence = detect_language(prose)
        if detected:
            document.metadata["detected_language"] = detected
            document.metadata["language_confidence"] = confidence
            if document.language is None:
                document.language = detected
            elif document.language != detected and confidence >= self.min_language_confidence:
                document.metadata["language_mismatch"] = True
        document.metadata["cleaning"] = [
            "unicode_reformat" if self.fix_unicode else "unicode_passthrough",
            "quote_unify" if self.unify_quotes else "quote_passthrough",
            "block_clean",
            "reference_strip" if self.remove_references else "reference_keep",
            "caption_strip" if self.remove_captions else "caption_keep",
            "math_to_text" if (self.remove_math and self.math_policy == "text") else "math_policy",
            "pii_redact" if self.redact_pii else "pii_off",
        ]
        document.metadata.setdefault("cleaning_stats", {}).update({
            "input_blocks": input_blocks,
            "output_blocks": len(cleaned),
            "dropped_blocks": dropped,
            "pii_counts": pii_counts,
            "artifacts": scan_artifacts(document.text),
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
        })
        if not document.text:
            document.kept = False
            document.drop_reason = document.drop_reason or "empty after cleaning"
        return document

    # ------------------------------------------------------------------ helpers
    def _is_droppable_line(self, line: str) -> bool:
        value = (line or "").strip()
        if not value:
            return True
        if _MAINTENANCE_RE.search(value):
            return True
        if BOILERPLATE_LINE_RE.search(value) or NAV_LINE_RE.fullmatch(value):
            return True
        if _RELATED_LINK_RE.search(value):
            return True
        if _PIPE_NAV_RE.match(value):
            return True
        if len(value.split()) <= 45 and (_MARKETING_RE.search(value) or _SPONSOR_RE.search(value)):
            return True
        for pattern in self.extra_boilerplate:
            if pattern.search(value):
                return True
        return False

    _COMPLETE_SENTENCE_RE = re.compile(r'[.!?](?:["\')\]]+)?(?=\s|$)')
    _TERMINAL_END_RE = re.compile(r'[.!?](?:["\')\]]+)?$')
    _FOOTNOTE_TAIL_RE = re.compile(r'([.!?]["\']?)\s*\d{1,2}(?:\s*,\s*\d{1,2})*\s*$')
    _PURE_LEADIN_RE = re.compile(
        r"(?i)^(?:for example|for a general|in traditional|consider|considering|"
        r"let us|one might|we know|we have|now|which|where|assuming|if|when|then)\b"
    )
    _DANGLING_PHRASE_RE = re.compile(
        r"(?i),?\s*(?:in which case|so that|such that|as follows|given by|"
        r"defined as|the following|the sequence)\s*[,;:]?$"
    )
    _DANGLING_WORD_RE = re.compile(
        r"(?i)(?:\b(?:of|like|such|including|the|a|an|and|or|but|for|to|with|"
        r"by|as|that|which|where|when|in|on|at|from|is|are|was|were|be|been|"
        r"being|if|then|so|case|follows|sequence|using|into|over|under|"
        r"between|during|without|within|via|per|than|because|while|although|"
        r"though|since|until|unless|whether))\s*[,;:]?$"
    )
    _CODE_LIKE_RE = re.compile(
        r"^(?:if|elif)\b(?=.*(?:[<>=]|\bthen\b))"
        r"|^(?:IF|ELSE|ELIF|FOR|WHILE|THEN|RETURN)\b"
        r"|^for\s+[A-Za-z_]\w*\s+in\s+.+(?::| do\b)"
        r"|^while\s+.+(?::| do\b)"
        r"|^(?:def|class|function)\s+[A-Za-z_]\w*"
        r"|^print\s*\("
        r"|^return\s+[\[\]\w.]+"
    )

    @classmethod
    def _has_terminal(cls, text: str) -> bool:
        return bool(cls._TERMINAL_END_RE.search(text.rstrip()))

    @classmethod
    def _sentence_matches(cls, text: str) -> list[re.Match]:
        matches: list[re.Match] = []
        for match in cls._COMPLETE_SENTENCE_RE.finditer(text):
            start = match.start()
            # Do not treat the last dot of an ellipsis as a sentence end.
            if start >= 2 and text[start - 2:start + 1] == "...":
                continue
            matches.append(match)
        return matches

    @classmethod
    def _trim_to_last_sentence(cls, text: str) -> str:
        matches = cls._sentence_matches(text)
        if not matches:
            return ""
        prefix = text[: matches[-1].end()].strip()
        return prefix if len(prefix.split()) >= 6 else ""

    _BROKEN_SENTENCE_RE = re.compile(
        r"(?i)(?:"
        r"\b(?:of|the|a|an|and|or|but|as|that|which|where|is|are|was|were|"
        r"be|been|being|then|such|this|these|those|can|could|will|would|"
        r"should|may|might|must|from|for|in|on|at|if|while|than|because|"
        r"defined|given|prove|proved|follows?|yields?|becomes?|equals?)\s*\.\s*$"
        r"|\bas where\b|\bof is\b|:\s*(?:which|where)\b"
        r"|\b(?:the|a|an|such)\s+an?\s*\.\s*$"
        r'|""|^Compared with\b|^(?:We|One|It)\s+(?:calculate|compute|obtain)\.$'
        r"|[=+×÷<>]\s*\.\s*$"
        r"|^\([^)]*\b(?:e\.g|i\.e)\.\s*$"
        r"|\(e\.g\.\s*$|\(i\.e\.\s*$"
        r"|\b(?:we|one)\s+(?:obtain|get|find|have)\s*:\s*(?:if|when|for|where)\b"
        r"|\bHere,\s*,|\bHere,\s+and\s+are\b"
        r")"
    )

    @classmethod
    def _drop_broken_sentences(cls, blocks: list[Block]) -> tuple[list[Block], int]:
        """Drop sentences that lost a formula or inline expression.

        Examples are "the loss is.", "as follows.", "such an." or
        "defined as where ...".  Complete neighbouring sentences are kept.
        """
        dropped = 0
        out: list[Block] = []
        for block in blocks:
            if block.type not in {"paragraph", "quote"}:
                out.append(block)
                continue
            protected = block.text.replace("...", "\u2026")
            # Keep common abbreviations inside their sentence so a lead-in such
            # as "Substituting Eq. 2 ..." is inspected as a whole.
            protected = re.sub(
                r"\b(?:Eq|Fig|No|Dr|Mr|Mrs|Ms|vs|etc|al|cf|approx|resp)\.",
                lambda match: match.group(0).replace(".", "\u2024"),
                protected,
            )
            # Inline "e.g. CAPTCHAs" is part of the same sentence.  Keep it
            # intact; only a dangling "(e.g." at the end is a broken fragment.
            protected = re.sub(
                r"\b(?:e\.g|i\.e)\.(?=\s+\S)",
                lambda match: match.group(0).replace(".", "\u2024"),
                protected,
            )
            sentences = [
                part.replace("\u2026", "...").replace("\u2024", ".").strip()
                for part in re.split(r'(?<=[.!?])\s+(?=["\'(\[]?[A-Z0-9])', protected)
                if part.strip()
            ]
            kept = [sentence for sentence in sentences if not cls._BROKEN_SENTENCE_RE.search(sentence)]
            removed = len(sentences) - len(kept)
            if removed:
                dropped += removed
            text = " ".join(kept).strip()
            if not text:
                dropped += 1
                continue
            if text[:1].islower():
                dropped += 1
                continue
            out.append(Block(type=block.type, text=text, level=block.level, meta=dict(block.meta)))

        content_types = {"paragraph", "quote", "list", "code", "table"}
        kept_blocks: list[Block] = []
        for index, block in enumerate(out):
            if block.type == "heading":
                nxt = out[index + 1] if index + 1 < len(out) else None
                if nxt is None or nxt.type == "heading" or nxt.type not in content_types:
                    dropped += 1
                    continue
            kept_blocks.append(block)
        return kept_blocks, dropped

    @classmethod
    def _looks_like_code(cls, text: str) -> bool:
        value = text.strip()
        if not value or len(value.splitlines()) > 3:
            return False
        return bool(cls._CODE_LIKE_RE.search(value))

    @staticmethod
    def _structural_intro(text: str) -> str:
        value = re.sub(
            r"(?i)\b(?:of|like|such as|including|the following|as follows|the sequence)\s*$",
            "",
            text,
        ).rstrip(" ,;:")
        return value + ":"

    @staticmethod
    def _looks_like_formula(text: str) -> bool:
        compact = re.sub(r"\s+", "", text)
        if not compact:
            return False
        letters = sum(ch.isalpha() for ch in compact)
        symbol_math = bool(re.search(r"[=<>≤≥∑∇‖⊤∈⋮⋱]", text))
        words = re.findall(r"[A-Za-z]{3,}", text)
        if symbol_math and letters / max(1, len(compact)) < 0.5:
            return True
        return symbol_math and len(words) < 4

    def _repair_dangling_prose(self, blocks: list[Block]) -> tuple[list[Block], int]:
        """Remove prose fragments left behind when display math was dropped.

        Wikipedia-style pages often put a formula in its own block.  With
        ``remove_math`` enabled the formula disappears but the lead-in remains
        ("... as follows,").  Those fragments are not usable training text.
        We merge lowercase continuations, trim a paragraph back to its last
        complete sentence, and drop pure lead-ins.  Lists/code/tables keep their
        introducing paragraph, with a colon instead of a dangling comma.
        """
        dropped = 0
        merged: list[Block] = []
        for block in blocks:
            if block.type not in {"paragraph", "quote"}:
                merged.append(block)
                continue
            if (
                block.type == "paragraph"
                and self._looks_like_code(block.text)
                and not self._looks_like_formula(block.text)
            ):
                merged.append(
                    Block(type="code", text=block.text.strip(), level=block.level, meta=dict(block.meta))
                )
                continue
            text = self._FOOTNOTE_TAIL_RE.sub(r"\1", block.text).strip()
            if not text:
                dropped += 1
                continue
            if merged and merged[-1].type in {"paragraph", "quote"}:
                previous = merged[-1]
                previous_text = previous.text.rstrip()
                starts_lower = text[:1].islower()
                continuation = starts_lower and (
                    not self._has_terminal(previous_text)
                    or previous_text.endswith((",", ";"))
                )
                if continuation:
                    merged[-1] = Block(
                        type=previous.type,
                        text=previous_text + " " + text,
                        level=previous.level,
                        meta=dict(previous.meta),
                    )
                    continue
            merged.append(Block(type=block.type, text=text, level=block.level, meta=dict(block.meta)))

        repaired: list[Block] = []
        for index, block in enumerate(merged):
            if block.type not in {"paragraph", "quote"}:
                repaired.append(block)
                continue
            text = block.text.strip()
            if not text:
                dropped += 1
                continue
            nxt = merged[index + 1] if index + 1 < len(merged) else None
            structural_next = nxt is not None and nxt.type in {"list", "code", "table", "formula"}
            if text[:1].islower():
                dropped += 1
                continue

            if self._has_terminal(text):
                pass
            elif text.endswith((",", ";")):
                prefix = self._trim_to_last_sentence(text)
                if prefix:
                    text = prefix
                elif structural_next:
                    text = self._structural_intro(text)
                else:
                    stripped = self._DANGLING_PHRASE_RE.sub("", text).rstrip(" ,;:")
                    if stripped != text and len(stripped.split()) >= 8:
                        text = stripped + "."
                    elif self._PURE_LEADIN_RE.search(text) or len(text.split()) < 8:
                        dropped += 1
                        continue
                    else:
                        text = text[:-1].rstrip() + "."
            elif text.endswith(":"):
                if structural_next:
                    pass
                elif self._looks_like_formula(text) or "=" in text:
                    dropped += 1
                    continue
                else:
                    text = text[:-1].rstrip() + "."
            elif not self._has_terminal(text):
                prefix = self._trim_to_last_sentence(text)
                if structural_next:
                    text = self._structural_intro(text)
                elif prefix:
                    text = prefix
                elif (
                    self._looks_like_formula(text)
                    or self._PURE_LEADIN_RE.search(text)
                    or self._DANGLING_WORD_RE.search(text)
                    or len(text.split()) < 6
                ):
                    dropped += 1
                    continue
                else:
                    text = text.rstrip(" ,;:") + "."

            if not text or text[:1].islower():
                dropped += 1
                continue
            repaired.append(Block(type=block.type, text=text, level=block.level, meta=dict(block.meta)))

        # A heading with no content before the next heading is an empty section.
        content_types = {"paragraph", "quote", "list", "code", "table"}
        kept: list[Block] = []
        for index, block in enumerate(repaired):
            if block.type == "heading":
                nxt = repaired[index + 1] if index + 1 < len(repaired) else None
                if nxt is None or nxt.type == "heading" or nxt.type not in content_types:
                    dropped += 1
                    continue
            kept.append(block)
        return kept, dropped

    _DIRECTORY_HEADING_RE = re.compile(
        r"(?i)\b(?:software|journals?|conferences?|publications?|gallery|"
        r"awards?|libraries|frameworks?|toolkits?|see also|external links|"
        r"further reading|examples?)\b"
    )
    _MAINTENANCE_MARKER_RE = re.compile(
        r"(?i)\s*\[(?:vague|citation needed|clarification needed|"
        r"better source needed|according to whom|who|when|where|why|dubious|"
        r"failed verification|not verified in body|original research)[^\]]*\]"
    )

    @classmethod
    def _is_directory_tail(cls, blocks: list[Block], start: int) -> bool:
        tail = blocks[start:]
        if not tail or not any(block.type == "list" for block in tail):
            return False
        if any(block.type not in {"heading", "list"} for block in tail):
            return False
        items = [
            line.strip()
            for block in tail
            if block.type == "list"
            for line in block.text.splitlines()
            if line.strip()
        ]
        if len(items) < 3:
            return False
        words = sum(len(item.lstrip("-* ").split()) for item in items) / len(items)
        return words <= 7.0

    @classmethod
    def _trim_low_value_tail(cls, blocks: list[Block]) -> tuple[list[Block], int]:
        """Remove trailing software/journal/conference directory sections."""
        trimmed = blocks
        removed = 0
        while True:
            candidates = [
                index
                for index, block in enumerate(trimmed)
                if block.type == "heading"
                and cls._DIRECTORY_HEADING_RE.search(block.text)
                and cls._is_directory_tail(trimmed, index)
            ]
            if not candidates:
                break
            start = min(candidates)
            removed += len(trimmed) - start
            trimmed = trimmed[:start]
        return trimmed, removed

    def _clean_prose(self, text: str, pii_counts: dict[str, int] | None = None) -> str:
        if not text:
            return ""
        kept: list[str] = []
        for raw_line in re.split(r"\n+", text):
            line = normalize_prose(
                raw_line,
                repair_spacing=self.repair_spacing,
                fix_unicode_text=self.fix_unicode,
            )
            line = strip_inline_citations(line)
            line = self._MAINTENANCE_MARKER_RE.sub("", line)
            line = re.sub(r"\[Note,\s*([^\]]+)\]", r"Note, \1", line)
            line = re.sub(r"(?<!\.)\.\.(?!\.)", ".", line)
            line = _strip_markup(line)
            if not line or self._is_droppable_line(line):
                continue
            if is_reference_heading(line) or is_reference_line(line):
                continue
            # Convert inline MediaWiki/LaTeX math to readable text.
            line = self._clean_math_inline(line)
            line = self._formula_sentence_break(line)
            if not line:
                continue
            if len(line.split()) < 2 and not re.search(r"[A-Za-z]{2,}", line):
                continue
            if self.redact_pii:
                line, counts = self.redactor.redact(line)
                if pii_counts is not None:
                    for key, value in counts.items():
                        pii_counts[key] = pii_counts.get(key, 0) + value
            kept.append(line)
        # Join wrapped lines into a paragraph.
        value = " ".join(kept).strip()
        value = re.sub(r"\s+", " ", value)
        # A redaction marker or a bare URL makes the sentence unusable, so the
        # whole sentence is dropped instead of shipping "[REDACTED]" noise.
        value = _drop_sentences_with(value, _REDACTION_MARKER_RE)
        value = _drop_sentences_with(value, _URL_RE)
        value = _drop_sentences_with(value, _SPONSOR_SENTENCE_RE)
        return value.strip()

    def _clean_list(self, text: str, pii_counts: dict[str, int] | None = None) -> str:
        items: list[str] = []
        for raw_line in (text or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            line = re.sub(r"^(?:[-*+•]|\d+[.)])\s*", "", line)
            line = normalize_prose(
                line,
                repair_spacing=self.repair_spacing,
                fix_unicode_text=self.fix_unicode,
            )
            line = strip_inline_citations(line)
            line = _strip_markup(line)
            if not line or self._is_droppable_line(line) or is_reference_line(line):
                continue
            if is_reference_heading(line):
                continue
            line = self._clean_math_inline(line)
            if not line:
                continue
            if self.redact_pii:
                line, counts = self.redactor.redact(line)
                if pii_counts is not None:
                    for key, value in counts.items():
                        pii_counts[key] = pii_counts.get(key, 0) + value
            # Link-only and redacted list items carry no training signal.
            if _URL_RE.search(line) or _REDACTION_MARKER_RE.search(line):
                continue
            items.append("- " + line)
        return "\n".join(items)

    @staticmethod
    def _formula_sentence_break(text: str) -> str:
        """Put a sentence break between an inline formula and following prose.

        Extraction sometimes yields ``... = softmax(x W + b) The matrix ...``
        or ``... = [0, 1] M The token ...``.  If the current sentence contains
        math operators, a following capitalised sentence start needs a period;
        a lone trailing matrix variable before it is discarded.
        """
        starters = (
            r"(?:The|This|These|Those|It|They|We|A|An|Each|Most|Many|"
            r"Some|All|Both|In|For|However|Therefore|Then|Here|There|"
            r"Its|Their|Consider|Let|Equivalently|Alternatively|"
            r"Specifically|When|If|While|Because|Since|Although|Though)"
        )
        math_re = re.compile(r"[=+×÷∑∇∈≤≥↦→∂√∞≈≠]")

        def before(match: re.Match) -> str:
            prefix = text[:match.start()]
            return re.split(r"(?<=[.!?])\s+", prefix)[-1] if prefix else ""

        def bracket_break(match: re.Match) -> str:
            return ". " if math_re.search(before(match)) else match.group(0)

        def variable_break(match: re.Match) -> str:
            return ". " if math_re.search(before(match)) else match.group(0)

        def token_break(match: re.Match) -> str:
            # Formula prose often ends with a short variable such as "z m"
            # before the next sentence.  Long words ("and", "the") are ordinary
            # prose and must not be split.
            prefix = text[:match.start()]
            if not math_re.search(before(match)):
                return match.group(0)
            previous = re.search(r"(\S+)\s*$", prefix)
            if (
                previous
                and len(previous.group(1)) <= 2
                and re.fullmatch(r"[A-Za-z0-9α-ωΑ-Ω]+", previous.group(1))
            ):
                return ". "
            return match.group(0)

        def phrase_break(match: re.Match) -> str:
            prefix = text[:match.start()].rstrip()
            if math_re.search(before(match)) and prefix.endswith((")", "]", "}")):
                return ". In other words"
            return match.group(0)

        text = re.sub(rf"(?<=[\]\)\}}])\s+(?={starters}\b)", bracket_break, text)
        text = re.sub(rf"\s+\b[A-Z]\s+(?={starters}\b)", variable_break, text)
        text = re.sub(rf"\s+(?={starters}\b)", token_break, text)
        text = re.sub(r"\s+(in other words\b)", phrase_break, text)
        return text

    @staticmethod
    def _clean_math_inline(text: str) -> str:
        if not text:
            return text
        if "{\\displaystyle" in text or "{\\textstyle" in text:
            from .normalize import strip_math_wrappers

            text = strip_math_wrappers(text)
        if re.search(r"\\(?:frac|sum|delta|nabla|alpha|beta|gamma|theta|lambda|sigma|cdot|times|le|ge|neq|approx|rightarrow|partial|infty)", text):
            text = math_to_text(text)
        # Math conversion can reintroduce spaces before punctuation/brackets.
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        text = re.sub(r"([(\[])\s+", r"\1", text)
        text = re.sub(r"\s+([)\]])", r"\1", text)
        return text


# --------------------------------------------------------------------------- legacy modifiers


class UnicodeReformatter(Modifier):
    def __init__(self, enabled: bool = True) -> None:
        super().__init__(name="unicode_reformat", enabled=enabled)

    def __call__(self, document):
        if not document.text:
            return document
        document.text = normalize_prose(document.text)
        document.metadata.setdefault("cleaning", []).append(self.name)
        return document


class QuotationUnifier(Modifier):
    def __init__(self, enabled: bool = True) -> None:
        super().__init__(name="quote_unify", enabled=enabled)

    def __call__(self, document):
        if not document.text:
            return document
        document.text = normalize_prose(document.text)
        document.metadata.setdefault("cleaning", []).append(self.name)
        return document


class BoilerplateStripper(Modifier):
    def __init__(self, enabled: bool = True, extra: Iterable[str] | None = None) -> None:
        self.extra = [re.compile(e, re.IGNORECASE) for e in (extra or [])]
        super().__init__(name="boilerplate_strip", enabled=enabled)

    def __call__(self, document):
        if not document.text:
            return document
        lines = []
        for line in document.text.splitlines():
            value = line.strip()
            if not value:
                lines.append("")
                continue
            if is_boilerplate_line(value) or _MAINTENANCE_RE.search(value):
                continue
            if any(pattern.search(value) for pattern in self.extra):
                continue
            lines.append(line)
        document.text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
        document.metadata.setdefault("cleaning", []).append(self.name)
        return document


class TechnicalArtifactCleaner(Modifier):
    def __init__(self, enabled: bool = True) -> None:
        super().__init__(name="technical_artifact_clean", enabled=enabled)

    def __call__(self, document):
        if not document.text:
            return document
        lines = []
        previous = ""
        for line in document.text.splitlines():
            value = strip_inline_citations(line.strip())
            if not value:
                continue
            if _MAINTENANCE_RE.search(value) or is_reference_heading(value):
                continue
            key = re.sub(r"\s+", " ", value).casefold()
            if key == previous:
                continue
            previous = key
            lines.append(value)
        document.text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
        document.metadata.setdefault("cleaning", []).append(self.name)
        return document


# --------------------------------------------------------------------------- preset chain


def default_cleaning_chain(
    *,
    fix_unicode: bool = True,
    unify_quotes: bool = True,
    redact_pii: bool = True,
    pii_entities: Iterable[str] | None = None,
    boilerplate_extra: Iterable[str] | None = None,
    **kwargs,
) -> CleaningChain:
    """Build the default block-aware cleaning chain.

    ``**kwargs`` accepts the new policy options (``remove_captions``,
    ``math_policy``, ``language`` …) while keeping the old call signature.
    """
    return CleaningChain(
        fix_unicode=fix_unicode,
        unify_quotes=unify_quotes,
        redact_pii=redact_pii,
        pii_entities=pii_entities,
        boilerplate_extra=boilerplate_extra,
        **kwargs,
    )
