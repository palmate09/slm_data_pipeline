"""Conservative PII redaction.

The previous fallback treated any digit string with separators as a phone
number, destroying ISBNs, DOIs, arXiv ids, page ranges and dates.  This module
uses ``phonenumbers`` (when available) plus strict validators and context
exclusions.  It redacts direct identifiers by default; PERSON/ADDRESS require
Presidio and are opt-in.
"""

from __future__ import annotations

import re

DEFAULT_ENTITIES = (
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "US_SSN",
    "CREDIT_CARD",
    "IBAN_CODE",
)

# A trailing "." is a sentence terminator, not part of the domain, so the
# look-ahead must not reject it: "email us at contact@example.com." is still
# an email address.
_EMAIL_RE = re.compile(
    r"(?<![\w.%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![\w-])"
)
_SSN_RE = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
_CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
_IBAN_RE = re.compile(r"(?<![A-Z0-9])[A-Z]{2}\d{2}[A-Z0-9]{10,30}(?![A-Z0-9])")
_PHONE_FALLBACK_RE = re.compile(
    r"(?<![\w.])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)[\s.-]?)?"
    r"\d{3,4}[\s.-]\d{3,4}(?:[\s.-]\d{3,4})?(?![\w.])"
)

_EXCLUDE_CONTEXT_RE = re.compile(
    r"(?i)(?:ISBN|ISSN|doi\s*:|arXiv\s*:|pp?\.|Retrieved|Archived|"
    r"S2CID|OCLC|PMID|PMC|JSTOR|Vol\.|No\.|Fig\.|Eq\.|version|v\d|"
    r"params|parameters|billion|million|thousand|d_model|layers?|heads?|"
    r"hidden|batch|embed|context length|vocab|tokens|steps|epochs|"
    r"hidden states|dim|size)"
)
_YEAR_RANGE_RE = re.compile(r"^(?:19|20)\d{2}\s*[-–]\s*(?:19|20)\d{2}$")
# "1.542B params", "117M parameters", "5% accuracy" are counts, not phones.
_UNIT_SUFFIX_RE = re.compile(r"[A-Za-z%]")


def _luhn_ok(value: str) -> bool:
    digits = [int(c) for c in re.sub(r"\D", "", value)]
    if len(digits) < 13 or len(digits) > 19:
        return False
    checksum = 0
    parity = len(digits) % 2
    for i, digit in enumerate(digits):
        if i % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


def _iban_ok(value: str) -> bool:
    compact = re.sub(r"\s+", "", value).upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{10,30}", compact):
        return False
    rearranged = compact[4:] + compact[:4]
    digits = "".join(str(int(c, 36)) for c in rearranged)
    try:
        return int(digits) % 97 == 1
    except ValueError:
        return False


def _context(text: str, start: int, end: int, window: int = 35) -> str:
    return text[max(0, start - window): min(len(text), end + window)]


class PiiRedactor:
    def __init__(
        self,
        *,
        entities: list[str] | tuple[str, ...] | None = None,
        replace_with: str = "[REDACTED]",
        use_presidio: bool = False,
        language: str = "en",
    ) -> None:
        requested = list(entities) if entities else list(DEFAULT_ENTITIES)
        # PERSON/ADDRESS are handled only by Presidio and are opt-in.
        self.entities = requested
        self.replace_with = replace_with
        self.use_presidio = use_presidio
        self.language = language
        self._presidio = None
        self._presidio_failed = False

    def redact(self, text: str) -> tuple[str, dict[str, int]]:
        if not text:
            return text, {}
        counts: dict[str, int] = {}
        text, counts["EMAIL_ADDRESS"] = self._sub_count(
            _EMAIL_RE, text, "EMAIL_ADDRESS", counts.get("EMAIL_ADDRESS", 0)
        ) if "EMAIL_ADDRESS" in self.entities else (text, 0)

        if "US_SSN" in self.entities:
            text, counts["US_SSN"] = self._sub_count(_SSN_RE, text, "US_SSN", counts.get("US_SSN", 0))

        if "CREDIT_CARD" in self.entities:
            text, counts["CREDIT_CARD"] = self._sub_validated(
                _CARD_RE, text, _luhn_ok, "CREDIT_CARD", counts.get("CREDIT_CARD", 0)
            )

        if "IBAN_CODE" in self.entities:
            text, counts["IBAN_CODE"] = self._sub_validated(
                _IBAN_RE, text, _iban_ok, "IBAN_CODE", counts.get("IBAN_CODE", 0)
            )

        if "PHONE_NUMBER" in self.entities:
            text, counts["PHONE_NUMBER"] = self._redact_phones(text, counts.get("PHONE_NUMBER", 0))

        # Optional Presidio for PERSON/ADDRESS or user-requested entities.
        extra_entities = [
            e for e in self.entities
            if e not in {"EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN", "CREDIT_CARD", "IBAN_CODE"}
        ]
        if (self.use_presidio or extra_entities) and not self._presidio_failed:
            text, presidio_counts = self._redact_presidio(text, extra_entities or None)
            for key, value in presidio_counts.items():
                counts[key] = counts.get(key, 0) + value

        return text, {k: v for k, v in counts.items() if v}

    # ------------------------------------------------------------------ helpers
    def _sub_count(self, pattern: re.Pattern, text: str, label: str, existing: int) -> tuple[str, int]:
        def repl(match: re.Match) -> str:
            nonlocal existing
            existing += 1
            return self.replace_with

        return pattern.sub(repl, text), existing

    def _sub_validated(
        self,
        pattern: re.Pattern,
        text: str,
        validator,
        label: str,
        existing: int,
    ) -> tuple[str, int]:
        def repl(match: re.Match) -> str:
            nonlocal existing
            value = match.group(0)
            if not validator(value):
                return value
            # Do not rewrite identifiers embedded in bibliographic context.
            ctx = _context(text, match.start(), match.end())
            if _EXCLUDE_CONTEXT_RE.search(ctx):
                return value
            existing += 1
            return self.replace_with

        return pattern.sub(repl, text), existing

    def _redact_phones(self, text: str, existing: int) -> tuple[str, int]:
        matches: list[tuple[int, int]] = []
        try:
            import phonenumbers
            from phonenumbers import Leniency

            for match in phonenumbers.PhoneNumberMatcher(
                text, "US", leniency=Leniency.POSSIBLE
            ):
                raw = match.raw_string.strip()
                if _YEAR_RANGE_RE.fullmatch(raw):
                    continue
                ctx = _context(text, match.start, match.end)
                if _EXCLUDE_CONTEXT_RE.search(ctx):
                    continue
                # Require a plausible phone shape; avoid bare 4-digit years.
                digits = re.sub(r"\D", "", raw)
                if len(digits) < 7:
                    continue
                # "1.542B params" / "117M" are model sizes, not phone numbers.
                tail = text[match.end: match.end + 1]
                if tail and _UNIT_SUFFIX_RE.match(tail):
                    continue
                matches.append((match.start, match.end))
        except Exception:  # noqa: BLE001 - optional dependency
            for match in _PHONE_FALLBACK_RE.finditer(text):
                raw = match.group(0).strip()
                digits = re.sub(r"\D", "", raw)
                if len(digits) < 7 or _YEAR_RANGE_RE.fullmatch(raw):
                    continue
                ctx = _context(text, match.start(), match.end())
                if _EXCLUDE_CONTEXT_RE.search(ctx):
                    continue
                tail = text[match.end(): match.end + 1]
                if tail and _UNIT_SUFFIX_RE.match(tail):
                    continue
                matches.append((match.start(), match.end()))

        if not matches:
            return text, existing
        # Deduplicate and apply from the end so offsets remain valid.
        unique = sorted(set(matches), reverse=True)
        for start, end in unique:
            text = text[:start] + self.replace_with + text[end:]
            existing += 1
        return text, existing

    def _redact_presidio(self, text: str, entities: list[str] | None) -> tuple[str, dict[str, int]]:
        if self._presidio is None:
            try:
                from presidio_analyzer import AnalyzerEngine
                from presidio_anonymizer import AnonymizerEngine

                self._presidio = (AnalyzerEngine(), AnonymizerEngine())
            except Exception:  # noqa: BLE001 - no models / no package
                self._presidio_failed = True
                return text, {}
        analyzer, anonymizer = self._presidio
        try:
            results = analyzer.analyze(
                text=text,
                entities=entities,
                language=self.language,
            )
            if not results:
                return text, {}
            counts: dict[str, int] = {}
            for item in results:
                counts[item.entity_type] = counts.get(item.entity_type, 0) + 1
            anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
            return anonymized.text, counts
        except Exception:  # noqa: BLE001
            self._presidio_failed = True
            return text, {}


class PiiModifier:
    """Backward-compatible facade used by the old cleaning chain."""

    name = "pii_redact"

    def __init__(
        self,
        *,
        enabled: bool = True,
        entities=None,
        replace_with: str = "[REDACTED]",
        force_presidio: bool = False,
    ) -> None:
        self.enabled = enabled
        self.redactor = PiiRedactor(
            entities=entities,
            replace_with=replace_with,
            use_presidio=force_presidio,
        )

    def __call__(self, document):
        if not self.enabled or not document.text:
            return document
        document.text, counts = self.redactor.redact(document.text)
        if counts:
            document.metadata["pii_counts"] = counts
        document.metadata.setdefault("cleaning", []).append(self.name)
        return document
