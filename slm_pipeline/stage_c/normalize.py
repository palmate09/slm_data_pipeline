"""Prose normalization, spacing repair and math-to-text conversion."""

from __future__ import annotations

import html as html_lib
import re
import unicodedata

try:
    import ftfy
except Exception:  # noqa: BLE001
    ftfy = None  # type: ignore[assignment]


_GREEK = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε",
    "zeta": "ζ", "eta": "η", "theta": "θ", "iota": "ι", "kappa": "κ",
    "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ", "pi": "π", "rho": "ρ",
    "sigma": "σ", "tau": "τ", "upsilon": "υ", "phi": "φ", "chi": "χ",
    "psi": "ψ", "omega": "ω", "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ",
    "Lambda": "Λ", "Xi": "Ξ", "Pi": "Π", "Sigma": "Σ", "Phi": "Φ",
    "Psi": "Ψ", "Omega": "Ω",
}

_MATH_SYMBOLS = {
    "cdot": "·", "times": "×", "div": "÷", "pm": "±", "mp": "∓",
    "le": "≤", "leq": "≤", "ge": "≥", "geq": "≥", "neq": "≠", "ne": "≠",
    "approx": "≈", "equiv": "≡", "sim": "∼", "propto": "∝",
    "rightarrow": "→", "to": "→", "leftarrow": "←", "leftrightarrow": "↔",
    "Rightarrow": "⇒", "Leftarrow": "⇐", "Leftrightarrow": "⇔",
    "in": "∈", "notin": "∉", "subset": "⊂", "supset": "⊃",
    "cup": "∪", "cap": "∩", "forall": "∀", "exists": "∃",
    "partial": "∂", "nabla": "∇", "infty": "∞", "sum": "∑", "prod": "∏",
    "int": "∫", "sqrt": "√", "ldots": "...", "dots": "...",
    "cdots": "...", "vdots": "⋮", "ddots": "⋱",
}


def fix_unicode(text: str) -> str:
    if not text:
        return text
    if ftfy is not None:
        try:
            text = ftfy.fix_text(text)
        except Exception:  # noqa: BLE001, S110
            pass
    return text


def _replace_wrapper(text: str, command: str) -> str:
    """Replace ``{\\command ...}`` with its inner text, balancing braces."""
    marker = "{\\" + command
    guard = 0
    while guard < 1000:
        guard += 1
        start = text.find(marker)
        if start < 0:
            break
        depth = 0
        end = start
        while end < len(text):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
                if depth == 0:
                    break
            end += 1
        if end >= len(text):
            text = text[:start] + text[start + len(marker):]
            continue
        inner = text[start + len(marker):end].strip()
        text = text[:start] + inner + text[end + 1:]
    return text


def _strip_latex_commands(text: str) -> str:
    # Simple textual commands first.
    for cmd in ("text", "mathrm", "mathbf", "mathit", "mathsf", "operatorname",
                "textbf", "textit", "textrm", "mbox"):
        pattern = re.compile(r"\\" + cmd + r"\{([^{}]*)\}")
        for _ in range(20):
            new = pattern.sub(r"\1", text)
            if new == text:
                break
            text = new
    # Fractions and roots.
    for _ in range(20):
        new = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"\1/\2", text)
        if new == text:
            break
        text = new
    for _ in range(20):
        new = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", text)
        if new == text:
            break
        text = new
    # Spacing / sizing commands.
    text = re.sub(r"\\(?:left|right|big|Big|bigg|Bigg|,|;|!|:|quad|qquad)", "", text)
    text = text.replace("\\ ", " ")

    def replace_command(match: re.Match) -> str:
        name = match.group(1)
        if name in _GREEK:
            return _GREEK[name]
        if name in _MATH_SYMBOLS:
            return _MATH_SYMBOLS[name]
        # Unknown command: keep its name (e.g. \log -> log) rather than
        # leaving a backslash in training text.
        return name

    text = re.sub(r"\\([A-Za-z]+)", replace_command, text)
    # Remaining TeX escapes.
    text = text.replace("\\%", "%").replace("\\&", "&").replace("\\_", "_")
    text = text.replace("\\{", "{").replace("\\}", "}").replace("\\$", "$")
    text = text.replace("\\", "")
    return text


def strip_math_wrappers(text: str) -> str:
    """Remove ``{\\displaystyle ...}`` / ``{\\textstyle ...}`` spans.

    MediaWiki pages usually contain the rendered math *and* a hidden TeX
    annotation.  Keeping both duplicates the expression, so for prose we drop
    the annotation and keep the rendered text around it.
    """
    if not text:
        return ""
    for command in ("displaystyle", "textstyle"):
        marker = "{\\" + command
        guard = 0
        while guard < 1000:
            guard += 1
            start = text.find(marker)
            if start < 0:
                break
            depth = 0
            end = start
            while end < len(text):
                if text[end] == "{":
                    depth += 1
                elif text[end] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                end += 1
            if end >= len(text):
                text = text[:start]
                continue
            text = text[:start] + " " + text[end + 1:]
    return re.sub(r"\s+", " ", text).strip()


def math_to_text(text: str) -> str:
    """Convert MediaWiki/LaTeX math into a plain-text approximation."""
    if not text:
        return ""
    text = _replace_wrapper(text, "displaystyle")
    text = _replace_wrapper(text, "textstyle")
    text = _strip_latex_commands(text)
    # Superscripts/subscripts lose their braces but keep a readable marker.
    text = re.sub(r"\^\{([^{}]*)\}", r"^\1", text)
    text = re.sub(r"_\{([^{}]*)\}", r"_\1", text)
    text = text.replace("{", "").replace("}", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def looks_like_formula(text: str) -> bool:
    value = (text or "").strip()
    if not value:
        return False
    if "\\" in value or "{\\displaystyle" in value:
        return True
    symbols = len(re.findall(r"[=+\-−×÷^_∑∫∂∇∞∈∉≤≥≠≈→←↔()]", value))
    words = len(re.findall(r"[A-Za-z]{2,}", value))
    return symbols >= 4 and symbols > words


def _replace_quotes_dashes(text: str) -> str:
    text = text.translate(str.maketrans({
        "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"',
        "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
        "\u2032": "'", "\u2033": '"', "\u00ab": '"', "\u00bb": '"',
        # En dash is a compound/range joiner ("bias–variance"); em dash marks a
        # parenthetical break.  Mapping both to " - " produced noisy text such
        # as "input - output" and "Yang - Mills", so keep their typographic roles.
        "\u2013": "-", "\u2014": " \u2014 ", "\u2212": "-",
        "\u2026": "...", "\u00a0": " ", "\u2009": " ", "\u202f": " ",
    }))
    # Keep hyphenated ranges compact: 2009 - 2010 -> 2009-2010.
    text = re.sub(r"(?<=\d)\s+-\s+(?=\d)", "-", text)
    # A spaced ASCII hyphen between two real words is a dash, not a minus
    # sign.  Convert it to an em dash; single-letter math operands (x - y)
    # are deliberately left alone.
    text = re.sub(r"(?<=[A-Za-z]{2})\s+-\s+(?=[A-Za-z]{2})", " \u2014 ", text)
    text = re.sub(r'(?<=[A-Za-z])\s+-\s+(?=["\u201c])', " \u2014 ", text)
    text = re.sub(r'(?<=["\u201d])\s+-\s+(?=[A-Za-z])', " \u2014 ", text)
    # A dash immediately followed by an ordinary English word is punctuation,
    # not a mathematical minus sign (e.g. "δ l - from back to front").
    text = re.sub(
        r"\s+-\s+(?=(?:from|to|with|and|or|but|which|where|when|while|"
        r"because|specifically|backwards?|forwards?|means|propagating|using)\b)",
        " \u2014 ",
        text,
    )
    # Drop invisible math/format controls (e.g. FUNCTION APPLICATION after
    # "cos"/"ln") and zero-width characters that survive HTML extraction.
    text = re.sub(r"[\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff]", "", text)
    # A removed citation can leave "unit.. In"; collapse only true doubles and
    # leave ellipses ("...") untouched.
    text = re.sub(r"(?<!\.)\.\.(?!\.)", ".", text)
    return text


def _fix_missing_spaces(text: str) -> str:
    # Missing space after sentence punctuation before a capitalised word.
    text = re.sub(r"([a-z0-9])([.!?])([A-Z][a-z])", r"\1\2 \3", text)
    # Missing space after a colon before a capitalised word.
    text = re.sub(r"([a-z]):([A-Z][a-z])", r"\1: \2", text)
    # Missing space after a closing bracket before a capitalised word.
    text = re.sub(r"([)\]])([A-Z][a-z])", r"\1 \2", text)
    # Common glued words seen in HTML extraction.
    text = re.sub(
        r"\b(learning|network|machine|deep|neural|data|model)(algorithms|models|"
        r"networks|sentence|sentences|processing|training)\b",
        r"\1 \2", text, flags=re.IGNORECASE,
    )
    text = re.sub(r"\bor(intra)\b", r"or \1", text)
    # A short lowercase function word glued to a following capitalised word.
    text = re.sub(
        r"\b(of|the|and|to|in|for|with|on|at|by|as|or|is|are|was|were|be|been|"
        r"this|that|these|those|from|into|than|then|when|where|which|while|who|"
        r"whom|whose|why|how)(?=[A-Z][a-z])",
        r"\1 ", text,
    )
    return text


_SPACED_TERMS = {
    # Words that MediaWiki math/HTML sometimes emits as individual letters
    # separated by spaces, e.g. "s o f t m a x".  Only exact known words are
    # joined; math variables such as "w i j" stay untouched.
    "onions", "potatoes", "burger",
    "Embed", "UnEmbed", "softmax", "Sublayer", "LayerNorm", "FFN",
    "Attention", "Multihead", "MultiheadAttention", "diag",
    "encoder", "decoder", "Encoder", "Decoder", "relu", "ReLU",
    "sigmoid", "tanh", "perceptron", "neural", "network", "learning",
    "machine", "model", "models", "input", "output", "weights", "tokens",
}
_SPACED_RUN_RE = re.compile(r"\b(?:[A-Za-z]\s+){2,}[A-Za-z]\b")


def _rejoin_spaced_terms(text: str) -> str:
    def replace(match: re.Match) -> str:
        candidate = re.sub(r"\s+", "", match.group(0))
        return candidate if candidate in _SPACED_TERMS else match.group(0)

    return _SPACED_RUN_RE.sub(replace, text)


def normalize_prose(
    text: str,
    *,
    repair_spacing: bool = True,
    fix_unicode_text: bool = True,
) -> str:
    """Normalize a prose block without touching code or formula blocks."""
    if not text:
        return ""
    text = html_lib.unescape(text)
    text = unicodedata.normalize("NFC", text)
    if fix_unicode_text:
        text = fix_unicode(text)
    text = text.replace("\u200b", "").replace("\ufeff", "").replace("\u2060", "")
    # Brand marks add noise to prose; they are not semantic content.
    text = text.replace("®", "").replace("™", "").replace("℠", "")
    text = _replace_quotes_dashes(text)
    text = _rejoin_spaced_terms(text)
    # Remove control characters except newlines/tabs.
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Join hyphenated line breaks.
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    text = re.sub(r"\s+", " ", text)
    # A lowercase coordinating conjunction after a period is a continuation,
    # not a new sentence: "LSTM. but are better" -> "LSTM, but are better".
    text = re.sub(r"[.!?]\s+(and|but|or|so|yet|nor)\b", r", \1", text)
    if repair_spacing:
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        text = re.sub(r"([(\[])\s+", r"\1", text)
        text = re.sub(r"\s+([)\]])", r"\1", text)
        text = _fix_missing_spaces(text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def normalize_heading(
    text: str,
    *,
    repair_spacing: bool = True,
    fix_unicode_text: bool = True,
) -> str:
    text = normalize_prose(
        text,
        repair_spacing=repair_spacing,
        fix_unicode_text=fix_unicode_text,
    )
    text = re.sub(r"^#{1,6}\s*", "", text)
    text = text.strip().rstrip(":").strip()
    return text
