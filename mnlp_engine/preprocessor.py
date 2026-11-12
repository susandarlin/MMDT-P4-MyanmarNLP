"""
Myanmar Unicode normalization.

Pipeline (normalize_myanmar):
    1. Zawgyi check  -> convert to Unicode if needed
    2. Unicode NFC normalization
    3. Remove invisible / non-text characters
    4. Fix common typing-order mistakes
    5. Collapse whitespace

Only the Python standard library is required. If `myanmartools` (detector) or
`pyidaungsu` / `rabbit` (converters) are installed they are used automatically;
otherwise a regex heuristic is used and text is left unchanged.
"""

import re
import unicodedata

# 1. Zawgyi detection & conversion
_ZAWGYI_HINT = re.compile(
    r"(?:^|[\s၊။])[ေျ]"
    r"|[ၠ-႗]"
    r"|်[ျ-ှ]"
    r"|္(?![က-အ])"
)

try:  # optional: Google's probabilistic detector
    from myanmartools import ZawgyiDetector
    _DETECTOR = ZawgyiDetector()
except Exception:  # library not installed
    _DETECTOR = None


def is_zawgyi(text, threshold=0.95):
    """Return True if `text` looks like Zawgyi-encoded Myanmar."""
    if not text or not re.search(r"[က-႟]", text):
        return False
    if _DETECTOR is not None:
        return _DETECTOR.get_zawgyi_probability(text) > threshold
    return bool(_ZAWGYI_HINT.search(text))


def zawgyi_to_unicode(text):
    """Convert Zawgyi -> Unicode using whichever converter is installed."""
    try:
        import pyidaungsu as pds
        return pds.cvt2uni(text)
    except Exception:
        pass
    try:
        from rabbit import Rabbit
        return Rabbit.zg2uni(text)
    except Exception:
        pass
    return text  # no converter available: keep the original text


# ---------------------------------------------------------------------------
# 3. Invisible / non-text characters
# ---------------------------------------------------------------------------
_INVISIBLE = re.compile(r"[​‌‍⁠﻿­]")   # ZWSP, ZWNJ, ZWJ, WJ, BOM, soft hyphen
_CONTROL = re.compile(r"[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]")
_ODD_SPACES = re.compile(r"[  -   　\t\r\n]")

# ---------------------------------------------------------------------------
# 4. Typing-order fixes (NFC does NOT fix these, because the marks have ccc=0)
# ---------------------------------------------------------------------------
_CONS = "က-အ"            # က - အ
_MEDIAL = "ျ-ှ"          # ျ ြ ွ ှ


def _fix_typing_order(text):
    # a) ေ typed BEFORE its consonant (visual order):  ေ + မ(+medials) -> မ(+medials) + ေ
    text = re.sub(
        rf"(^|[^{_CONS}္{_MEDIAL}])ေ([{_CONS}][{_MEDIAL}]*)",
        r"\1\2" + "ေ", text)
    # b) medials in the wrong order -> canonical ျ ြ ွ ှ (sort by code point)
    text = re.sub(rf"[{_MEDIAL}]{{2,}}",
                  lambda m: "".join(sorted(m.group(0))), text)
    # c) ု typed before ိ / ီ  ->  ိ ု
    text = re.sub(r"([ုူ])([ိီဲ])", r"\2\1", text)
    # d) ့ typed before ု / ူ / ံ  ->  vowel first, dot-below last
    text = re.sub(r"့([ုူံ])", r"\1" + "့", text)
    # e) same mark typed twice (e.g. ့့ or ််) -> once
    text = re.sub(r"([ါ-ှ])\1+", r"\1", text)
    # f) digit ၀ (zero) used as letter ဝ (wa) inside a word
    text = re.sub(r"(?<=[က-ဿ])၀(?![၀-၉])", "ဝ", text)
    text = re.sub(r"(?<![၀-၉])၀(?=[က-အါ-ှ])", "ဝ", text)
    return text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def normalize_myanmar(text):
    """Step 1: return a clean, canonical Unicode version of `text`."""
    if text is None:
        return ""
    text = str(text)
    if is_zawgyi(text):                               # 1. Zawgyi -> Unicode
        text = zawgyi_to_unicode(text)
    text = unicodedata.normalize("NFC", text)         # 2. canonical Unicode
    text = _INVISIBLE.sub("", text)                   # 3. invisible chars
    text = _CONTROL.sub("", text)
    text = _ODD_SPACES.sub(" ", text)
    text = _fix_typing_order(text)                    # 4. typing order
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r" {2,}", " ", text).strip()        # 5. one space
    return text


if __name__ == "__main__":
    samples = {
        "invisible + spaces": "မြန်​မာ   နိုင်ငံ",
        "ေ typed first":      "ေကျာင်း",          # ေ + က + ျ ...
        "ု before ိ":         "နုိင်",         # နုိင် -> နိုင်
        "medial order":       "မွျား",        # ွ before ျ
        "wa as zero":         "၀န်",               # ၀န် -> ဝန်
        "double mark":        "ပါ့့",
    }
    for name, s in samples.items():
        out = normalize_myanmar(s)
        print(f"{name:20} {s!r:28} -> {out!r:24} {out}")
