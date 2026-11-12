"""
Syllable breaking, punctuation stripping and stopword filtering.

Syllable rule: Ye Kyaw Thu's "sylbreak" (https://github.com/ye-kyaw-thu/sylbreak)
A new syllable starts at
  - a consonant (က-အ) that is NOT after a virama ္ (stacked letter, e.g. ဗုဒ္ဓ)
    and NOT followed by asat ် or virama ္ (e.g. န် in မြန်), or
    by ့ + ် (NFC order, e.g. င့် in နှင့်)  <- small fix to the original rule
  - an independent vowel / digit / English letter / punctuation / space.
"""

import re
from .preprocessor import normalize_myanmar

# ---------------------------------------------------------------------------
# 1. Sylbreak rule
# ---------------------------------------------------------------------------
MY_CONSONANT = r"က-အ"
EN_CHAR = r"a-zA-Z0-9"
OTHER_CHAR = r"ဣဤဥဦဧဩဪဿ၌၍၏၀-၉၊။!-/:-@\[-`{-~\s"
SS_SYMBOL = r"္"     # virama (stacking)
A_THAT = r"်"        # asat (killer)

DOT_BELOW = r"့"     # after NFC, ့ is stored BEFORE ် (e.g. နှင့် = င + ့ + ်)

BREAK_PATTERN = re.compile(
    r"((?<!" + SS_SYMBOL + r")[" + MY_CONSONANT + r"](?!" + DOT_BELOW + r"?[" + A_THAT + SS_SYMBOL + r"])"
    + r"|[" + EN_CHAR + r"]+"            # keep English words / numbers whole
    + r"|[၀-၉]+"                          # keep Myanmar numbers whole (၁၉၄၈)
    + r"|[" + OTHER_CHAR + r"])"
)


def syllable_break(text):
    """Insert a space before every syllable start and split: 'မြန်မာ' -> ['မြန်','မာ']."""
    return BREAK_PATTERN.sub(r" \1", text).split()


# ---------------------------------------------------------------------------
# 2. Punctuation & special-symbol stripping
# ---------------------------------------------------------------------------
PUNCT_PATTERN = re.compile(r"[။၊!-/:-@\[-`{-~“”‘’…–—«»•]")


def strip_punctuation(text):
    """Replace Myanmar (။ ၊) and ASCII/typographic punctuation with a space."""
    return PUNCT_PATTERN.sub(" ", text)


# ---------------------------------------------------------------------------
# 3. Stopword filtering
# ---------------------------------------------------------------------------
# High-frequency grammatical particles, exactly as listed in the Task 2 instructions.
# Negation words (မ, မဟုတ်, မရှိ) are deliberately NOT here: they decide
# SUPPORTS vs REFUTES.
STOP = {"သည်", "၏", "ကို", "တွင်", "နှင့်", "ဖြစ်သည်"}


def _split_stopwords(stopwords):
    """Split stopwords into single-syllable words and multi-syllable sequences.

    After sylbreak every token is ONE syllable, so a multi-syllable stopword
    like ဖြစ်သည် (ဖြစ် + သည်) can never equal a single token. It must be
    matched as a sequence of consecutive syllables instead.
    """
    single, multi = set(), []
    for w in stopwords:
        syls = tuple(syllable_break(w))
        if len(syls) == 1:
            single.add(syls[0])
        elif syls:
            multi.append(syls)
    multi.sort(key=len, reverse=True)     # match longest sequence first
    return single, multi


def remove_stopwords(tokens, stopwords=STOP):
    """Remove single-syllable stopwords and multi-syllable stopword sequences.

    e.g. ['ပုဂံ', 'ဖြစ်', 'သည်'] -> ['ပုဂံ']   (ဖြစ်သည် removed as a pair)
         ['ဖြစ်', 'ပေါ်']       -> ['ဖြစ်', 'ပေါ်'] (ဖြစ် alone is kept)
    """
    single, multi = _split_stopwords(stopwords)
    out, i = [], 0
    while i < len(tokens):
        for seq in multi:
            if tuple(tokens[i:i + len(seq)]) == seq:
                i += len(seq)
                break
        else:
            if tokens[i] not in single:
                out.append(tokens[i])
            i += 1
    return out


# ---------------------------------------------------------------------------
# Full tokenizer (Step 1 + Step 2) - use this everywhere
# ---------------------------------------------------------------------------
def tokenize(text, remove_stop=True):
    """normalize -> strip punctuation -> sylbreak -> (optional) stopword filter."""
    text = normalize_myanmar(text)          # Step 1: wash
    text = strip_punctuation(text)          # remove ။ ၊ and symbols
    tokens = syllable_break(text)           # Step 2: cut into syllables
    if remove_stop:
        tokens = remove_stopwords(tokens)
    return tokens


if __name__ == "__main__":
    for s in ["မြန်မာနိုင်ငံ",
              "ဗုဒ္ဓဘာသာ",
              "သုတ္တန်ပိဋကတွင် နိကာယ်ငါးရပ် ပါဝင်သည်။",
              "ပုဂံသည် ဂျပန်တွင် တည်ရှိသည် မဟုတ်ပါ။",
              "ပုဂံသည် ရှေးဟောင်းမြို့ဖြစ်သည်။ ဖြစ်ပေါ်လာသည်။"]:
        print(s)
        print("  syllables :", syllable_break(strip_punctuation(normalize_myanmar(s))))
        print("  tokens    :", tokenize(s))
