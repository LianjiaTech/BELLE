# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import re
import unicodedata

UNICODE_PUNCT = {
    "，": ",",
    "。": ".",
    "、": ",",
    "„": '"',
    "”": '"',
    "“": '"',
    "«": '"',
    "»": '"',
    "１": '"',
    "」": '"',
    "「": '"',
    "《": '"',
    "》": '"',
    "´": "'",
    "∶": ":",
    "：": ":",
    "？": "?",
    "！": "!",
    "（": "(",
    "）": ")",
    "；": ";",
    "–": "-",
    "—": " - ",
    "．": ". ",
    "～": "~",
    "’": "'",
    "…": "...",
    "━": "-",
    "〈": "<",
    "〉": ">",
    "【": "[",
    "】": "]",
    "％": "%",
    "►": "-",
}

UNICODE_PUNCT_RE = re.compile(f"[{''.join(UNICODE_PUNCT.keys())}]")


def replace_unicode_punct(text: str) -> str:
    return "".join((UNICODE_PUNCT.get(c, c) for c in text))


def remove_unicode_punct(text: str) -> str:
    """More aggressive version of replace_unicode_punct but also faster."""
    return UNICODE_PUNCT_RE.sub("", text)


def strip_accents(line: str) -> str:
    """Strips accents from a piece of text."""
    nfd = unicodedata.normalize("NFD", line)
    output = [c for c in nfd if unicodedata.category(c) != "Mn"]
    if len(output) == line:
        return line
    return "".join(output)


# Build a regex matching all control characters.
NON_PRINTING_CHARS_RE = re.compile(
    f"[{''.join(map(chr, list(range(0,32)) + list(range(127,160))))}]"
)
DIGIT_RE = re.compile(r"\d")
PUNCT_OR_NON_PRINTING_CHARS_RE = re.compile(
    (UNICODE_PUNCT_RE.pattern + NON_PRINTING_CHARS_RE.pattern).replace("][", "")
)


def remove_non_printing_char(text: str) -> str:
    return NON_PRINTING_CHARS_RE.sub("", text)


def normalize_spacing_for_tok(text: str, language: str = "en") -> str:
    res = (
        text.replace("\r", "")
        # remove extra spaces
        .replace("(", " (")
        .replace(")", ") ")
        .replace(" +", " ")
    )
    res = re.sub(r"\) ([\.\!\:\?\;\,])", r"\)\1", res)
    res = res.replace("( ", "(").replace(" )", ")")
    res = re.sub(r"(\d) \%", r"\1\%", res)
    res = res.replace(" :", ":").replace(" ;", ";")
    res = res.replace("`", "'").replace("''", ' " ')

    res = (
        res.replace("„", '"')
        .replace("“", '"')
        .replace("”", '"')
        .replace("–", "-")
        .replace("—", " - ")
        .replace(" +", " ")
        .replace("´", "'")
        .replace("([a-z])‘([a-z])", r"\1'\2/")
        .replace("([a-z])’([a-z])", r"\1'\2/")
        .replace("‘", '"')
        .replace("‚", '"')
        .replace("’", '"')
        .replace("''", '"')
        .replace("´´", '"')
        .replace("…", "...")
        # French quotes
        .replace(" « ", ' "')
        .replace("« ", '"')
        .replace("«", '"')
        .replace(" » ", '" ')
        .replace(" »", '"')
        .replace("»", '"')
        # handle pseudo-spaces
        .replace(" %", "%")
        .replace("nº ", "nº ")
        .replace(" :", ":")
        .replace(" ºC", " ºC")
        .replace(" cm", " cm")
        .replace(" ?", "?")
        .replace(" !", "!")
        .replace(" ;", ";")
        .replace(", ", ", ")
        .replace(" +", " ")
        .replace("．", ". ")
    )
    # English "quotation," followed by comma, style
    if language == "en":
        res = re.sub(r"\"([,\.]+)", r"\1\"", res)
    # Czech is confused
    elif language == "cs" or language == "cz":
        pass
    # German/Spanish/French "quotation", followed by comma, style
    else:
        res = res.replace(',"', '",')
        res = re.sub(
            r"(\.+)\"(\s*[^<])", r"\"\1\2", res
        )  # don't fix period at end of sentence

    if (
        language == "de"
        or language == "es"
        or language == "cz"
        or language == "cs"
        or language == "fr"
    ):
        res = re.sub(r"(\d) (\d)", r"\1,\2", res)
    else:
        res = re.sub(r"(\d) (\d)", r"\1.\2", res)
    return res


def normalize(line: str, accent=True, case=True, numbers=True, punct=1) -> str:
    line = line.strip()
    if not line:
        return line
    if case:
        line = line.lower()
    if accent:
        line = strip_accents(line)
    if numbers:
        line = DIGIT_RE.sub("0", line)
    if punct == 1:
        line = replace_unicode_punct(line)
    elif punct == 2:
        line = remove_unicode_punct(line)
    line = remove_non_printing_char(line)
    return line


def slow_normalize_for_dedup(line: str) -> str:
    return normalize(line, accent=False, case=True, numbers=True, punct=2)


def normalize_for_dedup(line: str) -> str:
    line = line.strip()
    if not line:
        return line
    # case
    line = line.lower()
    # numbers
    line = DIGIT_RE.sub("0", line)
    line = PUNCT_OR_NON_PRINTING_CHARS_RE.sub("", line)
    return line
