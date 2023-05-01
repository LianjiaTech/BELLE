# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import time
from typing import Dict, Optional

import sacremoses  # type: ignore

from cc_net import jsonql, text_normalizer


class RobustTokenizer(jsonql.Transformer):
    """Moses tokenizer with the expected preprocessing."""

    LANG_WITHOUT_ACCENT = {"en", "my"}

    def __init__(self, lang: str):
        super().__init__()
        self.lang = lang
        self.moses = sacremoses.MosesTokenizer(lang)
        self.rm_accent = lang in self.LANG_WITHOUT_ACCENT
        self.ready = True

    def do(self, text: str):
        text = text_normalizer.normalize(
            text, accent=self.rm_accent, case=False, numbers=False, punct=True
        )
        text = text_normalizer.normalize_spacing_for_tok(text, language=self.lang)
        return self.moses.tokenize(text, return_str=True, escape=False)


class DocTokenizer(jsonql.Transformer):
    """Tokenize the text found in `output_field and store the result in `output_field`."""

    def __init__(
        self,
        field: str,
        output_field: str = "tokenized",
        language_field: str = "language",
    ):
        super().__init__()
        self.field = field
        self.output_field = output_field
        self.language_field = language_field
        self.n_docs = 0
        self.tokenizers: Dict[str, RobustTokenizer] = {}

    def get_tokenizer(self, lang: str) -> Optional[RobustTokenizer]:
        cache = self.tokenizers
        if lang in cache:
            return cache[lang]
        if lang in ("th", "zh", "ja"):
            # TODO find a tokenizer for those languages
            return None

        cache[lang] = RobustTokenizer(lang)
        return cache[lang]

    def do(self, document):
        lang = document[self.language_field]
        tok = self.get_tokenizer(lang)
        if not tok:
            return document

        self.n_docs += 1
        lines = document[self.field].split("\n")
        tokenized = "\n".join(tok(l) for l in lines)
        document[self.output_field] = tokenized
        return document

    def summary(self):
        delay = (time.time() - self.start_time) / 3600
        speed = self.n_docs / delay
        return [
            f"Tokenized {self.n_docs:_} documents in {delay:.2}h ({speed:.1} doc/s)."
        ]
