# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import kenlm  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import sentencepiece  # type: ignore

from cc_net import jsonql, text_normalizer

LMDescriptor = Union[Dict[str, Path], Union[Path, str]]


def get_args():
    parser = argparse.ArgumentParser(
        description="Compute the score of each sentences of a document",
        parents=[jsonql.io_parser()],
    )
    parser.add_argument("--models", type=str)

    parser.add_argument("--sentences", action="store_true", default=False)
    parser.add_argument(
        "--languages", type=str, help="Ignore doc with another language"
    )
    parser.add_argument("--field", type=str, default=None)
    parser.add_argument("--newline", type=str, default="\n")
    return vars(parser.parse_args())


def pp(log_score, length):
    return 10.0 ** (-log_score / length)


class SentencePiece(jsonql.Transformer):
    # Sentence Pieces model have to be read back from disk.
    warning_when_pickling = True

    def __init__(
        self,
        model: Path,
        field: str,
        output_field: str = "tokenized",
        normalize: bool = False,
    ):
        super().__init__()
        self.model = model
        self.field = field
        self.output_field = output_field
        self.normalize = normalize
        self.sp: sentencepiece.SentencePieceProcessor = None

    def _prepare(self):
        if self.sp is not None:
            return
        self.sp = sentencepiece.SentencePieceProcessor()
        self.sp.load(str(self.model))
        return self

    def do(self, document: dict) -> dict:
        text = document[self.field]
        if self.normalize:
            text = text_normalizer.normalize(text)
        tokenized = self.sp.encode_as_pieces(text)
        document[self.output_field] = " ".join(tokenized)
        return document


class MultiSentencePiece(jsonql.Transformer):
    warning_when_pickling = True

    def __init__(
        self,
        models: Union[Path, Dict[str, Path]],
        field: str,
        output_field: str = "tokenized",
        normalize: bool = False,
    ):
        super().__init__()
        self.field = field
        self.output_field = output_field
        self.normalize = normalize
        self._prefetch: Sequence[str] = []

        if isinstance(models, Path):
            self.models = {
                m.name.split(".")[0]: m for m in models.parent.glob(models.name)
            }
        else:
            self.models = models
            self._prefetch = list(models.keys())
        self.sp: Dict[str, sentencepiece.SentencePieceProcessor] = {}

    def _prepare(self) -> None:
        for lang in self._prefetch:
            assert (
                self.get_sp(lang) is not None
            ), f"No model found for {lang} at {self.models.get(lang)}."

    def get_sp(self, lang) -> Optional[sentencepiece.SentencePieceProcessor]:
        sp = self.sp.get(lang)
        if sp is not None:
            return sp
        if lang not in self.models:
            return None

        start_load = time.time()
        self.log(f"Loading {self.models[lang]}...")
        sp = sentencepiece.SentencePieceProcessor()
        sp.load(str(self.models[lang]))
        self.sp[lang] = sp
        load_time = time.time() - start_load
        self.log(f"Loaded {self.models[lang]} (took {load_time / 60:.1f}min)")
        return sp

    def do(self, document: dict) -> Optional[dict]:
        text = document[self.field]
        if self.normalize:
            text = text_normalizer.normalize(text)
        sp = self.get_sp(document.get("language"))
        if sp is None:
            return document
        tokenized = sp.encode_as_pieces(text)
        document[self.output_field] = " ".join(tokenized)
        return document


class DocLM(jsonql.Transformer):
    def __init__(
        self,
        models: Union[Path, Dict[str, Path]],
        field: str,
        output_field: str = "perplexity",
        newline: str = "\n",
        normalize: bool = True,
        load_method: int = 2,
    ):
        super().__init__()
        self.field = field
        self.output_field = output_field
        self.newline = newline
        self.normalize = normalize
        self._prefetch: Sequence[str] = []
        self.lm_config = kenlm.Config()
        # This is the default settings
        # POPULATE will mmap the models and populate the pages.
        # Maybe that's not the best way when the models are on a network disk.
        # TODO: try copying models file, try READ or PARALLEL_READ
        self.lm_config.load_method = load_method

        if isinstance(models, Path):
            self.models = {
                m.name.split(".")[0]: m for m in models.parent.glob(models.name)
            }
        else:
            self.models = models
            self._prefetch = list(models.keys())
        self.lm: Dict[str, kenlm.Model] = {}
        self.n_lines = 0

    def _prepare(self) -> None:
        for lang in self._prefetch:
            assert (
                self.get_lm(lang) is not None
            ), f"No model found for {lang} at {self.models.get(lang)}."

    def get_lines(self, document: dict) -> List[str]:
        lang = document.get("language")
        if not lang:
            return []
        if lang not in self.models:
            return []

        content = document.get(self.field)
        if not content:
            return []

        lines = content.split(self.newline)
        self.n_lines += len(lines)
        return lines

    def get_lm(self, lang: Optional[str]) -> Optional[kenlm.Model]:
        if lang is None:
            return None
        lm = self.lm.get(lang)
        if lm is not None:
            return lm
        model = self.models.get(lang)
        if model is None:
            return None
        start_load = time.time()
        self.log(f"Loading {self.models[lang]}...")
        lm = kenlm.Model(str(model), self.lm_config)
        self.lm[lang] = lm
        load_time = time.time() - start_load
        self.log(f"Loaded {self.models[lang]} (took {load_time / 60:.1f}min)")

        return lm

    def do(self, document: dict) -> dict:
        lines = self.get_lines(document)
        model = self.get_lm(document.get("language"))
        if not lines or not model:
            return document

        doc_log_score, doc_length = 0, 0
        for line in lines:
            if self.normalize:
                line = text_normalizer.normalize(line)
            log_score = model.score(line)
            length = len(line.split()) + 1
            doc_log_score += log_score
            doc_length += length

        document[self.output_field] = round(pp(doc_log_score, doc_length), 1)
        return document

    def summary(self):
        delay = time.time() - self.start_time
        h = delay / 3600
        s = self.n_lines / delay

        summ = super().summary()
        summ.append(f"Processed {self.n_lines:_} lines in {h:.2}h ({s:.1} lines/s).")
        return summ


class SentencesLM(DocLM):
    """Returns the score of each individual paragraph."""

    def do(self, document: dict) -> Optional[str]:  # type: ignore
        lines = self.get_lines(document)
        model = self.get_lm(document.get("language"))
        if not lines or not model:
            return None

        sentences = []
        for line in lines:
            if self.normalize:
                line = text_normalizer.normalize(line)
            log_score = model.score(line)
            length = len(line.split()) + 1

            sentences.append(f"{pp(log_score, length)}\t{line}")

        return "\n".join(sentences)


class PerplexityBucket(jsonql.Transformer):
    def __init__(
        self, cutoff_csv: Path, percentile_head: int = 30, percentile_tail: int = 60
    ):
        super().__init__()
        self.cutoff_csv = cutoff_csv
        self.percentile_head = percentile_head
        self.percentile_tail = percentile_tail
        self.cutoffs: Dict[str, Tuple[float, float]] = {}

    def _prepare(self) -> None:
        cutoffs = pd.read_csv(self.cutoff_csv, index_col=0)
        self.cutoffs = {
            l: (cutoffs[l][self.percentile_head], cutoffs[l][self.percentile_tail])
            for l in cutoffs.columns
        }

    def get_bucket(self, doc: dict) -> str:
        perplexity = doc.get("perplexity", -1)
        lang = doc.get("language")
        if lang not in self.cutoffs or perplexity < 0:
            return "all"

        pp_head, pp_tail = self.cutoffs[lang]
        if perplexity < pp_head:
            return "head"
        if perplexity < pp_tail:
            return "middle"
        return "tail"

    def do(self, doc: dict) -> dict:
        doc["bucket"] = self.get_bucket(doc)
        return doc


class DropKeys(jsonql.Transformer):
    def __init__(self, *keys):
        super().__init__()
        self.keys = keys

    def do(self, document: dict) -> Optional[dict]:
        if not document:
            return None

        for key in self.keys:
            document.pop(key, None)
        return document


class RemoveSmall(jsonql.Transformer):
    def __init__(self, field, min_len):
        super().__init__()
        self.field = field
        self.min_len = min_len
        self.removed = 0

    def do(self, document: dict) -> Optional[dict]:
        if not document:
            return None

        content = document.get(self.field)
        if not content or len(content) < self.min_len:
            self.removed += 1
            return None
        return document

    def summary(self):
        r, n = self.removed, self.processed
        ratio = r / n if n else 0
        return [f"Removed {r} small documents out of {n} ({ratio:.1%})"]


def perplexity_to_bin(file: Path, output: Path, models, tok_field: str):
    pp_field = "perplexity"
    lm = DocLM(models, tok_field, output_field=pp_field)
    stats: List[float] = []
    max_stats = 1_000_000
    batch_size = 100_000
    i = 0
    batch = []
    with open(output, "wb") as o:
        for doc in jsonql.read_jsons(file):
            i += 1
            pp = lm(doc)[pp_field]
            if len(stats) < max_stats:
                stats.append(pp)
            batch.append(pp)
            if len(batch) >= batch_size:
                np.array(batch, dtype=np.float32).tofile(o)
                batch = []
        if len(batch) > 0:
            np.array(batch, dtype=np.float32).tofile(o)


if __name__ == "__main__":
    args = get_args()
    output = Path(args["output"])
    if output.suffix == ".bin":
        perplexity_to_bin(args["file"], output, args["models"], args["field"])
    else:
        jsonql.run_pipe(DocLM, args)
