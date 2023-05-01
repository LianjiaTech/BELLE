# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Tools to search sentences in CC similar to sentences in another corpus.
"""

import functools
import logging
import math
import subprocess
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

import func_argparse
import submitit
from kenlm import Model as KenlmModel  # type: ignore
from sentence_splitter import SentenceSplitter  # type: ignore
from sentencepiece import SentencePieceProcessor  # type: ignore

from cc_net import dedup, jsonql, perplexity, text_normalizer

KENLM = Path("./bin/lmplz")
KENLM_BUILD = Path("./bin/build_binary")
VOCAB_SIZE = 2 ** 16 - 10
PROCESSES = 16


def normalize(corpus: Path, output_dir: Path) -> Path:
    normalized = output_dir / (corpus.stem + ".normalized")
    if normalized.exists():
        return normalized

    print("Will normalize", corpus, "to", normalized)
    jsonql.run_pipes(
        jsonql.Mapper(text_normalizer.normalize),
        file=corpus,
        output=normalized,
        processes=PROCESSES,
    )
    return normalized


# TODO use classic files directory.
def sp_model(lang: str) -> Path:
    return Path(f"/checkpoint/guw/cc_clean/lm_sp/{lang}.sp.model")


def _dataset(dataset: Optional[Path], lang: str) -> Path:
    return (
        dataset
        or Path("/datasets01_101/common_crawl/020919") / f"{lang}_head_*.json.gz"
    )


class SentencePiece(jsonql.Transformer):
    def __init__(self, model: Path):
        super().__init__()
        self.model = model
        self.sp: SentencePieceProcessor = None  # type: ignore

    def _prepare(self):
        self.sp = SentencePieceProcessor()
        self.sp.load(str(self.model))

    def do(self, line: str) -> str:
        return " ".join(self.sp.encode_as_pieces(line))


class ExtractSentences(jsonql.Transformer):
    def __init__(
        self,
        sp_model: Path,
        lm_model: Path,
        field: str = "raw_content",
        threshold: float = float("+inf"),
    ):
        super().__init__()
        self.sp_model = sp_model
        self.lm_model = lm_model
        self.field = field
        self.threshold = threshold
        self.sp: SentencePieceProcessor = None
        self.lm: KenlmModel = None
        self.splitter: SentenceSplitter = None
        self.hashes: Set[int] = set()

    def _prepare(self):
        self.sp = SentencePieceProcessor()
        self.sp.load(str(self.sp_model))
        self.splitter = SentenceSplitter("en")
        self.lm = KenlmModel(str(self.lm_model))

    def do(self, document: dict) -> Optional[str]:
        content: Optional[str] = document.get(self.field)
        if not content:
            return None
        all_sentences = [
            s for l in content.split("\n") if l for s in self.splitter.split(text=l)
        ]
        unique_sentences = []
        for s in all_sentences:
            if not s:
                continue
            h = dedup.str_hash(s)
            if h in self.hashes:
                continue
            self.hashes.add(h)
            unique_sentences.append(s)

        scores = []
        for sentence in unique_sentences:
            normalized = text_normalizer.normalize(sentence)
            pieces = self.sp.encode_as_pieces(normalized)
            log_score = self.lm.score(" ".join(pieces))
            pp = -1
            if len(pieces):
                pp = perplexity.pp(log_score, len(pieces))
            scores.append(pp)

        res = filter(
            lambda pp_s: self.threshold > pp_s[0] > 0, zip(scores, unique_sentences)
        )
        return "\n".join(f"{pp}\t{s}" for (pp, s) in res) or None


def tokenize(corpus: Path, output_dir: Path, lang: str) -> Path:
    tokenized = output_dir / (corpus.stem + ".tokenized")
    if tokenized.exists():
        return tokenized

    print("Will SentencePiece", corpus, "to", tokenized)
    jsonql.run_pipes(
        SentencePiece(sp_model(lang)),
        file=normalize(corpus, output_dir),
        output=tokenized,
        processes=PROCESSES,
    )

    return tokenized


def train_lm(
    corpus: Path,
    output_dir: Path,
    lang: str = "en",
    vocab_size: int = VOCAB_SIZE,
    ngrams: int = 5,
):
    lm_text_file = output_dir / (corpus.stem + ".arpa")
    lm_bin_file = output_dir / (corpus.stem + ".arpa.bin")
    if lm_bin_file.exists():
        return lm_bin_file

    assert KENLM.exists(), f"{KENLM} binary to train kenlm model not found."

    normalized = normalize(corpus, output_dir)
    tokenized = tokenize(normalized, output_dir, lang)

    print("Will train LM", lm_text_file, "on", tokenized)
    kenlm_cmd = [
        str(KENLM),
        f"--order={ngrams}",
        "--memory=8G",
        f"--temp_prefix={jsonql._tmp_dir()}",
        f"--text={tokenized}",
        f"--arpa={lm_text_file}",
        f"--vocab_estimate={vocab_size}",
        "--discount_fallback",
    ]
    subprocess.run(kenlm_cmd, check=True)
    print("Will create binary model", lm_bin_file, "from", lm_text_file)
    subprocess.run([str(KENLM_BUILD), str(lm_text_file), str(lm_bin_file)], check=True)
    return lm_bin_file


def uniform_sampling_wrt_perplexity(
    paragraphes: Iterable[str],
    rounding: float = 100.0,
    cut: float = 1000.0,
    samples: int = 20,
) -> Iterable[str]:
    max_samples = math.floor(cut / rounding * samples)
    n = 0
    buckets = Counter([0.0])
    logging.info(f"Will sample {max_samples} sentences.")
    for lines in paragraphes:
        for line in lines.split("\n"):
            if not line:
                continue
            pp = float(line[: line.find("\t")])
            pp = math.floor(pp / rounding) * rounding
            if pp > cut:
                continue
            if buckets[pp] > samples:
                continue
            yield line
            buckets[pp] += 1
            if buckets[pp] > samples:
                logging.info(f"Bucket {pp} is full ({samples} samples, {n} total)")
            n += 1
            if n > max_samples:
                return


def sample(
    corpus: Path,
    output_dir: Path,
    dataset: Path = None,
    n: int = 10_000,
    lang: str = "en",
) -> Path:
    sample_file = output_dir / (corpus.stem + ".pp_sample.tsv")
    if sample_file.exists():
        return sample_file
    dataset = _dataset(dataset, lang)
    extractor = ExtractSentences(
        sp_model(lang), train_lm(corpus, output_dir), field="raw_content"
    )
    sampling = functools.partial(
        uniform_sampling_wrt_perplexity, rounding=100.0, cut=1000.0, samples=n // 10
    )

    print(f"Will sample data from {dataset} to {sample_file}")
    try:
        jsonql.run_pipes(
            extractor, sampling, file=dataset, output=sample_file, processes=PROCESSES
        )
    except Exception:
        sample_file.unlink()
        raise

    subprocess.run(["sort", "-n", "-o", sample_file, sample_file], check=True)
    subprocess.run(["head", sample_file], check=True)
    return sample_file


def mine(
    corpus: Path,
    output_dir: Path,
    threshold: float,
    dataset: Path = None,
    lang: str = "en",
) -> List[Path]:
    """Search sentences in CC similar to the one in the given corpus.

    Args:
        - corpus: corpus to train the LM one. Assumes one sentence per line.
        - output_dir: where to store the results
        - threshold: maximum perplexity to have
        - dataset: glob pattern matching CC shards.
        - lang: search in the files of this language
    """
    dataset = _dataset(dataset, lang)
    files = list(dataset.parent.glob(dataset.name))
    outputs = [output_dir / (f.stem + ".tsv") for f in files]
    if all(o.exists() for o in outputs):
        return outputs

    n = len(outputs)
    sp = [sp_model(lang)] * n
    lm = [train_lm(corpus, output_dir)] * n
    thresholds = [threshold] * n

    ex = submitit.AutoExecutor(output_dir / "mining_logs")
    ex.update_parameters(
        name="mine",
        cpus_per_task=PROCESSES,
        timeout_min=60 * 24 // PROCESSES,
        mem_gb=10,
    )
    jobs = ex.map_array(_mine, files, outputs, sp, lm, thresholds)
    print("Submited job array:", jobs[0])

    for j in submitit.helpers.as_completed(jobs):
        (i, o) = j.result()
        print("Mined sentences from", i, "to", o)

    return outputs


def _mine(
    file: Path, output: Path, sp: Path, lm: Path, threshold: float
) -> Tuple[Path, Path]:
    extractor = ExtractSentences(sp, lm, field="raw_content", threshold=threshold)
    jsonql.run_pipes(extractor, file=file, output=output, processes=PROCESSES)
    return (file, output)


if __name__ == "__main__":
    func_argparse.main(sample, mine)
