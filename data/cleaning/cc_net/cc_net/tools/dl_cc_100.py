# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import contextlib
import functools
import gzip
import logging
import multiprocessing
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterator, List, NamedTuple, Optional, Tuple

import cc_net
from cc_net import jsonql
from cc_net.process_wet_file import CCSegmentsReader

# Set this to a directory to use as cache for intermediary files.
# This helps for debugging.
WET_CACHE = None
# WET_CACHE = Path("wet_cache")

S3_BUCKET = "https://dl.fbaipublicfiles.com/cc100"
VERSION = "1.0.0"

CC_100_SNAPSHOTS = [
    "2018-05",
    "2018-09",
    "2018-13",
    "2018-17",
    "2018-22",
    "2018-26",
    "2018-30",
    "2018-34",
    "2018-39",
    "2018-43",
    "2018-47",
    "2018-51",
]

BIG_LANGUAGES = {
    "es_XX",
    "fr_XX",
    "de_DE",
    "ja_XX",
    "ru_RU",
    "zh_CN",
    "en_XX",
    "it_IT",
    "ar_AR",
    "nl_XX",
    "pl_PL",
    "pt_XX",
    "tr_TR",
    "zh_TW",
}


class Paragraph(NamedTuple):
    lang: str
    text: str
    lm_score: float


def _dl_shard(snapshot: str, shard: int) -> Iterator[Paragraph]:
    """
    Download metadata from a shards.

    Sample metadata:

    {
        "cc_segment": "crawl-data/CC-MAIN-2018-51/segments/1544376823009.19/wet/CC-MAIN-20181209185547-20181209211547-00000.warc.wet.gz",
        "digest": "sha1:222LWNHN5FM26XGS7WJSMI6IISTVWBKJ",
        "url": "http://personals.gearplay.com/ads/DRJONES.htm",
        "line_ids": [10],
        "languages": ["en_XX"],
        "lm_scores": [-2.658],
    }
    """
    snapshot = snapshot.replace("-", "_")
    name = f"snap_{snapshot}_batch_{shard}.json.gz"
    url = "/".join([S3_BUCKET, VERSION, name])
    shard_metadata: Dict[str, Dict[str, dict]] = defaultdict(dict)
    try:
        cache_file: Optional[Path] = None
        if WET_CACHE is not None:
            cache_file = WET_CACHE / name
        metadata_file = jsonql.open_remote_file(url, cache_file)
    except:
        logging.warning(f"Couldn't open {url}")
        return

    for meta in jsonql.read_jsons(metadata_file):
        shard_metadata[meta["cc_segment"]][meta["digest"]] = meta

    found_pars, missed_pars = 0, 0
    for seg, segment_metadata in shard_metadata.items():
        for doc in CCSegmentsReader([seg], cache_dir=WET_CACHE):
            if doc["digest"] not in segment_metadata:
                continue

            meta = segment_metadata[doc["digest"]]
            full_pars = [doc["title"]] + doc["raw_content"].split("\n")

            assert len(meta["line_ids"]) == len(meta["languages"])
            assert len(meta["line_ids"]) == len(meta["lm_scores"])
            for i, lang, score in zip(
                meta["line_ids"], meta["languages"], meta["lm_scores"]
            ):
                if snapshot != "2018-51" and lang in BIG_LANGUAGES:
                    # Big languages only come from "2018-51" snapshot
                    continue
                if i >= len(full_pars):
                    # This is because CC100 was created by saving only urls.
                    # Some urls appears in different snapshot with slightly different
                    # versions, but we don't know which one is correct.
                    # Here we read both versions, but some index may end up
                    # being incorrect.
                    # This impact ~3% documents.
                    missed_pars += 1
                    continue

                yield Paragraph(lang, full_pars[i], score)
                found_pars += 1
        if missed_pars > 0:
            logging.warning(
                f"Missed {missed_pars} ({missed_pars / found_pars:%}) paragraphes."
            )


def _split_by_par(
    paragraphes: Iterator[Paragraph], snapshot: str, shard: int, outdir: Path
) -> int:
    outdir.mkdir(exist_ok=True)
    outfiles = {}
    num_pars = 0
    try:
        for par in paragraphes:
            # MODIFY ME: filter paragraph if needed (languages, score, ...)
            if par.lang not in outfiles:
                (outdir / par.lang).mkdir(exist_ok=True)
                outfile = outdir / par.lang / f"snap_{snapshot}_batch_{shard}.gz"
                outfiles[par.lang] = gzip.open(outfile, "wt")

            print(par.text, file=outfiles[par.lang])
            num_pars += 1
    finally:
        for o in outfiles.values():
            o.close()

    logging.info(f"Extracted {num_pars:_d} paragraphs from shard {snapshot}_{shard}")
    return num_pars


def dl_shard(snapshot: str, shard: int, outdir: Path) -> int:
    return _split_by_par(_dl_shard(snapshot, shard), snapshot, shard, outdir)


@contextlib.contextmanager
def unordered_map(processes: int):
    if processes == 0:
        yield map
        return

    with multiprocessing.Pool(processes) as pool:
        yield pool.imap_unordered


def dl_snapshot(snapshot: str, outdir: Path, processes: int = 1) -> None:
    _dl_shard = functools.partial(dl_shard, snapshot, outdir=outdir)

    with unordered_map(processes) as umap:
        num_pars = sum(umap(_dl_shard, range(500)))

    logging.info(f"Extracted {num_pars:_d} paragraphs from snapshot {snapshot}.")


def dl(
    snapshot: str = None, outdir: Path = Path("data_cc100"), processes: int = 1
) -> None:
    """
    Download CC100 corpus.
    Will create one text file per language and CC snapshot.

    - snapshot: restrict to one snapshot. Useful for parallelization.
    - outdir: output directory
    - processes: number of processes to use
    """
    if snapshot is None:
        snapshots = CC_100_SNAPSHOTS
    else:
        snapshots = snapshot.split(",")

    invalids = [s for s in snapshots if s not in CC_100_SNAPSHOTS]
    assert not invalids, f"Invalid snapshots {invalids}, chose from {CC_100_SNAPSHOTS}"

    for snapshot in snapshots:
        dl_snapshot(snapshot, outdir, processes)


if __name__ == "__main__":
    import func_argparse

    func_argparse.single_main(dl)
