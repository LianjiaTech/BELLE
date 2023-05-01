# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Main script to download a CC dump, remove duplicates, split by language and
filter the documents.

The pipeline parameters are described in the `Config` class.
"""

import hashlib
import json
import time
import warnings
from argparse import ArgumentParser
from collections import defaultdict
from itertools import repeat
from pathlib import Path
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

import func_argparse

# Local scripts
from cc_net import dedup, execution, jsonql, minify, perplexity, process_wet_file
from cc_net import regroup as regroup_module
from cc_net import split_by_lang
from cc_net.execution import Executor

# Constant
FILE_DIR = Path(__file__).parent
CUTOFF_CSV = FILE_DIR / "data" / "cutoff.csv"

DEFAULT_PIPELINE = [
    "dedup",
    "lid",
    "keep_lang",
    "sp",
    "lm",
    "pp_bucket",
    "drop",
    "split_by_lang",
]


class Config(NamedTuple):
    """
    Mine Common Crawl with the given settings.

    config_name
    dump: CC dump id
    output_dir: working directory
    mined_dir: name of the destination folder, full path will be {ouput_dir}/{mined_dir}/{dump_id}
    execution: chose how to parallelize the execution
    num_shards: number of shards to split the dump
    num_segments_per_shard: allow to download a small portion of CC (eg for tests)
    min_len: remove documents shorter than this (in chars)
    hashes_in_mem: number of shards hashes to use for dedup
    lang_whitelist: only treat those languages
    lang_blacklist: ignore those languages
    lang_threshold: remove docs whose top language score is lower than this
    keep_bucket: keep only those perplexity bucket chose from (head, middle, tail, all)
    lm_dir: folder containing LMs
    lm_languages: only use LMs for the following languages
    cutoff: cutoff file to use for split in head/middle/tail
    mine_num_processes: number of processes to use for mining
    target_size: size of finals files produce during `regroup` stage
    cleanup_after_regroup: delete intermediary files after regroup
    task_parallelism: max number of task to run in parallel
    pipeline: restricts the mining pipeline to the given steps. Order is important !
    experiments: (HACK) enable specific experiments in the code
    """

    config_name: str = "base"
    dump: str = "2017-51"
    output_dir: Path = Path("data")
    mined_dir: str = "mined"
    execution: str = "auto"
    num_shards: int = 1600
    num_segments_per_shard: int = -1
    metadata: Optional[str] = None
    min_len: int = 300
    hash_in_mem: int = 50
    lang_whitelist: Sequence[str] = []
    lang_blacklist: Sequence[str] = []
    lang_threshold: float = 0.5
    keep_bucket: Sequence[str] = []
    lm_dir: Path = Path("data/lm_sp")
    cutoff: Path = CUTOFF_CSV
    lm_languages: Optional[Sequence[str]] = None
    mine_num_processes: int = 16
    target_size: str = "4G"
    cleanup_after_regroup: bool = True
    task_parallelism: int = -1
    pipeline: Sequence[str] = DEFAULT_PIPELINE
    experiments: Sequence[str] = []
    cache_dir: Optional[Path] = None

    def get_executor(
        self, name: str, timeout_hour: int = 1, mem_gb: int = 1, cpus: int = 1
    ) -> Executor:
        name = "_".join((name, self.config_name, *self.experiments))
        return execution.get_executor(
            name,
            self.output_dir / "logs",
            self.execution,
            timeout_hour=timeout_hour,
            mem_gb=mem_gb,
            cpus=cpus,
            task_parallelism=self.task_parallelism,
        )

    def get_cc_shard(self, shard: int) -> process_wet_file.CCShardReader:
        dump_cache: Optional[Path] = None
        if self.cache_dir:
            self.cache_dir.mkdir(exist_ok=True)
            dump_cache = self.cache_dir / self.dump
            dump_cache.mkdir(exist_ok=True)

        return process_wet_file.CCShardReader(
            self.dump,
            shard=shard,
            num_shards=self.num_shards,
            num_segments_per_shard=self.num_segments_per_shard,
            min_len=self.min_len,
            cache_dir=dump_cache,
        )

    @classmethod
    def from_json(cls, json_file: Path) -> "Config":
        raw_lines = json_file.read_text().splitlines()
        raw_lines = [l for l in raw_lines if not l.strip().startswith("//")]
        json_config = json.loads("".join(raw_lines))
        path_keys = ["cache_dir", "lm_dir", "output_dir"]
        for key in path_keys:
            if key in json_config:
                json_config[key] = Path(json_config[key])
        return Config(**json_config)

    @property
    def will_split(self) -> bool:
        return "split_by_lang" in self.pipeline or "split_by_segment" in self.pipeline

    def get_lm_languages(self) -> Sequence[str]:
        if self.lm_languages is not None:
            return self.lm_languages

        if self.lang_whitelist:
            return self.lang_whitelist

        languages = [m.name.split(".")[0] for m in self.lm_dir.glob("*.arpa.bin")]
        if self.lang_blacklist:
            languages = [l for l in languages if l not in self.lang_blacklist]
        return languages

    def get_mined_dir(self, regroup: bool = False) -> Path:
        if self.will_split and not regroup:
            return self.output_dir / f"{self.mined_dir}_split" / self.dump
        return self.output_dir / self.mined_dir / self.dump


BASE_CONFIG = Config()

BYLANG_CONFIG = Config(
    config_name="by_lang",
    mined_dir="mined_by_lang",
    pipeline=list(BASE_CONFIG.pipeline[:-1]) + ["split_by_lang"],
)

REPRODUCE_CONFIG = Config(
    config_name="reproduce",
    dump="2019-09",
    mined_dir="reproduce",
    pipeline=["fetch_metadata", "keep_lang", "keep_bucket", "split_by_lang"],
    metadata="https://dl.fbaipublicfiles.com/cc_net/1.0.0",
    # Optional filtering:
    # It won't change much the execution speed, but decreases the disk requirement.
    # Restrict languages
    lang_whitelist=["fr"],
    # Restrict perplexity buckets
    # Top languages have been split in perplexity buckets according
    # to a Wikipedia trained LM.
    # The buckets from low perplexity (good) to high (bad) are:
    # ["head", "middle", "tail"]
    # Languages without a LM have only one bucket "all".
    # It won't change much the execution speed, but decreases the disk requirement.
    keep_bucket=["head", "all"],
    mine_num_processes=1,
)

TEST_CONFIG = BASE_CONFIG._replace(
    config_name="test",
    dump="2019-09",
    output_dir=Path("test_data"),
    execution="local",
    num_shards=4,
    num_segments_per_shard=1,
    hash_in_mem=2,
    mine_num_processes=2,
    lang_whitelist=["de", "it", "fr"],
    target_size="32M",
    cleanup_after_regroup=False,
    cache_dir=Path("test_data/wet_cache"),
)

PREDEF_CONFIGS = {
    "base": BASE_CONFIG,
    "by_lang": BYLANG_CONFIG,
    "test": TEST_CONFIG,
    "test_slurm": TEST_CONFIG._replace(execution="slurm,partition=dev"),
    "debug": TEST_CONFIG._replace(config_name="debug", mine_num_processes=0),
    "reproduce": REPRODUCE_CONFIG,
    "augment": BASE_CONFIG._replace(
        config_name="augment", dump="2019-13", lang_blacklist=["en"]
    ),
}


def tmp(output: Path) -> Path:
    return output.parent / (output.stem + ".tmp" + output.suffix)


def finalize(tmp_output: Path, output: Path) -> None:
    if not tmp_output.exists():
        warnings.warn(f"Targeted tmp output {tmp_output} doesn't exists.")
        return

    tmp_index = tmp_output.parent / (tmp_output.name + ".index")
    tmp_output.rename(output)

    if tmp_index.exists():
        tmp_index.rename(output.parent / (output.name + ".index"))


def _transpose(iterable: Sequence[Tuple[Any, ...]], n=-1) -> Tuple[List, ...]:
    if n < 0:
        n = len(iterable[0])
    columns: tuple = tuple([] for _ in range(n))
    for row in iterable:
        assert len(row) == n, f"Found tuple of len({len(row)}, expected {n}: {row}"
        for i in range(n):
            columns[i].append(row[i])

    return columns


def hashes(conf: Config) -> List[Path]:
    """Computes hashes for each shard."""

    hashes_dir = conf.output_dir / "hashes" / conf.dump
    outputs = [hashes_dir / f"{shard:04d}.bin" for shard in range(conf.num_shards)]
    missing_outputs = [(shard, o) for shard, o in enumerate(outputs) if not o.exists()]

    if not missing_outputs:
        return outputs

    hashes_dir.mkdir(parents=True, exist_ok=True)
    # With FlatHashSet we need ~2Gb of RAM / shard, but we need to account for
    # overhead due to how the dynamic allocation works.
    ex = conf.get_executor(f"hashes_{conf.dump}", mem_gb=4, timeout_hour=6, cpus=2)
    ex(_hashes_shard, repeat(conf), *_transpose(missing_outputs))

    # Wait a bit so that files appears on the disk.
    time.sleep(20)
    assert all(o.exists() for o in outputs)
    return outputs


def _hashes_shard(conf: Config, shard: int, output: Path):
    tmp_output = tmp(output)
    jsonql.run_pipes(
        dedup.HashesCollector(field="raw_content", output=tmp_output),
        inputs=conf.get_cc_shard(shard),
    )
    finalize(tmp_output, output)
    return f"Hashed {output}"


HASHES_IN_MEM = [0, 1, 2, 5, 10, 20, 50, 100, 200, 400]


def mine(conf: Config) -> List[Path]:
    """Remove dups, run LID and LMs, and split by lang and quality."""
    mined_dir = conf.get_mined_dir()
    if conf.will_split:
        # Give a directories when splitting
        outputs = [mined_dir / f"{shard:04d}" for shard in range(conf.num_shards)]
    else:
        # Files otherwise
        outputs = [
            mined_dir / f"{shard:04d}.json.gz" for shard in range(conf.num_shards)
        ]

    if "mini_again" in conf.experiments:
        mined_dir = conf.output_dir / "mini_again" / conf.dump
        outputs = [mined_dir / f"{shard:04d}" for shard in range(conf.num_shards)]

    # TODO: try to reduce this / make it a function of "hash_in_mem" / num_langs
    mem_gb = 60 + 1 * conf.hash_in_mem
    timeout_hour = 5
    if "hashes" in conf.experiments:
        # HACK: used for generating paper figures
        outputs = [
            conf.output_dir / f"hashes_exp/{conf.dump}_0000_dedup{h:03d}.json.gz"
            for h in HASHES_IN_MEM
        ]
        mem_gb = int(max(HASHES_IN_MEM) * 1.2)
        timeout_hour = 8

    missing_outputs = [(shard, o) for shard, o in enumerate(outputs) if not o.exists()]

    if "mini_again" in conf.experiments:
        missing_outputs = [
            (shard, o)
            for shard, o in enumerate(outputs)
            if shard in [5, 139] and not o.exists()
        ]

    if not missing_outputs:
        return outputs

    mined_dir.mkdir(parents=True, exist_ok=True)
    ex = conf.get_executor(
        f"mine_{conf.dump}",
        mem_gb=mem_gb,
        timeout_hour=timeout_hour,
        cpus=conf.mine_num_processes + 1,
    )

    # Compute hashes firsts.
    if "dedup" in conf.pipeline:
        hashes_groups = list(jsonql.grouper(hashes(conf), conf.hash_in_mem))
        hashes_files: Iterable[List[Path]] = [
            hashes_groups[shard // conf.hash_in_mem] for shard, o in missing_outputs
        ]
    else:
        hashes_files = repeat([])

    ex(_mine_shard, repeat(conf), hashes_files, *_transpose(missing_outputs))

    assert all(o.exists() for o in outputs)
    return outputs


def _get_segment(tmp_output: Path, doc: dict) -> str:
    segment: str = doc["cc_segment"].split("/")[-1]
    return str(tmp_output / segment.replace(".warc.wet.gz", ".json.gz"))


def _mine_shard(conf: Config, hashes: List[Path], shard: int, output: Path) -> str:
    assert conf.pipeline
    tmp_output = tmp(output)
    if "hashes" in conf.experiments:
        # HACK: used for generating paper figures
        hashes_in_mem = shard
        hashes = hashes[: HASHES_IN_MEM[hashes_in_mem]]
        shard = 0
    cc_shard = conf.get_cc_shard(shard)

    steps: Dict[str, Optional[jsonql.Transformer]] = {}
    lang_id = Path("bin") / "lid.bin"
    steps["lid_before_dedup"] = split_by_lang.Classifier(
        model=lang_id, field="raw_content", out_field="lid_before_dedup", top=5
    )
    steps["dedup"] = dedup.DuplicatesRemover(field="raw_content", hashes_files=hashes)

    steps["lid"] = split_by_lang.Classifier(
        model=lang_id,
        field="raw_content",
        out_field="language",
        top=1,
        threshold=conf.lang_threshold,
    )
    steps["lid_after_dedup"] = split_by_lang.Classifier(
        model=lang_id, field="raw_content", out_field="lid_after_dedup", top=5
    )

    if conf.lang_blacklist:
        steps["keep_lang"] = jsonql.where(
            [lambda doc: doc.get("language") not in set(conf.lang_blacklist)]
        )
    elif conf.lang_whitelist:
        steps["keep_lang"] = jsonql.where(
            [lambda doc: doc.get("language") in set(conf.lang_whitelist)]
        )
    else:
        steps["keep_lang"] = None

    tok_field = "tokenized"
    steps["sp"] = perplexity.MultiSentencePiece(
        {l: conf.lm_dir / f"{l}.sp.model" for l in conf.get_lm_languages()},
        field="raw_content",
        output_field=tok_field,
        normalize=True,
    )
    steps["lm"] = perplexity.DocLM(
        {l: conf.lm_dir / f"{l}.arpa.bin" for l in conf.get_lm_languages()},
        field=tok_field,
        output_field="perplexity",
        normalize=False,  # Normalization is done before SentencePiece
        # load_method=kenlm.LoadMethod.PARALLEL_READ,
    )
    steps["pp_bucket"] = perplexity.PerplexityBucket(CUTOFF_CSV)
    steps["drop"] = perplexity.DropKeys(tok_field)

    steps["keep_bucket"] = None
    if conf.keep_bucket:
        steps["keep_bucket"] = jsonql.where(
            [lambda doc: doc.get("bucket", "all") in conf.keep_bucket]
        )

    if "fetch_metadata" in conf.pipeline:
        # TODO: better default
        assert conf.metadata is not None
        steps["fetch_metadata"] = minify.MetadataFetcher(
            f"{conf.metadata}/{conf.dump}/"
        )

    steps["minify"] = minify.Minifier()

    pattern = str(tmp_output / "{language}_{bucket}.json.gz")
    steps["split_by_lang"] = jsonql.split(pattern=str(pattern), mkdir=True)

    steps["split_by_segment"] = jsonql.split(
        split_fn=lambda doc: _get_segment(tmp_output, doc), mkdir=True
    )

    pipeline = filter(None, (steps[s] for s in conf.pipeline))

    jsonql.run_pipes(
        *pipeline,
        inputs=cc_shard,
        processes=conf.mine_num_processes,
        chunksize=100,
        # The splitter takes care of writing to files.
        output=tmp_output if not conf.will_split else None,
    )
    finalize(tmp_output, output)
    return f"Mined {output}"


def regroup(conf: Config, all_dirs: List[Path]) -> Path:
    """Reshards each language/quality after 'mine'."""
    regroup_dir = conf.get_mined_dir(regroup=True)
    assert all_dirs
    all_files = [f for d in all_dirs for f in d.glob("*.json.gz")]
    if not all_files:
        print(f"No .json.gz file found in {all_dirs[0]}")

    splits: Dict[str, List[Path]] = defaultdict(list)
    for f in all_files:
        split = f.name.split(".")[0]
        splits[split].append(f)

    print(f"Identified {len(all_files)} files to regroup from {len(splits)} splits.")
    inputs: List[List[Path]] = []
    outputs: List[Path] = []
    target_size = jsonql.parse_size(conf.target_size)
    for split, files in splits.items():
        cuts = list(regroup_module.determine_groups(files, target_size=target_size))
        if not cuts:
            continue

        pattern = f"{split}_????.json.gz"
        existing_outputs = sorted(regroup_dir.glob(pattern))

        if not conf.cleanup_after_regroup:
            # We still have all the inputs so it is safe to overwrite existing outputs.
            assert len(existing_outputs) <= len(cuts)
            existing_outputs = []

        if len(existing_outputs) > 0 and len(cuts) == 1:
            # append to existing file if size allows it.
            new_size = (
                sum(f.stat().st_size for f in cuts[0])
                + existing_outputs[-1].stat().st_size
            )
            if new_size < target_size:
                print(f"Will append {cuts[0]} to {existing_outputs[-1]}")
                cuts[0].insert(0, existing_outputs.pop(-1))

        n_existing = len(existing_outputs)
        for i, cut in enumerate(cuts):
            # avoid overwriting existing files.
            j = i + n_existing
            output = regroup_dir / f"{split}_{j:04}.json.gz"
            inputs.append(cut)
            outputs.append(output)
        print(
            str(regroup_dir / pattern),
            "->",
            len(cuts),
            f"shards ({n_existing} already there).",
        )

    ex = conf.get_executor(f"regroup_{conf.dump}", mem_gb=1, timeout_hour=12, cpus=2)
    ex(_regroup, repeat(conf), inputs, outputs)

    return regroup_dir


def _regroup(conf: Config, inputs: List[Path], output: Path) -> str:
    output.parent.mkdir(parents=True, exist_ok=True)
    regroup_module.fast_reshard(
        inputs, output, tmp=tmp(output), rm_original=conf.cleanup_after_regroup
    )
    return f"Regrouped {output}"


def move_segments(conf: Config, all_dirs: Sequence[Path]) -> Path:
    """Reshards each language/quality after 'mine'."""
    # check that mining is over.
    regroup_dir = conf.get_mined_dir(regroup=True)
    assert all_dirs, "Received no dirs to move"
    assert all(
        d.is_dir() for d in all_dirs
    ), f"move_segments was expecting dirs received files: {all_dirs[:10]}..."

    regroup_dir.parent.mkdir(exist_ok=True)
    regroup_dir.mkdir(exist_ok=True)
    ex = conf.get_executor(f"moveseg_{conf.dump}", mem_gb=1, timeout_hour=1, cpus=2)

    def _move_segments(subdir: Path, regroup_dir: Path) -> str:
        n = 0
        for f in subdir.iterdir():
            if not f.is_file() or f.is_symlink():
                continue
            n += f.name.endswith(".json.gz")
            new_name = regroup_dir / f.name
            target = new_name.resolve()
            assert f.resolve() != target
            # this make the job idempotent.
            f.rename(new_name)
            f.symlink_to(target)

        if n == 0:
            return ""

        return f"Moved {n} .json.gz files from {subdir} to {regroup_dir}"

    ex(_move_segments, all_dirs, repeat(regroup_dir))
    print(f"Results are in {regroup_dir}")
    return regroup_dir


def _validate_test(conf: Config, output_dir: Path, generate: bool = False):
    stats: Dict[str, dict] = {}
    for file in sorted(output_dir.glob("*.json.gz")):
        fname = "/".join((file.parent.name, file.name))
        # The order of documents is not guaranteed inside a shard,
        lines = sorted(jsonql.open_read(file))
        content = "\n".join(lines)
        size = len(content)
        checksum = hashlib.sha1(bytes(content, encoding="utf-8")).hexdigest()
        # first_document = json.loads(lines[0])
        stats[fname] = {"size": size, "checksum": checksum}

    def dump(x):
        return json.dumps(x, indent=2, ensure_ascii=False)

    print("*** Stats ***")
    stats_raw = dump(stats)
    stats_file = FILE_DIR / "data" / "test_stats.json"
    if generate:
        print("Saving stats to", stats_file)
        stats_file.write_text(stats_raw)
        return

    expected_stats: Dict[str, dict] = {}
    if stats_file.exists():
        expected_stats = json.loads(stats_file.read_text())

    if expected_stats == stats:
        print("Everything looks good !")
        return

    stats_file.with_suffix(".actual.json").write_text(stats_raw)
    print("*** Expected Stats ***")
    print(dump(expected_stats))

    print("*** Diff ***")
    for fname in sorted(expected_stats.keys()):
        print(fname)
        assert fname in expected_stats, "missing file " + fname
        if expected_stats[fname]["size"] != stats[fname]["size"]:
            print(
                "  - Expected size",
                expected_stats[fname]["size"],
                ", size",
                stats[fname]["size"],
            )
        if expected_stats[fname]["checksum"] != stats[fname]["checksum"]:
            print(
                "  - Expected checksum",
                expected_stats[fname]["checksum"],
                ", checksum",
                stats[fname]["checksum"],
            )


def get_main_parser() -> ArgumentParser:
    # Generates the 'main' parser by patching a 'Config' parser
    p = func_argparse.func_argparser(Config)

    # Override defaults value to None, so we know what was set by the user.
    # Note that it will keep the original default values in the help message.
    p.set_defaults(**{f: None for f in Config._fields})
    p.add_argument("--config", type=str, default="base")
    p.set_defaults(__command=main)
    return p


def main(config: str = "base", **config_as_dict: Any) -> None:
    # Use the given 'config' as default value.
    config_base = config
    if config_base in PREDEF_CONFIGS:
        conf = PREDEF_CONFIGS[config_base]
    elif Path(config_base).exists():
        conf = Config.from_json(Path(config_base))
    else:
        raise ValueError(
            f"Invalid value {config_base} for --config. "
            f"Choose from ({', '.join(PREDEF_CONFIGS)}) or give an existing .json file."
        )
    conf = conf._replace(**{k: v for (k, v) in config_as_dict.items() if v is not None})

    print(f"Will run cc_net.mine.main with the following config:", conf)

    all_files = mine(conf)
    if conf.will_split:
        assert all_files
        assert all(d.is_dir() for d in all_files)
        all_dirs = all_files
        if "split_by_lang" in conf.pipeline:
            # Only try regrouping if we split the shards.
            regroup(conf, all_dirs)
        elif "split_by_segment" in conf.pipeline:
            # If we split by segment then regrouping is trivial, since segments appear in only one shard.
            move_segments(conf, all_dirs)

    if conf.config_name == "test":
        _validate_test(conf, conf.get_mined_dir(regroup=True))


if __name__ == "__main__":
    func_argparse.parse_and_call(get_main_parser())
