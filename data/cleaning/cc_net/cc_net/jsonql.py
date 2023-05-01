# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Manipulate files containing one json per line.
"""
import argparse
import collections
import contextlib
import functools
import glob
import gzip
import importlib
import inspect
import io
import itertools
import json
import logging
import multiprocessing
import os
import re
import sys
import tempfile
import time
import typing as tp
import warnings
import zlib
from pathlib import Path
from typing import (
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    TextIO,
    Tuple,
    Union,
)

import numpy as np
import psutil  # type: ignore
import requests
from typing_extensions import Protocol

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(process)d:%(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M",
)

NEWLINE = " N3WL1N3 "

FilterFn = Callable[[dict], bool]
FileDescriptor = Union[Path, List[Path], str]
WritableFileLike = Union[FileDescriptor, TextIO, "SimpleIO", None]
ReadableFileLike = Union[Iterable[str], FileDescriptor, None]


def io_parser():
    """Parser shared by all commands to get input/output files."""
    parser = argparse.ArgumentParser(add_help=False)
    file_help = """File to read from. Can be specified several times for several files.
    Be careful that bash will expand glob patterns **before** sending the args
    to python. To use globs put it inside single quotes:
        jsonql where --file 'data/perplexity/*.json' '{length} > 100' | head -1
        jsonql --file 'data/perplexity/*.json' where '{length} > 100' | head -1
        [Invalid] jsonql where '{length} > 100' --file data/perplexity/*.json | head -1
        [Invalid] jsonql where --file data/perplexity/*.json '{length} > 100' | head -1
    """
    parser.add_argument("-f", "--file", type=Path, action="append", help=file_help)
    parser.add_argument("-o", "--output", type=Path, default="-")
    parser.add_argument("--processes", type=int, default=1)
    return parser


def get_parser():
    parser = argparse.ArgumentParser(
        description="Read a set of json files and allow to query them"
    )
    subparsers = parser.add_subparsers()

    def add_subparser(function, arguments):
        doc = function.__doc__.split("\n")[0]
        p = subparsers.add_parser(function.__name__, help=doc, parents=[io_parser()])
        p.set_defaults(command=function)
        for k, v in arguments.items():
            p.add_argument(k, **v)

    add_subparser(
        select,
        {
            "columns": dict(nargs="+", help="Extract the value of the given fields"),
            "--skip_empty": dict(
                action="store_true", help="Skip lines without the requested fields"
            ),
            "--separator": dict(
                default="\t", help="Separator to use between the different columns"
            ),
            "--newline": dict(
                default=NEWLINE,
                help="Replace newlines found in the text by the given string",
            ),
        },
    )

    add_subparser(
        where,
        {
            "clauses": dict(nargs="+", help=""),
            "--requires": dict(
                action="append", help="Python module required by the clauses code."
            ),
        },
    )

    add_subparser(
        merge,
        {
            "columns": dict(nargs="+", help=""),
            "--separator": dict(
                default="\t", help="Separator to use between the different columns"
            ),
            "--newline": dict(
                default=NEWLINE, help="Replace the given string by actual newlines"
            ),
        },
    )

    add_subparser(
        describe,
        {
            "columns": dict(nargs="*", help=""),
            "--bins": dict(
                default="auto", help="Number of bins for computing the histograms"
            ),
            "--cumulative": dict(
                action="store_true", help="Compute cumulative histograms"
            ),
            "--weights": dict(type=str, help="Column used to weight histograms"),
        },
    )

    add_subparser(split, {"--pattern": dict(type=str)})
    add_subparser(shard, {})
    return parser


def _split_array(array, sep):
    last = 0
    for i, x in enumerate(array):
        if x != sep:
            continue
        yield array[last:i]
        last = i + 1
    if last != len(array):
        yield array[last:]


def main(raw_args):
    parser = get_parser()
    pipeline = []
    file = "-"
    output = "-"
    processes = 1

    for args_group in _split_array(raw_args, "--"):
        args = vars(parser.parse_args(args_group))
        command = args.pop("command")
        file = args.pop("file") or file
        output = args.pop("output") or output
        processes = args.pop("processes") or processes
        pipeline.append(as_pipe(command, args))

    if not pipeline:
        parser.print_help()
        return

    run_pipes(*pipeline, file=Path(file), output=Path(output), processes=processes)


class Transformer:
    """
    Wrapper around functions transforming documents.

    This allows `run_pipes` to automatically parallelize the pipeline.
    Provides:
    * Automatic logging. Logging can be changed with the `summary` method.
        Loggin frequency with _log_freq (in second) or $JSONQL_LOG_FREQ env variable.
    * Automatic parallelization without pickling. The transformers are shared
    across processes, and the object is usually not pickled.
    * Basic pickling / unpickling in case it's still needed.
    By default will only pickle the arguments passed to the constructor.
    * Delayed initialization. Internal state which is not pickable should be set
    inside the `_prepare` function.
    """

    parallelisable: bool = True
    expect_json: bool = False
    warn_when_pickling: bool = False
    ready: bool = False

    def __init_subclass__(cls, expect_json: bool = None):
        """Detects if the subclass expects json as input."""
        spec = inspect.getfullargspec(cls.do)
        if expect_json is None:
            expect_json = spec.annotations.get(spec.args[1], None) == dict

        cls.expect_json = expect_json

    def __new__(cls, *args, **kwargs):
        """Creates the transformer and save the arguments passed to the constructor."""
        t = super().__new__(cls)
        Transformer.__init__(t, args, kwargs)
        return t

    def __init__(self, state_args: tuple = None, state_kwargs: dict = None):
        """
        Init the transformer counters.

        If state_args/state_kwargs are set they will override whatever was
        originally passed to the subclass constructor.
        """
        if state_args is not None:
            self.__args = state_args
        if state_kwargs is not None:
            self.__kwargs = state_kwargs

        self.start_time = time.time()
        self.__last_log = self.start_time
        self.processed = 0
        # Log every 5 min unless specified other wise.
        self._log_freq = int(os.environ.get("JSONQL_LOG_FREQ", 5 * 60))
        self.__cls = type(self)
        self._logger = logging.getLogger(self.__cls.__name__)

    def __call__(self, x):
        assert self.ready, f"{self} is not ready."
        if x is None:
            return
        y = self.do(x)
        self.processed += 1
        if time.time() - self.__last_log > self._log_freq:
            self.log_summary()
        return y

    def do(self, x):
        raise NotImplementedError(f"'do' not implemented in {type(self)}")

    def summary(self) -> List[str]:
        return [self.speed_summary()]

    def speed_summary(self) -> str:
        delay = time.time() - self.start_time
        h = delay / 3600
        s = self.processed / delay
        return f"Processed {self.processed:_} documents in {h:.2}h ({s:5.1f} doc/s)."

    def log(self, message):
        self._logger.info(message)

    def log_summary(self) -> None:
        if not self.ready:
            self.log("Not ready.")
            return
        summ = self.summary() or []
        for line in summ:
            self.log(line)
        self.__last_log = time.time()

    def map(self, source: Iterable) -> Iterator:
        if self.ready:
            for x in source:
                yield self(x)
            # since we have been prepared by caller,
            # caller is also responsible for calling `close`.
            return
        else:
            with self:
                for x in source:
                    yield self(x)

    def __getstate__(self) -> Tuple[tuple, dict, bool]:
        return (self.__args, self.__kwargs, self.expect_json)

    def __setstate__(self, state: Tuple[tuple, dict, bool]):
        if self.warn_when_pickling:
            warnings.warn(f"Unpickling transformer: {type(self)}. This can be slow.")
        (args, kwargs, expect_json) = state
        # When unpickling `__new__` isn't called so we have to doit ourselves.
        Transformer.__init__(self, state_args=args, state_kwargs=kwargs)
        type(self).__init__(self, *args, **kwargs)
        assert self.expect_json == expect_json
        # __setstate__ is called by multiprocessing right before calling
        # the object so we need to initialize everything.
        self.__enter__()

    def _prepare(self) -> None:
        pass

    def __enter__(self) -> "Transformer":
        # In multiprocessing __enter__ is always called twice, so we are idempotent.
        # Because we call __enter__ when deserializing this transformer and
        # also when the parent transformer is deserialized.
        self.start_time = time.time()
        if self.ready:
            return self
        self._prepare()
        self.ready = True
        return self

    def __exit__(self, *args) -> None:
        self.close()
        self.log_summary()

    def close(self) -> None:
        pass


def as_pipe(transformer, kwargs):
    if isinstance(transformer, type):
        return transformer(**kwargs)
    return lambda source: transformer(source, **kwargs)


def compose(fns: List[Transformer]) -> Transformer:
    if len(fns) == 1:
        return fns[0]
    return MultiTransformer(fns)


class MultiTransformer(Transformer):
    def __init__(self, transformers: List[Transformer]):
        super().__init__()
        self.transformers = transformers

    def __repr__(self) -> str:
        pipeline = " | ".join(type(t).__name__ for t in self.transformers)
        return f"<{pipeline}>"

    def do(self, x):
        for t in self.transformers:
            x = t(x)
        return x

    def _prepare(self):
        for t in self.transformers:
            t.__enter__()
        return self

    def __exit__(self, *args):
        for t in self.transformers:
            t.__exit__(*args)

    def summary(self):
        return itertools.chain(*(t.summary() for t in self.transformers))


class Mapper(Transformer):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def do(self, x):
        return self.fn(x)


def run_pipe(
    command,
    kwargs: dict = None,
    file: ReadableFileLike = None,
    output: WritableFileLike = None,
):
    kwargs = kwargs or {}
    if isinstance(kwargs, argparse.ArgumentParser):
        kwargs = vars(kwargs.parse_args())
    file = file or Path(kwargs.pop("file", "-"))
    output = output or Path(kwargs.pop("output", "-"))

    return run_pipes(as_pipe(command, kwargs), file=file, output=output)


def run_pipes(
    *fns: Union[Transformer, Callable[[Iterable], Iterable]],
    inputs: Iterable[dict] = None,
    file: ReadableFileLike = None,
    output: WritableFileLike = None,
    processes: int = 1,
    chunksize: int = 10_000,
):
    """
    Run full document processing pipeline.

    - fns: list of functions to run over the documents. Can be:
        * `Iterable -> Iterable` function
        * jsonql.Transformer instance
        Using transformers allow the pipeline to process documents in parallel.
    - inputs: iterable to read the documents from
    - file: if inputs is not given, will read documents from this file.
    - output: writable file like.
    - processes: number of processes to use. -1 means all CPU available.
    - chunksize: chunksize for multiprocessing.Pool.imap_unordered
    """
    expect_json = len(fns) and isinstance(fns[0], Transformer) and fns[0].expect_json
    if expect_json and inputs is None:
        fns = (JsonReader(),) + fns
    transformers = []
    for t in fns:
        if not isinstance(t, Transformer):
            break
        if not t.parallelisable:
            break
        transformers.append(t)
    pipes = fns[len(transformers) :]

    log = logging.getLogger(__name__).info
    if inputs is None:
        data: Iterable = open_read(file)
    else:
        data = inputs

    if processes == -1:
        processes = os.cpu_count() or 0

    with contextlib.suppress(BrokenPipeError), contextlib.ExitStack() as stack:
        if transformers:
            log(f"preparing {transformers}")
            transform = stack.enter_context(compose(transformers))
            if processes <= 1:
                data = transform.map(data)
            else:
                p = multiprocessing.current_process()
                log(f"Will start {processes} processes from {p.name}, Pid: {p.pid}")
                pool = stack.enter_context(
                    multiprocessing.Pool(
                        processes=processes,
                        initializer=_set_global_transformer,
                        initargs=(transform,),
                    )
                )
                data = pool.imap_unordered(
                    _global_transformer, data, chunksize=chunksize
                )

        for fn in pipes:
            if isinstance(fn, Transformer):
                data = fn.map(data)
            else:
                data = fn(data)

        write_jsons(data, output)


# Allows to share transformer acroos subprocess.
# Used by `run_pipes`
_GLOBAL_TRANSFORMER: Optional[Transformer] = None


def _set_global_transformer(transformer: Transformer):
    global _GLOBAL_TRANSFORMER
    p = multiprocessing.current_process()
    logging.info(
        f"Started subprocess {p.name}:{p.pid} from {os.getppid()} for {transformer}"
    )
    assert transformer.ready, f"{transformer} isn't ready"
    _GLOBAL_TRANSFORMER = transformer


def _global_transformer(document: str) -> Optional[dict]:
    assert _GLOBAL_TRANSFORMER is not None
    return _GLOBAL_TRANSFORMER(document)


def lines(file: ReadableFileLike) -> Iterator[str]:
    return (line.strip("\n") for line in open_read(file))


def read_jsons(file: ReadableFileLike, strict=False) -> Iterator[dict]:
    reader = JsonReader(strict=strict)
    lines = open_read(file)
    for line in lines:
        if line is None:
            continue
        yield reader(line)

    reader.log_summary()


def write_jsons(source: Iterable[dict], file: WritableFileLike) -> None:
    eol = os.linesep
    with open_write(file) as o:
        for res in source:
            if res is None:
                continue
            if isinstance(res, dict):
                json.dump(res, o, ensure_ascii=False)
                o.write(eol)
                continue
            if isinstance(res, str):
                res = res.rstrip("\n")
            print(res, file=o)


class JsonReader(Transformer):
    def __init__(self, strict: bool = False):
        super().__init__()
        self.ready = True
        self.strict = strict
        self.num_errors = 0

    def do(self, line: str) -> Optional[dict]:
        if line is None:
            return None
        if isinstance(line, dict):
            return line
        line = line.rstrip("\n")
        if not line:
            return None
        try:
            return json.loads(line)
        except json.decoder.JSONDecodeError as e:
            self.log_error(e)
            if self.strict:
                raise
            return None

    def log_error(self, e: json.decoder.JSONDecodeError):
        self.num_errors += 1
        if self.num_errors > 10:
            return

        MAX_LEN = 80
        snippet, snippet_len = e.doc, len(e.doc)
        col = e.pos
        if snippet_len > MAX_LEN:
            if col < MAX_LEN:
                start = 0
            elif snippet_len - col < MAX_LEN:
                start = snippet_len - MAX_LEN
            else:
                start = col - MAX_LEN // 2
            snippet = e.doc[start : start + MAX_LEN]
            col = col - start
        logging.warning(
            "\n".join(
                [
                    f"Invalid json (length={len(e.doc)}) {e}",
                    snippet,
                    " " * (col - 1) + "^",
                ]
            )
        )

    def summary(self):
        summ = super().summary()
        if self.num_errors > 0:
            summ.append(f"Skipped {self.num_errors} invalid json.")
        return summ


def compile_column(column, newline):
    if callable(column):
        return column

    if column == "*":
        return json.dumps

    if re.match(r"[_a-z][_a-z0-9]*", column):

        def extract_col(doc):
            v = doc.get(column, "")
            if isinstance(v, str) and newline != "\n":
                v = v.rstrip("\n").replace("\n", newline)
            return v

        return extract_col

    return compile_expr(column)


def select(lines, columns, skip_empty=False, separator="\t", newline="\n"):
    """Yields the content of the requested columns."""
    column_parsers = [compile_column(c, newline) for c in columns]
    for doc in read_jsons(lines):
        values = []
        empty = True
        for parse_col in column_parsers:
            v = parse_col(doc)
            values.append(str(v) or "")
            empty = empty and v is None

        if skip_empty and empty:
            continue

        yield separator.join(values)


def compile_expr(clause: Union[str, FilterFn], requires: List[str] = None):
    if not isinstance(clause, str):
        return clause

    args_re = r"(?i:\{([_a-z][_a-z0-9]*)\})"
    args_list = list(re.findall(args_re, clause))
    if not args_list:
        # This is only a warning because you may want to have eg random sampling
        # that doesn't depend on the document.
        logging.warn(
            f"Warning: No variable found in expression: <{clause}>\n"
            "Variables should be written inside braces, eg: {language}=='en'"
        )
    python_like = re.sub(args_re, r"doc.get('\1', None)", clause)
    requires = requires or []
    modules = {r: importlib.import_module(r) for r in requires}
    return eval(f"lambda doc: {python_like}", modules)


class where(Transformer):
    """Filters the data using python code.

    Ex: `jsonql where 'len({text}) > 100'`
    """

    def __init__(
        self, clauses: Sequence[Union[str, FilterFn]], requires: List[str] = []
    ):
        super().__init__()
        self.raw_clauses = clauses
        self.requires = requires
        self.n_selected = 0
        self.clauses: List[FilterFn] = []

    def _prepare(self):
        self.clauses = [compile_expr(c, self.requires) for c in self.raw_clauses]

    def do(self, doc: dict) -> Optional[dict]:
        assert self.clauses
        if not doc or not all((c(doc) for c in self.clauses)):
            return None
        self.n_selected += 1
        return doc

    def summary(self):
        n_selected, n_docs = self.n_selected, self.processed
        selectivity = n_selected / n_docs if n_docs else 0
        return [f"Selected {n_selected} documents out of {n_docs} ({selectivity:5.1%})"]


def merge(lines, columns, separator="\t", newline=NEWLINE):
    """Reads tab separated columns and output a json using the given headers.

    Headers are of form {key}[%{type}]
    {type} can be one of {"f": float, "i": int, "b": bool, "s": string}.
    Default type is string.
    A special header "_" means interpret this column as json, and append all other
    columns to it. Must appear only once and on last position.

    Ex:
    `echo '1\thello' | jsonql merge n t` --> `{"n": "1", "t": "hello"}`
    `echo '1\thello" | jsonql merge n%i t` --> `{"n": 1, "t": "hello"}`
    `echo '1\thello\t{"f": "bar"}' | jsonql merge n%i t _` --> `{"n": 1, "t": "hello", "f": "bar"}`
    """
    handle_newlines = lambda s: s.replace(newline, "\n")
    type_mapping: Dict[str, Callable] = {
        "f": float,
        "i": int,
        "b": bool,
        "s": handle_newlines,
    }
    type_parsing = [
        type_mapping.get(f.split("%")[-1], handle_newlines) for f in columns
    ]
    columns = [f.split("%")[0] for f in columns]
    doc_index = columns.index("_") if "_" in columns else -1
    read_json = JsonReader()

    def parse(line):
        parts = line.split(separator, len(columns) - 1)
        doc: Dict[str, tp.Any] = {}
        for i, value in enumerate(parts):
            if columns[i] == "_":
                doc.update(read_json(parts[doc_index]))
            else:
                try:
                    doc[columns[i]] = type_parsing[i](value)
                except ValueError:
                    logging.error(
                        f"Error when parsing column {i} of line: {line[:100]}..."
                    )
        return doc

    for line in lines:
        yield json.dumps(parse(line))


class split(Transformer):
    """Split a files in several smaller files based on the value of a field."""

    # Not parallelisable since we are writing to files.
    parallelisable = False

    def __init__(
        self,
        pattern: Union[Path, str] = None,
        split_fn: Callable[[dict], str] = None,
        mkdir: bool = False,
    ):
        super().__init__()
        assert not (
            pattern and split_fn
        ), "split can't have both a pattern and a split_fn"
        if split_fn is not None:
            self.split_fn = split_fn
        else:
            assert pattern, "split need either a pattern or a split_fn"
            self.split_fn = self.make_split_fn(str(pattern))
        self.mkdir = mkdir
        self.o: dict = {}

    def make_split_fn(self, pattern: str) -> Callable[[dict], str]:
        candidates = list(re.findall(r"(?i:\{([_a-z][_a-z0-9]*)\})", pattern))
        return lambda doc: pattern.format(**{c: doc[c] for c in candidates})

    def do(self, doc):
        filename = self.split_fn(doc)
        if not filename:
            return
        o = self.o.get(filename, None)
        if o is None:
            if self.mkdir:
                Path(filename).parent.mkdir(parents=True, exist_ok=True)
            self.o[filename] = open_write(filename)
        print(json.dumps(doc, ensure_ascii=False), file=self.o[filename], flush=True)

    def summary(self):
        summ = super().summary()
        summ.append(f"Found {len(self.o)} splits.")
        return summ

    def close(self):
        for file in self.o.values():
            file.close()


def histogram(values, bins, weights):
    hist, bins = np.histogram(values, bins=bins)
    # n_bins = len(hist)

    if weights is not None:
        # Bins can't be auto-determined if weights is supplied.
        # So we first compute the bins without the weights then recompute
        # the histogram with the weights.
        hist, bins = np.histogram(values, bins=bins, weights=weights)
    # cumsum = np.cumsum(hist)
    # total = cumsum[-1]

    # for i in range(n_bins - 1):
    #     if cumsum[i] / total > 0.9:
    #         useful_range = np.linspace(bins[0], bins[i + 1], n_bins)
    #         new_bins = np.append(useful_range, [bins[-1]])
    #         return np.histogram(values, bins=new_bins, weights=weights)

    return hist, bins


def _parse_bins(bins):
    try:
        if isinstance(bins, str):
            if "," in bins:
                bins = [int(b) for b in bins.split(",")]
            else:
                bins = int(bins)
    except ValueError:
        pass
    return bins


ALL_DOCUMENTS = "<ALL_DOCUMENTS>"
MAX_LABEL_LEN = 100


def bar_chart(hist, bins):
    n = sum(hist)
    max_h = max(hist)
    out = []
    for i, h in enumerate(hist):
        h_size = 80 * h // max_h
        dh_size = 80 * (h - hist[i - 1]) // max_h
        if h_size == 0 or dh_size == 0:
            continue
        bar = "█" * h_size
        out.append(f"{bins[i]:8.3f} {bar:80} ({h:5d}, {h / n:5.1%}) {bins[i+1]:8.3f}")
    out.append(f"{bins[-1]:8.3f}")
    return out


def display_stats(stats, key, weights=None, bins="auto", cumulative=False):
    out = []
    documents = stats[ALL_DOCUMENTS]
    count = stats.get(key, 0)
    r = count / documents if documents else 0
    out.append(f"Field {key} saw {count} times ({r:5.1%})")

    length = stats.get(key + ".length", None)
    avg_length = length // count if length else 0
    if length is not None:
        out[-1] += f", average length is {length // count}"

    values = stats.get(key + ".val", None)
    if values:
        out[-1] += f", histogram is: (bins={bins})"
        if weights:
            if weights not in stats:
                logging.warn(f"Warning: weights column {weights} not found.")
            if weights + ".val" not in stats:
                logging.warn(
                    f"Warning: weights column {weights} is not a numeric column."
                )
            weights = stats.get(weights + ".val")
        hist, bins = histogram(values, _parse_bins(bins), weights)
        if cumulative:
            hist = np.cumsum(hist)
        out += bar_chart(hist, bins)

    cnt = stats.get(key + ".cnt", None)
    if avg_length < MAX_LABEL_LEN and cnt and max(cnt.values()) > 1:
        cnt = sorted(cnt.items(), key=lambda kv: kv[1], reverse=True)
        out[-1] += ", top 100 labels:"
        for label, n in cnt[:100]:
            if n < 5:
                continue
            out.append(f"{label:25}: {n:6} ({n / count:5.1%})")

    return out


def describe(source, columns=None, weights=None, **kwargs):
    """Compute some statistics about a dataset.

    Stats can be restricted to a subset of columns."""
    MAX_HIST_SIZE = 100_000_000
    MAX_CNT_SIZE = 1000
    stats = {ALL_DOCUMENTS: 0}
    needed = columns + [weights] if columns else None

    for doc in read_jsons(source):
        stats[ALL_DOCUMENTS] += 1
        for k, v in doc.items():
            if needed and k not in needed:
                continue
            stats[k] = get_or_set(stats, k, 0) + 1
            if isinstance(v, str):
                stats[k + ".length"] = get_or_set(stats, k + ".length", 0) + len(v)
                if len(v) > MAX_LABEL_LEN:  # Don't treat too long string as labels
                    continue
                cnt = get_or_set(stats, k + ".cnt", collections.defaultdict(int))
                if v in cnt or len(cnt) < MAX_CNT_SIZE:
                    cnt[v] += 1
            elif type(v) in (int, float):
                values = get_or_set(stats, k + ".val", [])
                if len(values) < MAX_HIST_SIZE:
                    values.append(v)
            elif type(v) is list and len(v) and type(v[0]) in (int, float):
                values = get_or_set(stats, k + ".val", [])
                if len(values) < MAX_HIST_SIZE:
                    values += v
            elif type(v) is dict:
                cnt = get_or_set(stats, k + ".cnt", collections.defaultdict(int))
                for label in v:
                    if label in cnt or len(cnt) < MAX_CNT_SIZE:
                        cnt[label] += 1

    documents = stats[ALL_DOCUMENTS]
    yield f"Stats computed on {documents} documents:"
    for k in stats:
        if columns and k not in columns:
            continue
        if "." in k or k == ALL_DOCUMENTS:
            continue
        for line in display_stats(stats, k, weights=weights, **kwargs):
            yield line


def shard(lines):
    """Shard a file in several smaller ones."""
    # The creation of the shard is handle in a generic way. Do we need this ?
    return lines


# *** Utils ***


def get_or_set(dictionary, key, default):
    if key not in dictionary:
        dictionary[key] = default
    return dictionary[key]


class SimpleIO(Protocol):
    """A subset of methods from TextIO."""

    def close(self) -> None:
        ...

    def write(self, line: str) -> int:
        ...

    def __enter__(self) -> "SimpleIO":
        ...

    def __exit__(self, exc_type, exc_value, traceback):
        ...


def open_read(filename: ReadableFileLike) -> Iterable[str]:
    """Open the given file, list of files or files matching the given glob and read lines.

    `filename` is None or "-" -> reads from stdin
    `filename` is a Path / str -> interprets filename as a glob and open files matching it
    `filename` is a list -> opens sequentially all files from the list using `open_read`
    `filename` is something else -> returns the object wrapped in a `nullcontext`
        This allows to pass already openened files or iterables.

    `open_read` will decompress gzip files, given they have ".gz" suffix.
    """
    if filename is None:
        return sys.stdin

    if isinstance(filename, list):
        assert isinstance(filename[0], Path)
        if len(filename) == 0:
            return []
        if len(filename) > 1:
            return _yield_from(filename)
        filename = tp.cast(Path, filename[0])
    if isinstance(filename, str):
        if filename.startswith("http://") or filename.startswith("https://"):
            return open_remote_file(filename)

        filename = Path(filename)
    if not isinstance(filename, Path):
        # we might have received an iterable, return it unmodified.
        return filename  # type: ignore

    # Expand glob patterns only when reading
    files = [Path(f) for f in sorted(glob.glob(str(filename)))]
    if len(files) > 1:
        return _yield_from(files)
    if len(files) == 1:
        filename = files[0]

    assert isinstance(filename, Path)

    if filename.name.endswith("]"):
        return block_reader(filename)

    logging.getLogger(__name__).info(f"Opening {filename} with mode 'rt'")
    if filename.suffix == ".gz":
        file: TextIO = gzip.open(filename, "rt")  # type: ignore
    else:
        file = open(filename, "rt")

    return _close_when_exhausted(file)


def _close_when_exhausted(file: TextIO) -> Iterable[str]:
    with file:
        yield from file


def _yield_from(files: list) -> Iterable[str]:
    for file in files:
        yield from open_read(file)


def open_write(
    filename: WritableFileLike, max_size: str = "4G"
) -> tp.ContextManager[TextIO]:
    """Open the given file, list of files or files matching the given glob.

    The return value is a ContextManager meant to be used inside a `with` block:
    ```
    with open_write("foo.txt") as o:
        ...

    Write mode:
        replaces "?" from filename by numbers ranging from 0 to 9, generatings files of size `max_size`.
        If filename ends with ".gz", creates a blocked gzip file with random access.
    """
    if filename is None:
        return contextlib.nullcontext(sys.stdout)

    if isinstance(filename, list):
        if len(filename) > 1:
            return MultiFile(filename, "w", max_size)
        else:
            filename = tp.cast(Path, filename[0])
    if isinstance(filename, str):
        filename = Path(filename)
    if not isinstance(filename, Path):
        assert hasattr(filename, "write"), f"{filename} doesn't have a .write method."
        # We return a 'TextIO' even though we only check for `.write` method,
        # this works better with eg `print`.
        return contextlib.nullcontext(tp.cast(TextIO, filename))

    mode = "wt"
    if "?" in filename.name:
        return sharded_file(filename, mode, max_size)

    logging.getLogger(__name__).info(f"Opening {filename} with mode {mode}")
    # TODO: should we use another format ?
    if filename.suffix == ".gz":
        return BlockedGzipWriter(Path(filename), mode, block_size="64M")

    return open(filename, "wt")


def parse_size(size):
    unit_map = {"B": 1, "K": 1024, "M": 1024 ** 2, "G": 1024 ** 3}
    unit = size[-1].upper()
    assert (
        unit in unit_map
    ), f"Unsupported size unit for {size}. Use one of: {unit_map.keys()}."
    return int(size[:-1]) * unit_map[unit]


class MultiFile(SimpleIO):
    def __init__(self, files: Iterable[Path], mode="w", max_size="4G"):
        self.name = str(files)
        self.mode = mode
        self.files = iter(files)
        self.max_size = parse_size(max_size)
        self.current_handle: Optional[TextIO] = None
        self.current_block_size = 0
        self._open_next_handle()  # Opening 1st handle allows to write directly.

    def write(self, content) -> int:
        # Avoid splitting newlines to a new file.
        # use current_block_size since it's faster than `tell()`
        if content != "\n" and self.current_block_size >= self.max_size:
            self._open_next_handle()
        if self.current_handle is None:
            raise Exception("No more files to write to...")

        written = self.current_handle.write(content)
        self.current_block_size += written
        return written

    def _open_next_handle(self) -> bool:
        self.close()
        file = next(self.files, None)
        if file is None:
            return False

        self.current_handle = open_write(file).__enter__()
        self.current_block_size = 0
        return True

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    @property
    def closed(self):
        return self.current_handle is None

    def close(self):
        if self.current_handle is None:
            return

        # log("Closing", self.current_handle.name, "with mode", self.current_handle.mode)
        self.current_handle.__exit__(None, None, None)
        self.current_handle = None


# not sure it helps since connections are reseted anyway.
_session = functools.lru_cache()(requests.Session)


def request_get_content(url: str, n_retry: int = 3) -> bytes:
    """Retrieve the binary content at url.

    Retry on connection errors.
    """
    t0 = time.time()
    logging.info(f"Starting download of {url}")
    for i in range(1, n_retry + 1):
        try:
            r = _session().get(url)
            r.raise_for_status()
            break
        except requests.exceptions.RequestException as e:
            # Sleep and try again on error, unless it's a 404.
            message = e.args[0] if isinstance(e.args[0], str) else ""
            if i == n_retry or "Client Error" in message:
                raise e
            warnings.warn(
                f"Swallowed error {e} while downloading {url} ({i} out of {n_retry})"
            )
            time.sleep(10 * 2 ** i)
    dl_time = time.time() - t0
    dl_speed = len(r.content) / dl_time / 1024
    logging.info(
        f"Downloaded {url} [{r.status_code}] took {dl_time:.0f}s ({dl_speed:.1f}kB/s)"
    )
    return r.content


def open_remote_file(url: str, cache: Path = None) -> Iterable[str]:
    """Download the files at the given url to memory and opens it as a file.
    Assumes that the file is small, and fetch it when this function is called.
    """
    if cache and cache.exists():
        return open_read(cache)

    # TODO: open the remote file in streaming mode.
    # The hard part is that we need to write the content on disk at the same time,
    # to implement disk caching.
    raw_bytes = request_get_content(url)
    content = io.BytesIO(raw_bytes)
    if url.endswith(".gz"):
        f: TextIO = gzip.open(content, mode="rt")  # type: ignore
    else:
        f = io.TextIOWrapper(content)

    if cache and not cache.exists():
        # The file might have been created while downloading/writing.
        tmp_cache = _tmp(cache)
        tmp_cache.write_bytes(raw_bytes)
        if not cache.exists():
            tmp_cache.replace(cache)
        else:
            tmp_cache.unlink()

    return _close_when_exhausted(f)


def sharded_file(file_pattern: Path, mode: str, max_size: str = "4G") -> MultiFile:
    folder, name = file_pattern.parent, file_pattern.name
    assert "?" in name, f"Can't expand give file_pattern: {file_pattern}"

    n = name.count("?")
    assert 0 < n < 8
    assert "?" * n in name, f"The '?' need to be adjacents in {file_pattern}"
    assert "r" not in mode
    files = (folder / name.replace("?" * n, f"%0{n}d" % i) for i in range(10 ** n))

    return MultiFile(files, mode, max_size)


class SplitFile:
    def __init__(self, filename: Path, chunk: int, n_chunks: int, mode: str = "r"):
        assert mode == "r"
        size = os.path.getsize(filename)
        self.handle = open(filename, mode)
        start = chunk * size // n_chunks
        self.end: int = (chunk + 1) * size // n_chunks

        if start > 0:
            self.handle.seek(start - 1)
            # Skip incomplete line. This avoid crashing when reading eg the middle
            # of a unicode char. `self.handle.buffer` is a binary file reader.
            self.handle.buffer.readline()  # type: ignore

    def __enter__(self):
        return self

    def __iter__(self):
        while True:
            line = self.handle.readline()
            if not line:
                return

            yield line
            if self.handle.tell() >= self.end:
                return

    def readlines(self):
        return list(self.__iter__())

    def close(self):
        self.handle.close()

    def __exit__(self, *args):
        self.close()


def get_block_readers(filename: Path, n_readers, mode="t"):
    index_filename = filename.parent / (filename.name + ".index")
    if not index_filename.exists():
        return [gzip.open(filename, "r" + mode)]
    index: List[int] = np.load(index_filename)
    n_chunks = len(index)
    chunk_per_reader = int(np.ceil(n_chunks / n_readers))
    n_readers = int(np.ceil(n_chunks / chunk_per_reader))

    start = 0
    readers = []
    for i in range(n_readers):
        end = index[min((i + 1) * chunk_per_reader - 1, n_chunks - 1)]
        r = _blocked_gzip_reader(filename, start, end, mode)
        readers.append(r)
        start = end
    return readers


def block_reader(filename: Path) -> Iterable[str]:
    root, pattern = str(filename)[:-1].split("[", 1)
    assert root.endswith(".gz"), "Can only read block of a .gz file for now."

    ii, nn = pattern.strip().split("/")
    i, n_readers = int(ii), int(nn)

    index_filename = root + ".index"
    assert os.path.exists(
        index_filename
    ), f"Index {index_filename} not found for {filename}"
    index: List[int] = np.load(index_filename)
    n_chunks = len(index)
    chunk_per_reader = int(np.ceil(n_chunks / n_readers))
    n_readers = int(np.ceil(n_chunks / chunk_per_reader))
    # I'm not sure how to handle the case where there is less reader than expected.
    # Currently we return empty readers.

    start = 0
    if i > 0:
        start = index[min((i - 1) * chunk_per_reader, n_chunks - 1)]
    end = index[min(i * chunk_per_reader, n_chunks - 1)]
    return _blocked_gzip_reader(root, start, end, mode="t")


def _blocked_gzip_reader(filename, start, end, mode="t") -> Iterable[str]:
    handle = gzip.open(filename, "r" + mode)
    handle.seek(start)
    try:
        while handle.tell() < end:
            line = handle.readline()
            if not line:
                break
            yield line
    finally:
        handle.close()


class BlockedGzipWriter(MultiFile):
    """Writes a Gzip files which can be read by block.

    Decreasing the block size may hurt compression, but provides more split points.
    """

    def __init__(self, filename: Path, mode: str, block_size: str = "256M"):
        assert "w" in mode
        self.filename = Path(filename)
        self.index: List[int] = []
        self.zipfile: Optional[gzip.GzipFile] = None
        super().__init__([], mode, block_size)

    def _open_next_handle(self) -> bool:
        """Here we never actually close/open handles,
        we just write the end of block sequence."""
        if not self.current_handle:
            mode = self.mode + "t"
            self.current_handle = tp.cast(TextIO, gzip.open(self.filename, mode))
            assert isinstance(self.current_handle.buffer, gzip.GzipFile)
            self.zipfile = self.current_handle.buffer
            return True

        # Use Z_FULL_FLUSH to allow random access:
        # https://github.com/madler/zlib/blob/cacf7f1d4e3d44d871b605da3b647f07d718623f/zlib.h#L313
        self.current_handle.buffer.flush(zlib_mode=zlib.Z_FULL_FLUSH)  # type: ignore
        self.index.append(self.current_handle.tell())
        self.current_block_size = 0
        return True

    def flush(self):
        assert self.current_handle is not None
        self.current_handle.flush()

    def close(self):
        if self.current_handle is None:
            return
        self.current_handle.flush()
        self.index.append(self.current_handle.tell())
        self.current_handle.close()
        self.current_handle = None
        index = np.array(self.index, dtype=np.uint64)
        with open(str(self.filename) + ".index", "wb") as o:
            np.save(o, index)


def grouper(iterable, n):
    group = []
    for x in iterable:
        group.append(x)
        if len(group) == n:
            yield group
            group = []
    if group:
        yield group


PROCESS = psutil.Process()


def mem_footprint_gb(pid=None):
    rss = PROCESS.memory_info().rss
    return rss / 1_000_000_000


def _tmp(output: Path) -> Path:
    suffix = "".join(output.suffixes)
    suffix = ".tmp" + suffix
    prefix = output.name[: -len(suffix)]
    _, tmp_path = tempfile.mkstemp(dir=output.parent, prefix=prefix, suffix=suffix)
    return Path(tmp_path)


@functools.lru_cache()
def _tmp_dir() -> Path:
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        return Path("/scratch/slurm_tmpdir") / job_id

    checkpoint = Path("/checkpoint") / os.environ.get("USER", "")
    if checkpoint.exists():
        tmp = checkpoint / "tmp"
        tmp.mkdir(exist_ok=True)
        return tmp

    return Path("/tmp")


if __name__ == "__main__":
    multiprocessing.set_start_method("fork")
    main(sys.argv[1:])
