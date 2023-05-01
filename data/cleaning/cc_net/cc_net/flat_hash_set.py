# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import sys
import time
import warnings
from typing import Iterable, Iterator, Sequence, Sized, Tuple, Type

import numpy as np

HASH_TYPE: Type[np.uint64] = np.uint64

GETPY_WARNING = False


class AbstractDedupHashSet(Sized, Iterable[np.uint64]):
    """A dict-like that returns `True` for keys that have been added more than once.

    The API is batched and expect np.array as input. This batching grants better
    perf when using the C++ implementation.
    """

    dtype: Type[np.uint64] = HASH_TYPE

    def __repr__(self):
        implementation = type(self).__name__
        return f"[{implementation}, len: {len(self)}"

    def __len__(self) -> int:
        ...

    def __contains__(self, values: Sequence[np.uint64]) -> np.ndarray:
        ...

    def __getitem__(self, values) -> np.ndarray:
        ...

    def __setitem__(self, keys, values) -> None:
        ...

    def items(self) -> Iterable[Tuple[np.uint64, np.uint8]]:
        ...

    def keys(self) -> Iterable[np.uint64]:
        ...

    def __iter__(self) -> Iterator[np.uint64]:
        return iter(self.keys())

    def add(self, h, contains=None):
        """Add the given keys. First time a key is added the value is set to 0,
        then it's set to one."""
        if not isinstance(h, np.ndarray):
            h = np.array(h, dtype=HASH_TYPE)
        if contains is None:
            contains = self.__contains__(h)

        self.__setitem__(h, contains)
        return contains

    def merge(self, keys, values):
        contains = self.__contains__(keys)
        self.__setitem__(keys, contains | values)

    def dump(self, filename):
        return self.dump_np(filename)

    def load(self, filename):
        return self.load_np(filename)

    def dump_np(self, filename):
        kv_type = np.dtype([("k", HASH_TYPE), ("v", np.uint8)])
        items = np.fromiter(self.items(), dtype=kv_type, count=len(self))
        with open(filename, "wb") as f:
            np.save(f, items)

    def load_np(self, filename):
        items = np.load(str(filename))
        keys = items["k"].copy()
        values = items["v"].copy()
        self.merge(keys, values)

    def dump_np2(self, filename):
        keys = np.fromiter(
            (k for (k, v) in self.items()), dtype=HASH_TYPE, count=len(self)
        )
        with open(filename, "wb") as f:
            np.save(f, keys)

        values = np.fromiter(
            (v for (k, v) in self.items()), dtype=np.uint8, count=len(self)
        )
        with open(str(filename) + ".val", "wb") as f:
            np.save(f, values)

    def load_np2(self, filename):
        keys = np.load(filename)
        values = np.load(str(filename) + ".val")
        self.merge(keys, values)


class NaiveHashSet(dict, AbstractDedupHashSet):
    """Pure python implementation of AbstractDedupHashSet.

    This implementation is quite fast, since Python dict are heavily optimized.
    """

    def __init__(self, iterable=None):
        super().__init__()
        global GETPY_WARNING
        if GETPY_WARNING:
            warnings.warn(
                "Module 'getpy' not found. Deduplication will take more RAM."
                " Try `pip install cc_net[getpy]"
            )
        GETPY_WARNING = False

    def __contains__(self, values):
        """Returns `True` if the object has been added at list once."""
        contains_point = super().__contains__
        return np.fromiter(
            map(contains_point, values), count=len(values), dtype=np.uint8
        )

    def __getitem__(self, values):
        """Returns `True` if the object has been added at list twice."""
        get_point = super().get
        return np.fromiter(
            map(lambda x: get_point(x, False), values),
            count=len(values),
            dtype=np.uint8,
        )

    def __setitem__(self, keys, values):
        assert len(keys) == len(values)
        for k, v in zip(keys, values):
            dict.__setitem__(self, k, v)


try:
    import getpy as gp  # type: ignore

    class _FlatHashSet(gp.Dict, AbstractDedupHashSet):
        """C++ backed implementation of AbstractDedupHashSet.

        This implementation is slightly slower than the Python one but uses
        3x less RAM.
        See https://github.com/atom-moyer/getpy.
        """

        def __init__(self):
            super().__init__(HASH_TYPE, np.uint8, default_value=False)

        def __contains__(self, h):
            """Returns `True` if the object has been added at list once."""
            if not isinstance(h, np.ndarray):
                h = np.array(h, dtype=HASH_TYPE)
            c = gp.Dict.__contains__(self, h)
            c.dtype = np.uint8
            return c

        def dump(self, filename):
            return self.dump_gp(filename)

        def load(self, filename):
            return self.load_gp(filename)

        def dump_gp(self, filename):
            return gp.Dict.dump(self, str(filename))

        def load_gp(self, filename):
            """Override gp.Dict.load, to correctly merge values instead of overwriting."""
            other = gp.Dict(HASH_TYPE, np.uint8, default_value=False)
            other.load(str(filename))
            n = len(other)
            keys = np.fromiter(
                (k for (k, v) in other.items()), dtype=HASH_TYPE, count=n
            )
            values = np.fromiter(
                (v for (k, v) in other.items()), dtype=np.uint8, count=n
            )
            self.merge(keys, values)

    FlatHashSet: Type[AbstractDedupHashSet] = _FlatHashSet
except ImportError:
    GETPY_WARNING = True
    FlatHashSet = NaiveHashSet


def timeit(message, function, *args):
    start = time.time()
    function(*args)
    end = time.time()
    print(message, f"took {end - start:.0f}s")


def compare_load(*filenames):
    assert filenames, "No file given"

    def load_list():
        hashes = []
        for f in filenames:
            h = FlatHashSet()
            h.load(f)
            print(f"Loaded {h} from {f}.")
            hashes.append(h)
        return hashes

    def load_all(load, ext):
        hashes = FlatHashSet()
        for f in filenames:
            load(hashes, f + ext)

    def dump_all(hashes, dump, ext):
        for h, f in zip(hashes, filenames):
            dump(h, f + ext)

    hashes = load_list()
    dump_gp = getattr(FlatHashSet, "dump_gp")
    if dump_gp is not None:
        timeit("Dumping using gp.dump", dump_all, hashes, dump_gp, ".gp.test")
    timeit("Dumping using dump_np", dump_all, hashes, FlatHashSet.dump_np, ".npy.test")
    timeit(
        "Dumping using dump_np2", dump_all, hashes, FlatHashSet.dump_np2, ".npy2.test"
    )

    load_gp = getattr(FlatHashSet, "load_gp")
    if load_gp is not None:
        timeit("Loading using gp.load", load_all, load_gp, ".gp.test")
    timeit("Loading using load_np", load_all, FlatHashSet.load_np, ".npy.test")
    timeit("Loading using load_np2", load_all, FlatHashSet.load_np2, ".npy2.test")

    # Loading 10 shards:
    # [dedup] Dumping using gp.dump took 52s
    # [dedup] Dumping using dump_np took 270s
    # [dedup] Dumping using dump_np2 took 483s
    #
    # [dedup] Loading using gp.load took 654s
    # [dedup] Loading using load_np took 82s
    # [dedup] Loading using load_np2 took 76s


if __name__ == "__main__":
    compare_load(*sys.argv[1:])
