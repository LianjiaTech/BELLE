# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import pytest

from cc_net.flat_hash_set import HASH_TYPE, FlatHashSet, NaiveHashSet


def as_dict(flat_hash_set) -> dict:
    return {k: v for (k, v) in flat_hash_set.items()}


need_getpy = pytest.mark.skipif(
    FlatHashSet == NaiveHashSet, reason="getpy isn't installed"
)


def same_behavior(test_case):
    def run_case():
        naive = as_dict(test_case(FlatHashSet))
        flat = as_dict(test_case(NaiveHashSet))
        assert naive == flat

    return need_getpy(run_case)


@same_behavior
def test_setitem(hash_set_cls):
    h = hash_set_cls()
    h[np.arange(10, dtype=h.dtype)] = np.zeros(10, dtype=np.uint8)
    h[np.arange(5, dtype=h.dtype)] = np.ones(5, dtype=np.uint8)
    return h


@same_behavior
def test_add_dup(hash_set_cls):
    h = hash_set_cls()
    h.add(np.arange(10, dtype=h.dtype))
    h.add(np.arange(5, dtype=h.dtype))

    expected = {i: i < 5 for i in range(10)}
    assert expected == as_dict(h), f"add_dup with {hash_set_cls.__name__}"
    return h


@need_getpy
def test_gp_dict():
    import getpy as gp  # type: ignore

    h = gp.Dict(HASH_TYPE, np.uint8)
    h[np.arange(10, dtype=HASH_TYPE)] = np.zeros(10, dtype=np.uint8)
    h[np.arange(5, dtype=HASH_TYPE)] = np.ones(5, dtype=np.uint8)
    expected = {i: i < 5 for i in range(10)}
    assert expected == as_dict(h)


def check_reload(h, dump, load, tmp_path):
    dump_path = tmp_path / dump.__name__
    dump(h, dump_path)
    h2 = type(h)()
    load(h2, dump_path)
    assert as_dict(h) == as_dict(h2)


@pytest.mark.parametrize("hash_set_cls", [FlatHashSet, NaiveHashSet])
def test_loading(tmp_path, hash_set_cls):
    h = hash_set_cls()
    x = np.random.randint(0, 2 ** 32, (100,), dtype=h.dtype)
    h.add(x)

    check_reload(h, hash_set_cls.dump, hash_set_cls.load, tmp_path)
    check_reload(h, hash_set_cls.dump_np, hash_set_cls.load_np, tmp_path)
    if hasattr(hash_set_cls, "dump_gp"):
        check_reload(h, hash_set_cls.dump_gp, hash_set_cls.load_gp, tmp_path)
