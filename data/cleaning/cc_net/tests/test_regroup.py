# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import time

from cc_net import jsonql, regroup


def check_regroup(tmp_path, regroup_fn, check_blocks_boundaries=False):
    n_shards = 4
    n_docs = 20
    shards = [
        [dict(id=i, shard=s, raw_content="hello world") for i in range(n_docs)]
        for s in range(n_shards)
    ]
    shards_files = [tmp_path / f"{s:04d}.json.gz" for s in range(n_shards)]
    for shard, shard_file in zip(shards, shards_files):
        jsonql.run_pipes(inputs=shard, output=shard_file)
    regroup_file = tmp_path / "regroup.json.gz"
    start = time.time()
    regroup_fn(shards_files, regroup_file)
    duration = time.time() - start
    print(f"{regroup_fn.__module__}.{regroup_fn.__name__} took {duration}s")

    regrouped = list(jsonql.read_jsons(regroup_file))
    assert [doc for shard in shards for doc in shard] == regrouped

    readers = jsonql.get_block_readers(regroup_file, n_shards)
    if not check_blocks_boundaries:
        assert [doc for shard in shards for doc in shard] == [
            doc for reader in readers for doc in jsonql.read_jsons(reader)
        ]
        return

    for shard, reader in zip(shards, readers):
        block = [doc for doc in jsonql.read_jsons(reader)]
        assert shard == block


def test_regroup(tmp_path):
    # With regroup boundaries will be every 256Mb.
    check_regroup(tmp_path, regroup.reshard, check_blocks_boundaries=False)


def test_fast_regroup(tmp_path):
    # With fast regroup boundaries should match the shards.
    check_regroup(tmp_path, regroup.fast_reshard, check_blocks_boundaries=True)
