# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import subprocess
from pathlib import Path
from typing import List

import func_argparse
import numpy as np

from cc_net import jsonql


def get_index(file: Path) -> Path:
    return file.parent / (file.name + ".index")


def _get_tmp(output: Path) -> Path:
    return output.parent / (output.stem + ".tmp" + output.suffix)


def reshard(
    inputs: List[Path],
    output: Path,
    tmp: Path = None,
    free_original: bool = False,
    rm_original: bool = False,
) -> Path:
    """Read the given files and concatenate them to the output file.

    Can remove original files on completion, or just write dummy content into them to free disk.
    """
    if tmp is None:
        tmp = _get_tmp(output)
    logging.info(f"Resharding {inputs} to {tmp}, will move later to {output}")
    jsonql.run_pipes(file=inputs, output=tmp)
    tmp.replace(output)
    tmp_index = get_index(tmp)
    if tmp_index.exists():
        tmp_index.replace(get_index(output))

    if not (free_original or rm_original):
        return output

    for _input in inputs:
        if rm_original:
            _input.unlink()
        elif free_original:
            # Overwrite the previous file.
            # This frees up disk space and allows doit to properly track the success.
            _input.write_text(f"Resharded into {output}")
        if get_index(_input).is_file():
            get_index(_input).unlink()

    return output


def fast_reshard(
    inputs: List[Path],
    output: Path,
    tmp: Path = None,
    free_original: bool = False,
    rm_original: bool = False,
) -> Path:
    """Same as reshard but don't re-compress the output.

    This will lead to a bigger output file, especially if the shards are very small.
    """
    if tmp is None:
        tmp = _get_tmp(output)
    with open(tmp, "wb") as o:
        subprocess.run(["cat"] + [str(f) for f in inputs], stdout=o)

    tmp.replace(output)
    indexes_files = [get_index(i) for i in inputs]
    existing_indexes = sum(i.exists() for i in indexes_files)
    assert (
        existing_indexes == len(indexes_files) or existing_indexes == 0
    ), "some indexes don't exist."
    if existing_indexes > 0:
        indexes = [np.load(idx) for idx in indexes_files]
        for i in range(len(indexes) - 1):
            indexes[i + 1] += indexes[i][-1]
        with open(str(output) + ".index", "wb") as o:
            np.save(o, np.concatenate(indexes))

    if not (free_original or rm_original):
        return output

    for _input in inputs:
        if rm_original:
            _input.unlink()
        elif free_original:
            # Overwrite the previous file.
            # This frees up disk space and allows doit to properly track the success.
            _input.write_text(f"Resharded into {output}")
        if get_index(_input).is_file():
            get_index(_input).unlink()

    return output


def determine_groups(
    inputs: List[Path], target_size: int = 4 * 1024 ** 3
) -> List[List[Path]]:
    if len(inputs) == 0:
        return []

    sample = inputs[:10]
    typical_size = sum(s.stat().st_size for s in sample) / len(sample)
    group_size = min(target_size // typical_size, len(inputs))
    group_size = max(group_size, 1)

    return jsonql.grouper(inputs, group_size)


if __name__ == "__main__":
    func_argparse.single_main(reshard)
