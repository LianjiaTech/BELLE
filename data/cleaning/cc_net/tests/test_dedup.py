# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import json
from pathlib import Path
from typing import Iterable, Sequence

from cc_net import dedup, jsonql
from cc_net.dedup import str_hash
from cc_net.flat_hash_set import FlatHashSet


def text(*args: str) -> str:
    return "\n".join(args)


def write_docs(file: Path, docs: Iterable[Sequence[str]]):
    file.parent.mkdir(exist_ok=True)
    with open(file, "w") as f:
        for sentences in docs:
            doc = dict(text=text(*sentences))
            print(json.dumps(doc), file=f)


def as_dict(hash_set):
    if not isinstance(hash_set, dict):
        hash_set = {k: v for (k, v) in hash_set.items()}
    return hash_set


def load_hashes(file):
    results = dedup.FlatHashSet()
    results.load(file)
    return as_dict(results)


LENGTHS = ["original_length", "length"]


def assert_documents_equal(expected, actual, ignoring={}):
    expected = [{k: doc[k] for k in doc if k not in ignoring} for doc in expected]
    actual = [{k: doc[k] for k in doc if k not in ignoring} for doc in expected]
    assert expected == actual


def test_simple_dedup(tmp_path: Path) -> None:
    write_docs(
        tmp_path / "docs.json",
        [
            ["_Hello", "_World", "I'm so original"],
            ["_world", "I'm originaler", "_Hello"],
        ],
    )
    results = list(dedup.deduplicate(tmp_path / "docs.json", field="text"))
    expected = [
        # First document is untouched
        dict(
            text=text("_Hello", "_World", "I'm so original"),
            original_nlines=3,
            nlines=3,
            line_ids=[0, 1, 2],
        ),
        # Second documents loses several lines
        dict(text="I'm originaler", original_nlines=3, nlines=1, line_ids=[1]),
    ]

    assert_documents_equal(expected, results, ignoring=LENGTHS)


def test_dedup_with_dump(tmp_path: Path):
    hashes = tmp_path / "hashes.bin"
    documents = [
        dict(text=text("_Hello", "_World", "I'm so original")),
        dict(text=text("_world", "I'm originaler", "_Hello")),
    ]
    collector = dedup.HashesCollector(field="text", output=hashes)
    list(collector.map(documents))
    results = load_hashes(hashes)
    expected = {
        str_hash(l): l.startswith("_")
        for l in ["_hello", "_world", "i'm so original", "i'm originaler"]
    }
    assert expected == results


def test_dedup_with_np_dump(tmp_path: Path):
    hashes = tmp_path / "hashes.bin"
    documents = [
        dict(text=text("_Hello", "_World", "I'm so original")),
        dict(text=text("_world", "I'm originaler", "_Hello")),
    ]
    with dedup.HashesCollector(field="text", output=hashes) as d:
        list(d.map(documents))

    results = FlatHashSet()
    results.load_np(hashes)
    expected = set(
        str_hash(l) for l in ["_hello", "_world", "i'm so original", "i'm originaler"]
    )
    assert expected == set(results.keys())


def test_dedup_from_hashes(tmp_path: Path):
    documents = [
        dict(text=text("_Hello", "World", "I'm so original")),
        dict(text=text("Good morning", "World", "I'm originaler")),
    ]
    seen = ["_hello", "i'm originaler", "world"]
    hashes = [str_hash(h) for h in seen]
    h = dedup.FlatHashSet()
    h.add(hashes)
    # Note: 'world' appears only once and won't be treated as a duplicate.
    h.add(hashes[:-1])
    h.dump(tmp_path / "hashes.bin")

    results = list(
        dedup.DuplicatesRemover("text", [tmp_path / "hashes.bin"]).map(documents)
    )
    expected = [
        dict(
            text=text("World", "I'm so original"),
            original_nlines=3,
            nlines=2,
            line_ids=[1, 2],
        ),
        dict(
            text=text("Good morning", "World"),
            original_nlines=3,
            nlines=2,
            line_ids=[0, 1],
        ),
    ]

    assert_documents_equal(expected, results, ignoring=LENGTHS)


def test_dedup_fast(tmp_path: Path):
    data = tmp_path / "data"
    part_0 = [["Hello", "_World", "I'm so original"]]
    write_docs(data / "part_0.json", part_0)
    part_1 = [["Good morning", "_World", "I'm originaler"]]
    write_docs(data / "part_1.json", part_1)
    parts = [data / "part_0.json", data / "part_1.json"]

    res = tmp_path / "res"
    res.mkdir()
    h = tmp_path / "hashes.bin"
    field = "text"
    jsonql.run_pipes(dedup.HashesCollector(field, output=h), file=parts)
    for part in parts:
        jsonql.run_pipes(
            dedup.DuplicatesRemover(field, [h]), file=part, output=res / part.name
        )
        jsonql.run_pipes(
            dedup.DuplicatesRemover(field, [h]), file=part, output=res / part.name
        )

    results_0 = list(jsonql.read_jsons(res / "part_0.json"))
    expected_0 = [
        dict(
            text=text("Hello", "I'm so original"),
            original_nlines=3,
            nlines=2,
            line_ids=[0, 2],
        )
    ]
    assert_documents_equal(expected_0, results_0, ignoring=LENGTHS)

    results_1 = list(jsonql.read_jsons(res / "part_1.json"))
    expected_1 = [
        dict(
            text=text("Good morning", "I'm originaler"),
            original_nlines=3,
            nlines=2,
            line_ids=[0, 2],
        )
    ]

    assert_documents_equal(expected_1, results_1, ignoring=LENGTHS)

    words = [w for part in [part_0, part_1] for doc in part for w in doc]
    expected = {str_hash(s.lower()): s.startswith("_") for s in words}
    assert expected == load_hashes(h)


def test_remove_duplicates_sharded(tmp_path: Path):
    data = tmp_path / "data"
    part_0 = [["Hello", "_World", "I'm so original"]]
    write_docs(data / "part_0.json", part_0)
    part_1 = [["_Good morning", "_World", "I'm originaler"]]
    write_docs(data / "part_1.json", part_1)

    h = tmp_path / "hashes"
    h.mkdir()
    h0 = FlatHashSet()
    h0.add([str_hash(s.lower()) for doc in part_0 for s in doc])
    h0.add([str_hash("_world")])
    h0.dump(h / "part_0.bin")
    assert {
        str_hash("hello"): False,
        str_hash("_world"): True,
        str_hash("i'm so original"): False,
    } == as_dict(h0)

    h1 = FlatHashSet()
    h1.add([str_hash(s.lower()) for doc in part_1 for s in doc])
    h1.add([str_hash("_good morning")])
    h1.dump(h / "part_1.bin")
    assert {
        str_hash("_good morning"): True,
        str_hash("_world"): False,
        str_hash("i'm originaler"): False,
    } == as_dict(h1)

    res = tmp_path / "res"
    res.mkdir()
    # dedup.DISABLE_MULTI_PROCESSING = True  # Simplifies debugging
    dedup.remove_duplicates_sharded(
        files=[data / "part_0.json", data / "part_1.json"],
        outputs=[res / "part_0.json", res / "part_1.json"],
        field="text",
        hashes_dir=h,
    )

    results_0 = list(jsonql.read_jsons(res / "part_0.json"))
    expected_0 = [
        dict(
            text=text("Hello", "I'm so original"),
            original_nlines=3,
            nlines=2,
            line_ids=[0, 2],
        )
    ]
    assert_documents_equal(expected_0, results_0, ignoring=LENGTHS)

    # First pass removes "_world", second "_good morning".
    results_1 = list(jsonql.read_jsons(res / "part_1.json"))
    expected_1 = [
        dict(text=text("I'm originaler"), original_nlines=3, nlines=1, line_ids=[2])
    ]

    assert_documents_equal(expected_1, results_1, ignoring=LENGTHS)
