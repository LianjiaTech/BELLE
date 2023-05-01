# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import io
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest

from cc_net import jsonql


def bar(small_bar: str) -> str:
    return small_bar.replace(" ", " " * 10).replace("█", "█" * 10)


def get_output(transformer, data, **kwargs):
    with io.StringIO() as output:
        # Convert data to a generator so that it's not interpreted as a file list.
        jsonql.run_pipe(transformer, kwargs, file=(x for x in data), output=output)
        return output.getvalue()


def test_split(tmp_path: Path):
    data = [
        dict(text="Hello world", lang="en"),
        dict(text="Boujour les amis", lang="fr"),
        dict(text="Rock your boat", lang="en"),
    ]
    with jsonql.split(tmp_path / "{lang}.json") as split:
        list(split.map(data))
        summary = split.summary()
    assert "Found 2 splits." in summary
    en_docs = list(jsonql.read_jsons(tmp_path / "en.json"))
    assert [data[0], data[2]] == en_docs

    fr_docs = list(jsonql.read_jsons(tmp_path / "fr.json"))
    assert [data[1]] == fr_docs


def test_split_bad_pattern(tmp_path: Path):
    data = [dict(text="Hello world", lang="en")]
    with pytest.raises(KeyError):
        with jsonql.split(tmp_path / "{language}.json") as split:
            list(split.map(data))


def test_histogram():
    data = [0.1, 0.1, 0.1, 0.1, 0.4, 0.4, 0.9, 0.9]
    hist, bins = jsonql.histogram(data, bins=8, weights=None)
    np.testing.assert_almost_equal(bins, [0.1 * x for x in range(1, 10)])
    np.testing.assert_almost_equal(hist, [4, 0, 0, 2, 0, 0, 0, 2])

    data = [0, 0.1, 0.1, 0.1, 0.1, 0.4, 0.4, 0.8, 0.8, 1]
    hist, bins = jsonql.histogram(data, bins=10, weights=None)
    np.testing.assert_almost_equal(bins, [0.1 * x for x in range(11)])
    np.testing.assert_almost_equal(hist, [1, 4, 0, 0, 2, 0, 0, 0, 2, 1])


def test_display_stats():
    stats = {
        jsonql.ALL_DOCUMENTS: 100,
        "title": 80,
        "title.length": 80 * 50,
        "text": 100,
        "text.length": 100 * 1000,
        "popularity": 8,
        "popularity.val": [0.1, 0.1, 0.1, 0.1, 0.4, 0.4, 0.9, 0.9],
    }

    (title,) = jsonql.display_stats(stats, "title")
    assert "title" in title
    assert "saw 80 times" in title
    assert "average length is" in title
    assert "\n" not in title

    (text,) = jsonql.display_stats(stats, "text")
    assert "text" in text
    assert "saw 100 times" in text
    assert "average length is" in text
    assert "\n" not in text

    histogram = jsonql.display_stats(
        stats, "popularity", bins=[x / 10 for x in range(1, 10)]
    )
    assert "popularity" in histogram[0]
    assert "saw 8 times" in histogram[0]
    assert "histogram is" in histogram[0]
    assert "0.100 " + bar("████████") in histogram[1]
    assert "0.400 " + bar("████    ") in histogram[2]
    assert "0.800 " + bar("████    ") in histogram[3]

    cum_histogram = jsonql.display_stats(stats, "popularity", bins=8, cumulative=True)
    assert "popularity" in cum_histogram[0]
    assert "saw 8 times" in cum_histogram[0]
    assert "histogram is" in cum_histogram[0]
    assert "0.100 " + bar("████    ") in cum_histogram[1]
    assert "0.400 " + bar("██████  ") in cum_histogram[2]
    assert "0.800 " + bar("████████") in cum_histogram[3]


def test_describe():
    def sample(pop):
        return dict(title="Lorem", text="Lorem ipsum dolor sit amet.", popularity=pop)

    data = [sample(pop) for pop in [0.1, 0.1, 0.1, 0.1, 0.4, 0.4, 0.9, 0.9]]
    desc = get_output(
        jsonql.describe, data, columns=None, bins=[x / 10 for x in range(1, 10)]
    )

    assert "Field title saw 8 times (100.0%), average length is 5" in desc
    assert "Field text saw 8 times (100.0%), average length is 27" in desc
    assert "Field popularity saw 8 times (100.0%), histogram is" in desc
    assert "0.100 " + bar("████████") in desc
    assert "0.400 " + bar("████    ") in desc
    assert "0.800 " + bar("████    ") in desc

    desc = get_output(jsonql.describe, data, columns=["text"])
    assert "Field title saw 8 times (100.0%), average length is 5" not in desc
    assert "Field text saw 8 times (100.0%), average length is 27" in desc
    assert "Field popularity, histogram is:" not in desc


def test_custom_pipe():
    def transformer(source, sep=" "):
        for i, line in enumerate(source):
            res = f"{i}{sep}{line}"
            yield res

    data = ["hello", "world"]
    assert get_output(transformer, data) == "0 hello\n1 world\n"
    assert get_output(transformer, data, sep="_") == "0_hello\n1_world\n"


def test_open_read_write(tmp_path: Path):
    def _lines(filename: Path) -> Sequence[str]:
        # jsonql.lines calls open_read
        return list(jsonql.lines(filename))

    tmp = tmp_path
    with jsonql.open_write(tmp / "a.txt") as o:
        print("a", file=o)
    assert _lines(tmp / "a.txt") == ["a"]

    jsonql.write_jsons([{"a": 1}], tmp / "a.txt")
    assert _lines(tmp / "a.txt") == ['{"a": 1}']

    with jsonql.open_write(tmp / "a.gz") as o:
        print("a", file=o)
    assert _lines(tmp / "a.gz") == ["a"]

    with jsonql.open_write([tmp / "a0.txt", tmp / "a1.txt"]) as o:
        print("a", file=o)
    assert _lines(tmp / "a0.txt") == ["a"]
    assert not (tmp / "a1.txt").is_file()

    with jsonql.open_write([tmp / "b0.txt", tmp / "b1.txt"], max_size="1k") as o:
        print("0" * 2000, file=o)
        print("1" * 2000, file=o)
    assert _lines(tmp / "b0.txt") == ["0" * 2000]
    assert _lines(tmp / "b1.txt") == ["1" * 2000]

    with jsonql.open_write(tmp / "a_????.json") as o:
        print("a", file=o)
    assert _lines(tmp / "a_0000.json") == ["a"]
    assert not (tmp / "a_0001.json").is_file()
    assert _lines(tmp / "a_*.json") == ["a"]

    with jsonql.open_write(tmp / "b_??.json", max_size="1k") as o:
        print("0" * 2000, file=o)
        print("1" * 2000, file=o)
    assert _lines(tmp / "b_00.json") == ["0" * 2000]
    assert _lines(tmp / "b_01.json") == ["1" * 2000]
    assert _lines(tmp / "b_*.json") == ["0" * 2000, "1" * 2000]


def test_split_file(tmp_path: Path):
    file = tmp_path / "test.txt"
    content = "Hello\nWorld\n"

    with open(file, "w") as o:
        o.write(content)

    with jsonql.SplitFile(file, chunk=0, n_chunks=2) as f:
        assert f.readlines() == ["Hello\n"]

    with jsonql.SplitFile(file, chunk=1, n_chunks=2) as f:
        assert f.readlines() == ["World\n"]


def test_split_file_middle_of_line(tmp_path: Path):
    file = tmp_path / "test.txt"
    content = "Hello _|_\nWorld\n"
    # split is here   ^

    with open(file, "w") as o:
        o.write(content)

    with jsonql.SplitFile(file, chunk=0, n_chunks=2) as f:
        assert f.readlines() == ["Hello _|_\n"]

    with jsonql.SplitFile(file, chunk=1, n_chunks=2) as f:
        assert f.readlines() == ["World\n"]


def test_split_file_middle_of_char(tmp_path: Path):
    file = tmp_path / "test.txt"
    content = "Hello\U0001F40D\nWorld\n"
    # split is here       ^^

    with open(file, "w") as o:
        o.write(content)

    with jsonql.SplitFile(file, chunk=0, n_chunks=2) as f:
        assert f.readlines() == ["Hello🐍\n"]

    with jsonql.SplitFile(file, chunk=1, n_chunks=2) as f:
        assert f.readlines() == ["World\n"]


def test_blocked_gzip(tmp_path: Path):
    file = tmp_path / "test.gz"
    f = str(file)
    # Each object is 10/11 bytes long. We have 2 of them by block.
    content = ['{"xx": %d}' % i for i in range(80)]
    with jsonql.BlockedGzipWriter(file, "wt", block_size="20B") as o:
        for line in content:
            print(line, file=o)

    jr = jsonql.JsonReader(strict=True)
    expected = list(jr.map(content))
    # read as one file
    assert expected == list(jsonql.read_jsons(file))
    # read first block
    assert expected[:2] == list(jsonql.read_jsons(f + "[0/40]"))
    # read last block
    assert expected[-2:] == list(jsonql.read_jsons(f + "[39/40]"))

    readers = jsonql.get_block_readers(file, 9)
    read_as_several_files = [list(jsonql.read_jsons(r)) for r in readers]
    # 40 splits of 2 docs, 9 readers -> 5 splits, 10 docs per reader
    assert list(jsonql.grouper(expected, 10)) == read_as_several_files


def test_enter_exit(capsys):
    class MyTransformer(jsonql.Transformer):
        def __enter__(self):
            print("trans: started")
            self.ready = True
            return self

        def __exit__(self, *args):
            print("trans: done")

        def do(self, x):
            return (x, x)

    def acc(values):
        print("acc: started")
        res = 0
        for (x, _) in values:
            res += int(x)
        print("acc: done")
        yield f"acc: result={res}"

    t = MyTransformer()
    data = (str(x) for x in range(10))
    print("pipeline: started")
    # Print to stdout.
    jsonql.run_pipes(t, acc, file=data)
    print("pipeline: done")
    out = capsys.readouterr().out
    assert (
        "\n".join(
            [
                "pipeline: started",
                "trans: started",
                "acc: started",
                "acc: done",
                f"acc: result=45",
                # Transformers are closed at the very end.
                "trans: done",
                "pipeline: done\n",
            ]
        )
        == out
    )


def test_write_to_stdout(capsys):
    lines = [str(x) for x in range(10)]
    jsonql.run_pipes(file=iter(lines))
    out = capsys.readouterr().out
    assert out == "\n".join(lines) + "\n"


def test_write_to_stdout_handle_newlines(capsys):
    lines = [str(x) + "\n" for x in range(10)]
    jsonql.run_pipes(file=iter(lines))
    out = capsys.readouterr().out
    assert out == "".join(lines)


def test_multiprocess(capsys):
    mult = jsonql.Mapper(lambda x: f"2x = {2 * int(x)}")
    jsonql.run_pipes(mult, processes=2, file=(str(x) for x in range(10)))
    out = set(capsys.readouterr().out.strip("\n").split("\n"))
    assert set(f"2x = {2 * x}" for x in range(10)) == out
