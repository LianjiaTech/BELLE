# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import json
from pathlib import Path

import pytest

import cc_net
import cc_net.minify as minify
from cc_net import jsonql, process_wet_file
from cc_net.minify import (
    HASH_SIZE,
    decode_hashes,
    encode_hashes,
    encode_line_ids,
    get_hashes,
)


def test_encode_decode():
    sentences = ["Hello world !", "Is everyone happy in here ?"]
    hashes = get_hashes(sentences)
    assert all([len(h) == HASH_SIZE for h in hashes])
    hashes_int = [minify._b2i(h) for h in hashes]
    encoded = encode_hashes(hashes)
    decoded = decode_hashes(encoded)
    assert all([len(d) == HASH_SIZE for d in decoded])

    decoded_int = [minify._b2i(d) for d in decoded]

    assert hashes_int == decoded_int
    assert hashes == decoded


def test_minify():
    doc = {
        "raw_content": "Hello world !\nIs everyone happy in here ?",
        "language": "en",
        "perplexity": 120.0,
        "line_ids": [0, 4],
    }
    expected = {"line_ids": "AAAEAA==", "language": "en", "perplexity": 120.0}
    minifier = minify.Minifier()
    assert expected == minifier(doc)


@pytest.fixture
def http_from_disk(monkeypatch):
    def read_sample_file(url: str, n_retry: int = 3) -> bytes:
        expected_url = process_wet_file.WET_URL_ROOT + "/crawl-data/sample.warc.wet"
        assert expected_url == url
        file = Path(__file__).parent / "data" / "sample.warc.txt"
        return file.read_bytes()

    monkeypatch.setattr(cc_net.jsonql, "request_get_content", read_sample_file)


def test_minify_and_fetch(http_from_disk, tmp_path: Path):
    full_quotes = """Don't part with your illusions. When they are gone you may still exist, but you have ceased to live.
Education: that which reveals to the wise, and conceals from the stupid, the vast limits of their knowledge.
Facts are stubborn things, but statistics are more pliable.
Fiction is obliged to stick to possibilities. Truth isn't."""
    # We don't need no education.
    chosen_quotes = "\n".join(
        l for l in full_quotes.splitlines() if "Education" not in l
    )

    cc_doc = {
        "url": "http://sample_english.com",
        "date_download": "2019-03-18T00:00:00Z",
        "digest": "sha1:XQZHW7QWIG54HVAV3KPRW6MK5ILDNCER",
        "source_domain": "sample_english.com",
        "title": "Famous Mark Twain Quotes",
        "raw_content": full_quotes,
        "cc_segment": "crawl-data/sample.warc.wet",
        "nlines": 4,
        "length": 353,
    }

    ccnet_metadata = {
        "language": "en",
        "language_score": 0.99,
        "perplexity": 151.5,
        "bucket": "head",
        "raw_content": chosen_quotes,
        "nlines": 3,
        "length": len(chosen_quotes),
        "original_nlines": 4,
        "original_length": 353,
        "line_ids": [0, 2, 3],
    }
    ccnet_doc = dict(cc_doc, **ccnet_metadata)
    mini = minify.Minifier()(ccnet_doc.copy())
    assert mini is not ccnet_doc

    important_fields = [
        "url",
        "digest",
        "cc_segment",
        "language",
        "language_score",
        "perplexity",
        "bucket",
        "line_ids",
    ]
    expected = {k: ccnet_doc[k] for k in important_fields}
    expected["line_ids"] = encode_line_ids(expected["line_ids"])  # type: ignore
    assert expected == mini

    with jsonql.open_write(tmp_path / "sample.json") as o:
        print(json.dumps(mini), file=o)
    fetcher = minify.MetadataFetcher(tmp_path)
    # line_ids is removed when unminifying
    ccnet_doc.pop("line_ids")
    assert ccnet_doc == fetcher(cc_doc)


def test_fetch(http_from_disk, tmp_path: Path):
    mini_docs = [
        {
            "url": "http://sample_chinese.com",
            "digest": "sha1:Y4E6URVYGIAFNVRTPZ5S3J64RTZTP6HJ",
            "cc_segment": "crawl-data/sample.warc.wet",
            "line_ids": encode_line_ids([2]),
            "bucket": "not_that_great",
        },
        {
            "url": "http://sample_english.com",
            "digest": "sha1:XQZHW7QWIG54HVAV3KPRW6MK5ILDNCER",
            "cc_segment": "crawl-data/sample.warc.wet",
            "line_ids": encode_line_ids([3]),
            "bucket": "top_notch",
        },
    ]
    with jsonql.open_write(tmp_path / "sample.json") as o:
        for mini in mini_docs:
            print(json.dumps(mini), file=o)

    fetcher = minify.MetadataFetcher(tmp_path)
    cc = process_wet_file.CCSegmentsReader(["crawl-data/sample.warc.wet"])
    docs = [d for d in fetcher.map(cc) if d is not None]
    assert cc.retrieved_segments == 1

    # Note: documents are retrieved as they are ordered in the .warc.wet file
    assert [
        "Facts are stubborn things, but statistics are more pliable.",
        "事實是固執的東西，但統計數字卻比較柔和。",
    ] == [d["raw_content"] for d in docs]
    assert ["top_notch", "not_that_great"] == [d["bucket"] for d in docs]
