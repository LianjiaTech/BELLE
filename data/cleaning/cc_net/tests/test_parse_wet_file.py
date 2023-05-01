# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from pathlib import Path

from cc_net import process_wet_file


def test_parsing():
    sample = Path(__file__).parent / "data" / "sample.warc.txt"
    with open(sample) as f:
        documents = list(process_wet_file.parse_warc_file(f))

    expected_urls = [
        "http://sample_english.com",
        "http://sample_chinese.zh",
        "http://sample_russian.ru",
    ]
    assert expected_urls == [d["url"] for d in documents]
    expected_domains = ["sample_english.com", "sample_chinese.zh", "sample_russian.ru"]
    assert expected_domains == [d["source_domain"] for d in documents]

    expected_date = [
        "2019-03-18T00:00:00Z",
        "2019-03-18T00:00:01Z",
        "2019-03-18T00:00:02Z",
    ]
    assert expected_date == [d["date_download"] for d in documents]

    expected_title = [
        "Famous Mark Twain Quotes",
        "馬克·吐溫名言",
        "Цитаты знаменитого Марка Твена",
    ]
    assert expected_title == [d["title"] for d in documents]

    expected_quotes = """Don't part with your illusions. When they are gone you may still exist, but you have ceased to live.
Education: that which reveals to the wise, and conceals from the stupid, the vast limits of their knowledge.

Facts are stubborn things, but statistics are more pliable.
Fiction is obliged to stick to possibilities. Truth isn't.
"""

    assert expected_quotes == documents[0]["raw_content"]
