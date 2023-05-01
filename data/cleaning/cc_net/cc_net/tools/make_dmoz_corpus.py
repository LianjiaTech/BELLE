# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

"""
This code is used to train a fastText classifier to label document with DMOZ categories.

The data, distributed under the cc-by 3.0 license
(https://web.archive.org/web/20140605215533/http://www.dmoz.org/license.html),
can be downloaded from
https://web.archive.org/web/20140617145301/http://rdf.dmoz.org/rdf/content.rdf.u8.gz.
"""

import urllib.request
from io import StringIO
from pathlib import Path
from typing import Dict, Set
from urllib.parse import urlparse

import func_argparse
from lxml import etree  # type: ignore

from cc_net import jsonql

TaggedUrls = Dict[str, Set[str]]
DMOZ_TAGS_URL = "https://web.archive.org/web/20140617145301/http://rdf.dmoz.org/rdf/content.rdf.u8.gz"


def add_tags(url: str, tags: Set[str], url2tags: TaggedUrls):
    if url in url2tags:
        url2tags[url] &= tags
    else:
        url2tags[url] = tags


def load_tags(filename: Path = None) -> TaggedUrls:
    if filename is None:
        with StringIO("".join(jsonql.open_remote_file(DMOZ_TAGS_URL))) as dmoz:
            tree = etree.parse(dmoz)
    else:
        tree = etree.parse(str(filename))

    root = tree.getroot()
    url2tags: Dict[str, Set[str]] = {}
    for external_page in root.iterfind("{http://dmoz.org/rdf/}ExternalPage"):
        url = external_page.get("about")
        domain = urlparse(url).netloc
        for topic in external_page.iterfind("{http://dmoz.org/rdf/}topic"):
            # print(url, topic.text)
            # Tags looks like Top/Arts/Animation/Anime/Collectibles
            tags = set(topic.text.split("/")[1:])
            add_tags(url, tags, url2tags)
            add_tags(domain, tags, url2tags)
    return url2tags


def dl(output: Path) -> None:
    urllib.request.urlretrieve(DMOZ_TAGS_URL, output)


def make_corpus(file: Path, tags_file: Path = None, output: Path = None) -> None:
    """
    Loads a tags file and create a training dataset using the given webpages.

    Arguments:
        - file: CC shard file
        - tags_file: dmoz tagging file, (like the one produced by `dl`)
        - output: ""
    """
    url2tags = load_tags(tags_file)
    with jsonql.open_write(output) as o:
        for document in jsonql.read_jsons(file):
            if not document:
                continue
            url = document["url"]
            domain = document["source_domain"]

            if url in url2tags:
                tags = url2tags[url]
            elif domain in url2tags:
                tags = url2tags[domain]
            else:
                continue

            if len(tags) == 0:
                continue

            fasttext_tags = ["__label__" + tag for tag in tags]
            content = document["tokenized"].replace("\n", " ").lower()
            if len(content) > 200:
                print(" ".join(fasttext_tags), content, file=o)  # type: ignore


if __name__ == "__main__":
    func_argparse.single_main(make_corpus)
