# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Creates mono-lingual corpus from Wikipedia.
"""

import functools
import re
import subprocess
import urllib.request
from pathlib import Path
from typing import Dict

import func_argparse
from bs4 import BeautifulSoup  # type: ignore

from cc_net import jsonql, text_normalizer

CIRRUS_URL = "https://dumps.wikimedia.org/other/cirrussearch"
CIRRUS_DUMP_RE = re.compile(r"^(.*)wiki-\d+-cirrussearch-content\.json\.gz")


def tmp(file: Path) -> Path:
    return file.parent / ("tmp." + file.name)


def opening(file: Path, output: Path = None, n_docs: int = 1_000_000):
    """Will dump the tokenized opening text of the given Wikipedia.

    Args:
        - file: File containing the Wikipedia dump.
        - output: Output file.
        - n_docs: How many docs to parse
        - tokenize: whether to tokenize the text
        - lang: Language code used to chose the tokenizer
    """
    assert file.exists()
    return jsonql.run_pipes(
        functools.partial(extract_opening_text, n_docs=n_docs),
        file=file,
        output=tmp(output) if output else None,
    )
    if output:
        tmp(output).replace(output)


def extract_opening_text(source, n_docs: int = 10_000):
    i = 0
    for doc in jsonql.read_jsons(source):
        if not doc:
            continue

        text = doc.get("opening_text")
        if not text:
            continue

        yield text_normalizer.normalize(text)
        i += 1
        if i >= n_docs:
            break


def dl(lang: str, output_dir: Path, date: str = None):
    """Download the cirrus extract for the given lang.

    See https://dumps.wikimedia.org/other/cirrussearch for the full list of files.

    Args:
        - lang: The Wikipedia code for the language.
        - output_dir: Output directory. File will be `{lang}.json.gz`
        - date: Date of a specific Cirrus dump.
    """

    urls = get_cirrus_urls(date)
    assert (
        lang in urls
    ), f"--lang {lang} not found. Available languages are: {urls.keys()}"

    assert output_dir, "--output_dir folder needed."
    output_dir.mkdir(exist_ok=True)
    output = output_dir / (lang + ".json.gz")
    print(f"Downloading {lang} wiki from {urls[lang]} to {output}")
    wget(urls[lang], output)


def get_cirrus_urls(date: str = None) -> Dict[str, str]:
    if date is None:
        cirrus_page = BeautifulSoup(
            urllib.request.urlopen(CIRRUS_URL), features="html.parser"
        )
        dumps = [a.get("href").strip("/") for a in cirrus_page.findAll("a")]
        dumps.remove("..")
        dumps.remove("current")
        # We take the oldest dump since the most recent might be incomplete.
        # The page only link to the N latest dumps so the dump won't be too old.
        date = min(dumps)

    cirrus_url = "/".join((CIRRUS_URL, date))
    print("Will use the Wikipedia dump from:", date, cirrus_url)
    cirrus_page = BeautifulSoup(
        urllib.request.urlopen(cirrus_url), features="html.parser"
    )
    urls = {}
    for link in cirrus_page.findAll("a"):
        match = CIRRUS_DUMP_RE.match(link.get("href"))
        if not match:
            continue

        urls[match.group(1)] = "/".join([cirrus_url, link.get("href")])
    assert urls, f"No valid download urls found at {cirrus_url}"
    return urls


def wget(url: str, output: Path):
    subprocess.run(["wget", url, "-O", tmp(output), "-q"], check=True)
    tmp(output).replace(output)
    assert (
        output.stat().st_size > 10_000
    ), f"File {output} downloaded from {url} looks too small"


if __name__ == "__main__":
    func_argparse.main(dl, opening)
