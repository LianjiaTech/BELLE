# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import base64
import hashlib
import itertools
import urllib.parse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Union

import numpy as np

from cc_net import jsonql
from cc_net.execution import get_executor
from cc_net.jsonql import mem_footprint_gb

HASH_SIZE = 4
HASH_TYPE = np.uint32

PUBLIC_FIELDS = ["url", "digest"]
COMPUTED_FIELDS = ["cc_segment", "language", "language_score", "bucket", "perplexity"]
DATA = Path(__file__).parent.parent / "data"


# This is similar to dedup methods but with use 32 bits hashes.
def _b2i(b: bytes) -> int:
    return np.frombuffer(b[:HASH_SIZE], dtype=HASH_TYPE, count=1, offset=0).item(0)


def _str_hash(s: str) -> int:
    h = hashlib.sha1(bytes(s, encoding="utf-8"))
    return _b2i(h.digest())


def get_hashes(lines: Iterable[str]) -> List[bytes]:
    h = HASH_SIZE
    return [hashlib.sha1(bytes(l, encoding="utf-8")).digest()[:h] for l in lines]


def encode_hashes(hashes: Iterable[bytes]) -> str:
    return base64.b64encode(b"".join(hashes)).decode("ascii")


def encode_as_hashes(lines: Iterable[str]) -> str:
    return encode_hashes(get_hashes(lines))


def decode_hashes(compact: str) -> List[bytes]:
    all_hashes = base64.b64decode(compact)
    res = []
    assert len(all_hashes) % HASH_SIZE == 0
    for i in range(len(all_hashes) // HASH_SIZE):
        chunk = all_hashes[i * HASH_SIZE : (i + 1) * HASH_SIZE]
        res.append(chunk)

    return res


def encode_line_ids(line_ids: Sequence[int]) -> str:
    arr = np.array(line_ids, dtype="<u2")
    return base64.b64encode(arr.tobytes()).decode("ascii")


def decode_line_ids(compact: str) -> List[int]:
    ids_bytes = bytearray(base64.b64decode(compact))
    return np.ndarray(len(ids_bytes) // 2, dtype="<i2", buffer=ids_bytes)


def get_doc_key(digest: str) -> int:
    assert digest.startswith("sha1:")
    h = base64.b32decode(digest[5:])
    return _b2i(h[:HASH_SIZE])


class Minifier(jsonql.Transformer):
    ready = True

    def __init__(self):
        self.fields = frozenset(COMPUTED_FIELDS + PUBLIC_FIELDS)

    def do(self, doc: dict) -> Optional[dict]:
        line_ids: List[int] = doc.pop("line_ids")
        fields = self.fields
        keys = list(doc.keys())
        for k in keys:
            if k not in fields:
                doc.pop(k, None)
        p = doc.get("perplexity", 0)
        doc["line_ids"] = encode_line_ids(line_ids)
        if p:
            doc["perplexity"] = round(p, 1)
        s = doc.get("language_score", 0)
        if s:
            doc["language_score"] = round(s, 2)
        return doc


class MetadataFetcher(jsonql.Transformer):
    """Reads documents from CC snapshot and join precomputed metadata.

    CC snapshots are split in segments. Each segment is 64Mb long.
    The metadata must also be stored in segments of the same size and names.
    """

    def __init__(self, folder: Union[Path, str]):
        self.ready = True
        self.metadata: Dict[int, dict] = {}

        self._segments: Set[str] = set()
        self.read_doc = 0
        self.missed_doc = 0
        self.missed_par = 0
        self.processed_par = 0

        if isinstance(folder, str):
            # detect path passed as string
            if urllib.parse.urlparse(folder).scheme == "":
                folder = Path(folder)
                assert folder.exists(), f"Metadata folder not found: {folder}"

        self.folder = folder
        self.segment: str = ""
        self.segments_read_twice = 0

    def meta_file(self, segment: str) -> str:
        file_name = segment.split("/")[-1]
        assert file_name.endswith(".warc.wet.gz") or file_name.endswith(".warc.wet")
        if isinstance(self.folder, str):
            return urllib.parse.urljoin(
                self.folder, file_name.replace(".warc.wet", ".json")
            )
        meta_file = self.folder / file_name.replace(".warc.wet", ".json")
        assert (
            meta_file.exists()
        ), f"Couldn't find metadata file for segment {segment} at {meta_file}"
        return str(meta_file)

    def fetch_metadata(self, segment: str) -> None:
        meta_file = self.meta_file(segment)
        k = get_doc_key
        self.metadata = {}
        collision = 0
        for m in jsonql.read_jsons(meta_file):
            key = k(m["digest"])
            if key in self.metadata:
                collision += 1
            self.metadata[key] = m

        self.log(f"Loaded {len(self.metadata)} metadatas from {meta_file}")
        if collision > 0:
            self._logger.warning(f"Found {collision} collisions !")

        self.segment = segment
        if segment in self._segments:
            self.log("Cache miss")
            self.segments_read_twice += 1
        self._segments.add(segment)

    def do(self, doc: dict) -> Optional[dict]:
        if self.segment != doc["cc_segment"]:
            self.fetch_metadata(doc["cc_segment"])
        digest = doc["digest"]
        key = get_doc_key(digest)
        if key not in self.metadata:
            return None

        metadata = self.metadata.pop(key)
        return self.clean(metadata, doc)

    def clean(self, metadata: dict, full_doc: dict) -> Optional[dict]:
        line_ids = decode_line_ids(metadata.pop("line_ids"))
        lines = full_doc["raw_content"].split("\n")
        cleaned = []
        for l in line_ids:
            if l >= len(lines) or l < 0:
                self.missed_par += 1
                continue
            cleaned.append(lines[l])

        self.processed_par += len(line_ids)
        if not cleaned:
            self.missed_doc += 1
            return None

        full_doc["raw_content"] = "\n".join(cleaned)
        full_doc["original_nlines"] = full_doc["nlines"]
        full_doc["original_length"] = full_doc["length"]
        full_doc["nlines"] = len(cleaned)
        full_doc["length"] = len(full_doc["raw_content"])
        for key, value in metadata.items():
            full_doc[key] = value
        return full_doc

    def summary(self) -> List[str]:
        summ = super().summary()
        mem = mem_footprint_gb()
        len_cache = len(self.metadata)
        summ.append(
            f"Read {self.read_doc:_}, stocking {len_cache:_} doc in {mem:.1f}g."
        )
        if self.missed_doc:
            r = self.missed_doc / self.processed
            summ.append(f"! Missed {self.missed_doc} documents ({r:.1%}) !")

        if self.missed_par:
            r = self.missed_par / self.processed
            summ.append(f"! Missed {self.missed_par} paragraphs ({r:.1%}) !")
        return summ


def _expand_files(files: List[Path]) -> List[Path]:
    if len(files) == 1 and files[0].is_dir():
        folder = files[0]
        files = sorted(folder.glob("*.json.gz"))
        print(f"Found {len(files)} files under {folder}/*.json.gz")
    assert files, "No files found"
    return files


def minify_file(file: Path, output: Path) -> str:
    """Minify the given file."""
    jsonql.run_pipes(Minifier(), file=file, output=output)
    return f"Minified {output}"


def minify(
    files: List[Path], output_dir: Path, execution: str = "mp", parallelism: int = -1
):
    """Minify all the files in the given folder."""
    files = _expand_files(files)
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / "files.txt", "w") as o:
        for f in files:
            print(f.name, file=o)
    outputs = [output_dir / f.name for f in files]
    ex = get_executor(
        "minify",
        output_dir / "logs",
        execution,
        timeout_hour=2,
        cpus=1,
        task_parallelism=parallelism,
    )
    ex(minify_file, files, outputs)


def fetch_metadata_file(
    file: Union[Path, str],
    metadata_dir: Union[Path, str],
    output: Path,
    cache_dir: Path = None,
):
    unminifier = MetadataFetcher(metadata_dir)
    tmp = output.with_name("tmp." + output.name)
    jsonql.run_pipes(unminifier, file=file, output=tmp)
    tmp.rename(output)
    return f"Fetched metadata for {file}. Results at {output}."


def fetch_metadata(
    files: List[str],
    metadata_dir: Union[Path, str],
    output_dir: Path,
    execution: str = "mp",
    parallelism: int = -1,
    cache_dir: Path = None,
):
    if len(files) == 1 and Path(files[0]).is_dir():
        folder = Path(files[0])
        files = [str(f) for f in sorted(folder.glob("*.json.gz"))]
        print(f"Found {len(files)} files under {folder}/*.json.gz")

    assert len(files) > 0, "No files given."
    output_dir.mkdir(exist_ok=True)

    outputs = [output_dir / str(f).split("/")[-1] for f in files]
    if cache_dir is None:
        cache_dir = output_dir / "wet_cache"
        cache_dir.mkdir(exist_ok=True)
    if str(cache_dir) == "none":
        cache_dir = None
    files = [f for f, o in zip(files, outputs) if not o.exists()]
    outputs = [o for o in outputs if not o.exists()]
    if not files:
        return
    ex = get_executor(
        "unminify",
        output_dir / "logs",
        execution,
        timeout_hour=8,
        cpus=1,
        task_parallelism=parallelism,
        mem_gb=32,
    )
    ex(fetch_metadata_file, files, outputs, itertools.repeat(cache_dir))


if __name__ == "__main__":
    import func_argparse

    func_argparse.main(minify_file, minify, fetch_metadata, fetch_metadata_file)
