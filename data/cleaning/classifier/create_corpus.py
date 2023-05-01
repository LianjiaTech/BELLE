import glob, os
import json
import sys
import re
import hashlib
import gzip
import os


## Load data from the Wikipedia corpus
## And output them as label "__label__wiki"
#
files = ["/nfs/a100-80G-16/pengyiping/data/cc_net/data/mined/output_zhwiki/zh_head_0000.json.gz", "/nfs/a100-80G-16/pengyiping/data/cc_net/data/mined/output_zhwiki/zh_middle_0000.json.gz"]
unique = {}
i = 0
for f in files:
    for jstr in gzip.open(f, "rt"):
        i = i + 1
        result = json.loads(jstr)
        result["class"] = "wiki"

        if result["digest"] in unique:
            continue
        unique["digest"] = 1

        if(len(result["raw_content"]) < 1000):
            continue

        print("__label__wiki " + " ".join(result["raw_content"].splitlines()))

jobs = []
for file in glob.glob("/nfs/a100-80G-16/pengyiping/data/cc_net/data/mined/output/*.gz"):
    if ("middle" in file or "head" in file) and "dedup" not in file:
        jobs.append(file)

## Fetch `perfile` number of webpages for each CommonCrawl partition
#
perfile = i / len(jobs)

## Output Commoncrawl data as label "__label__wiki"
#
n = 0
for job in jobs:
    j = 0
    for jstr in gzip.open(job, "rt"):
        j = j + 1
        if j > perfile:
            break
        result = json.loads(jstr)
        result["class"] = "cc"
        print("__label__cc " + " ".join(result["raw_content"].splitlines()))
        n = n + 1
