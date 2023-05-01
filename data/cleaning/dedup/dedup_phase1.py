import glob, os
import json
import sys
import re
import hashlib
import gzip
import os

from multiprocessing import Pool

# Get all jobs.
# Each job corresponds to a file ends with .gz, with middle or head in it
#
jobs = []
os.chdir(sys.argv[1])
for file in glob.glob("*.gz"):
    if ("middle" in file or "head" in file) and "dedup" not in file:
        jobs.append(file)

print("TOTAL # JOBS:", len(jobs))


# Output (URL, digest) pairs for each job
#
def run(job):
    print(job)
    ofile = gzip.open( job + ".dedup", "wt")
    for jstr in gzip.open(job, "rt"):
        result = json.loads(jstr)
        ofile.write(result['url'] + " " + result['digest'] + "\n")
    ofile.close()

with Pool(64) as p:
    p.map(run, jobs)
