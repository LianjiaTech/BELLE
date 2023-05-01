import glob, os
import json
import sys
import re
import hashlib
import gzip
import os

from multiprocessing import Pool, Value
import multiprocessing

import gc

# Get all jobs
#
jobs = []
os.chdir(sys.argv[1])
for file in glob.glob("*.gz"):
    if ("middle" in file or "head" in file) and "dedup" not in file:
        jobs.append(file)

print("TOTAL # JOBS:", len(jobs))

# Load all pairs of (fileid, digest)
#
counter = Value('i', 0)
lock = multiprocessing.Lock()
def load(job):
    load_job = {}

    global counter
    with counter.get_lock():
        counter.value += 1
    print(counter.value, job)

    # test: early stop
    #if counter.value > 10:
    #    return {}

    for line in gzip.open(job + ".dedup", mode='rt'):
        (fileid, digest) = line.split(" ")
        load_job[fileid] = digest
    return load_job

with Pool(64) as p:
    loaded_ = p.map(load, jobs)

loaded = {}
for j in range(0, len(jobs)):
    loaded[jobs[j]] = loaded_[j]

# Dedup
# unique fileIDs are in unique_fileid
# also write unique fileID for each job in its own file
#
table = {}
unique_fileid = {}
#ufile = gzip.open("uniqie_fileids", "wt")
for job in loaded:
    print("loaded", job, len(loaded[job]))
    ufile = gzip.open(job + ".uniqie_fileids", "wt")
    for fileid in loaded[job]:
        digest = loaded[job][fileid]
        if digest not in table:
            table[digest] = 1
            unique_fileid[fileid] = 1
            ufile.write(fileid + "\n")
    ufile.close()
print("total unique", len(unique_fileid))

# GC
#
del loaded_
del loaded
gc.collect()

# Write out the result
#
def write(job):
    
    global counter
    with counter.get_lock():
        counter.value += 1
    print("write", counter.value, job)

    ofile = gzip.open( job + ".result", "wt")
    wrote = 0
    total = 0
    for jstr in gzip.open(job, "rt"): 
        result = json.loads(jstr)
        if result['url'] in unique_fileid:
            wrote = wrote + 1
            ofile.write(jstr)
        total = total + 1
    print("    wrote", wrote, "/", total)
    ofile.close()

with Pool(64) as p:
    p.map(write, jobs)
