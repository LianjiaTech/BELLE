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

# For each row, run classifier and output
#    (text: [...], source, pred_label, pred_label_prob, wiki_prob)
#
def run(job):

    import fasttext
    model = fasttext.load_model("/nfs/a100-80G-16/pengyiping/data/RedPajama-Data/model.bin")
    
    print(job)
    ofile = gzip.open(job + ".dedup.classifier.gz", "wt")
    ostat = open(job + ".dedup.classifier.gz.stat", "wt")
    line = 0
    for jstr in gzip.open(job + ".result", "rt"):
        result = json.loads(jstr)
        content = result["raw_content"]
        output = {}
        
        # run classifier
        text = " ".join(content.strip().splitlines())
        pred = model.predict(text)
        (pred_label, pred_prob) = pred
        pred_label = pred_label[0]

        wiki_prob = pred_prob[0]
        if pred_label == "__label__cc":
            wiki_prob = 1 - wiki_prob

        output["pred_label"] = pred_label
        output["pred_label_prob"] = pred_prob[0]
        output["wiki_prob"] = wiki_prob

        output["text"] = content
        output["source"] = "cc/" + job + f"/line{line}"
        line = line + 1
        
        nchars = len(content)
        ostat.write(f"{nchars}\t{wiki_prob}\n")
        ofile.write(json.dumps(output) + "\n")
            
    ofile.close()
    ostat.close()

with Pool(224) as p:
    p.map(run, jobs)