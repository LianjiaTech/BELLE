# Data Cleaning Pipeline
Following [RedPajama-Data](https://github.com/togethercomputer/RedPajama-Data),
BELLE incorporates [cc_net](https://github.com/facebookresearch/cc_net) with
extra deduplication and quality filter to select high quality Chinese pretraining
data from wild web crawls and other datasets.


## Prerequisites

### cc_net
`data/cleaning/cc_net` contains the entire repo of [cc_net](https://github.com/facebookresearch/cc_net), plus tweaks including directly search directory for files and longer timeout.

Refer to the official installation guide of [cc_net](https://github.com/facebookresearch/cc_net).
Need `boost` installed and can be found by `cmake`. On macOS run `brew install boost cmake`.

```
cd data/cleaning/cc_net

# install python dependencies
pip install -r requirements

# install Language ID model, KenLM
# Note: SentencePiece is only required if you train an LM ground-up.
make bin/lid.bin ./bin/lmplz

# install per-language LM for Chinese
make lang=zh dl_lm
```

### Chinese Wikipedia Reference Quality Classifier
[RedPajama](https://github.com/togethercomputer/RedPajama-Data/tree/main/data_prep/cc#quality-classifier) uses Wikipedia reference (like LLaMA) to train a quality classifier.

We provide a pretrained model trained on references from [zhwiki 20230401 dump](https://dumps.wikimedia.org/zhwiki/20230401/) [here](LINK NEEDED!!!). Download and place it somewhere.

### FastText
Download and build [fasttext](https://fasttext.cc/) and place it somewhere in your $PATH.


## Data Preparation (work-in-progress)
`cc_net` accepts `warc` format. Convert your dataset to `.warc.gz` or `.warc.wet.gz`
before starting the pipeline.

BELLE provides recipes to convert several open datasets in `data/cleaning/recipes` directory.

### Liwu MNBVC
[MNBVC](https://github.com/esbatmop/MNBVC) is a super collection of vast Chinese corpus.
Download and extract `MNBVC` into a directory, e.g. /data/MNBVC

```
# find all .txt files
mkdir liwu
find /data/MNBVC -type f -name "*.txt" > liwu/liwu_txt.list

# shuffle and split into sub-lists
shuf liwu/liwu_txt.list > liwu/liwu_txt.list.shuf
mkdir liwu/liwu_txt.list.shuf.split
split -l 2000 -a4 -d liwu/liwu_txt.list.shuf liwu/liwu_txt.list.shuf.split/split_

# convert into .wet format
mkdir liwu/output_liwu
ls liwu/liwu_txt.list.shuf.split/* | xargs -P60 -I{} python data/cleaning/recipes/liwu/convert_txt_to_wet.py {}

# output .wet files saved into liwu/output_liwu
```

### OSCAR 2301
[OSCAR 2301](https://huggingface.co/datasets/oscar-corpus/OSCAR-2301) is `cc` cleaned and
split into different languages. To use the Chinese subset, clone the `OSCAR 2301` repo:
```
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/oscar-corpus/OSCAR-2301
cd OSCAR-2301
git lfs pull --include "zh_meta/*"
```

and unpack into `json` for readability (maybe skipped and directly into .wet files in the future):
```
mkdir unpack
PYTHONPATH=$PWD python ../data/cleaning/recipes/oscar_2301/unpack_oscar.py
```

then convert into `.wet` file:
```
cd ..
mkdir OSCAR-2301/output_oscar
ls OSCAR-2301/unpack/*.jsonl | xargs -P60 -I{} python data/cleaning/recipes/oscar_2301/convert_json_to_wet.py {}

# output .wet files saved into OSCAR-2301/output_oscar
```

### Custom Dataset
See `data/cleaning/recipes/liwu/convert_txt_to_wet.py` and `data/cleaning/recipes/oscar_2301/convert_json_to_wet.py` for example about how to write `.wet` files.


## Run the Pipeline

### cc_net
Assuming your `.wet` files is located in `/path/to/your/dataset/output_dataset`:

```
cd data/cleaning/cc_net

PYTHONPATH=$PWD python -u cc_net/__main__.py --dump output_dataset --task_parallelism 90 --num_shards 1 --mine_num_processes 1 --hash_in_mem 1 --cache_dir /path/to/your/dataset
```
output files stored in `data/cleaning/cc_net/data/mined/output_dataset`

### Deduplication
```
python data/cleaning/dedup/dedup_phase1.py data/cleaning/cc_net/data/mined/output_dataset
python data/cleaning/dedup/dedup_phase2.py data/cleaning/cc_net/data/mined/output_dataset
```

### Quality Filter
```
python data_prep/cc/classifier/classify.py data/cleaning/cc_net/data/mined/output_dataset
for file in $(ls data/cleaning/cc_net/data/mined/output_dataset/*.gz.dedup.classifier.gz); do bash filter/cc_classifier.sh "$file"; done
```
