# cc_net

Tools to download and clean Common Crawl as introduced in our paper [CCNet](https://arxiv.org/abs/1911.00359).

If you found these resources useful, please consider citing:

```
@inproceedings{wenzek2020ccnet,
  title={CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data},
  author={Wenzek, Guillaume and Lachaux, Marie-Anne and Conneau, Alexis and Chaudhary, Vishrav and Guzm{\'a}n, Francisco and Joulin, Armand and Grave, {\'E}douard},
  booktitle={Proceedings of The 12th Language Resources and Evaluation Conference},
  pages={4003--4012},
  year={2020}
}
```

[![CircleCI](https://circleci.com/gh/facebookresearch/cc_net.svg?style=svg)](https://circleci.com/gh/facebookresearch/cc_net)


## Installation

We only tried this on Linux but installation should be possible on MacOS too.

1. Create or simlink a `data` folder to where you want to download the corpus.

2. Run `make install`. This will download some resources and install required packages.

3. If you have a C++ 17 compiler you can also run
`pip install .[getpy]`, it provides more memory efficient hashset.

4. Install the following tools manually if `make install` failed:
- `lmplz` and `build_binary` from [KenLM](https://github.com/kpu/kenlm)
- `spm_train` and `spm_encode` from [Sentence Piece](https://github.com/google/sentencepiece)

## Training Language Models

The `Makefile` is used to train Sentence Piece and LM on Wikipedia data.

* `make help` shows help
* `make lang=de lm` trains a Sentence Piece and a LM on German Wikipedia
* `make all_lm` trains the same model than in the paper
* `make lang=de dl_lm` downloads the LM trained for the paper
* `make dl_all_lm` downloads all of them

## Pipeline overview

The full mining pipeline is divided in 3 steps:

- `hashes` downloads one Common-Crawl snapshot, and compute hashes for each paragraph
- `mine` removes duplicates, detects language, run the LM and split by lang/perplexity buckets
- `regroup` regroup the files created by `mine` in chunks of 4Gb

Each step needs the previous step to be over before starting.
You can launch the full pipeline using `python -m cc_net`.

* `python -m cc_net --help` shows help
* `python -m cc_net --dump 2019-13` treats a specific snapshot
* `python -m cc_net -l my -l gu` 
restricts to specific languages
* `python -m cc_net --lm_dir my_lms/` uses custom LMs
* `python -m cc_net --lang_threshold 0.3` set a specific field in `mine.Config`
* `python -m cc_net --config test` runs on a tiny subset of a snapshot
* `python -m cc_net --config config/my_config.json` uses configuration from the given config file

## Reproducing our work

Given the CPU required to run the full pipeline on such a big corpus we share a mapping from url to the information we computed.
You can reconstruct the corpus used in the paper by using:

```sh
python -m cc_net --conf reproduce --dump 2019-09
```

## Extract XLM-R data

[Unsupervised Cross-lingual Representation Learning at Scale (XLM-RoBERTa)](https://arxiv.org/pdf/1911.02116.pdf)
paper was trained on data extracted by an internal version of cc_net.

Due to the format being a little bit different please use the following command instead:

```sh
python cc_net/tools/dl_cc_100.py --help
python cc_net/tools/dl_cc_100.py --outdir data_cc100 --process 8
```

If you use this version of the data please also consider citing:

```bibtex
@article{conneau2019unsupervised,
  title={Unsupervised Cross-lingual Representation Learning at Scale},
  author={Conneau, Alexis and Khandelwal, Kartikay and Goyal, Naman and Chaudhary, Vishrav and Wenzek, Guillaume and Guzm{\'a}n, Francisco and Grave, Edouard and Ott, Myle and Zettlemoyer, Luke and Stoyanov, Veselin},
  journal={arXiv preprint arXiv:1911.02116},
  year={2019}
}
```


## Adapting to your infrastructure

Given the computation cost of running the full pipeline we distributed the computation
on a [Slurm](https://slurm.schedmd.com/) cluster using [submitit](https://github.com/facebookincubator/submitit).
`submitit` will default to spawning processes on your machine if Slurm cluster is found.
You should tweak `--task_parallelism` to something adapated to your machine.
Defaults are 512 for mining and 20 for reproducing.

To run the tasks in-process use `--execution debug`.


## Output format

Generated files are compressed JSON files. There is one JSON object per line.

__List of fields__:

- url: webpage URL (part of CC)
- date_download: date of download (part of CC)
- digest: sha1 digest of the webpage (part of CC)
- length: number of chars
- nlines: number of lines
- source_domain: web domain of the webpage
- title: page title (part of CC)
- raw_content: webpage content after deduplication
- original_nlines: number of lines before deduplication
- original_length: number of chars before deduplication
- language: language detected by FastText LID
- language_score: language score
- perplexity: perplexity of a LM trained on Wikipedia

__Sample JSON object__:
```json
{
  "url": "http://www.pikespeakhospice.org/members/1420",
  "date_download": "2019-02-15T18:40:25Z",
  "digest": "sha1:VQW3KXUOALO543IJGTK2JLVEAN2XXKHI",
  "length": 752,
  "nlines": 5,
  "source_domain": "www.pikespeakhospice.org",
  "title": "LeeRoy Aragon",
  "raw_content": "Date Honored: March 2017\nHe was a man of integrity, a hard worker, and a dedicated family man. He loved spending time with family camping, fishing, hunting, boating and just hanging out.\nHis Catholic faith was extremely important to him as he gave of his time and talents to the community. He had many friends through church and the Knights of Columbus. He was a meticulous handyman, and enjoyed building and fixing things and restoring antique furniture to perfection. He was a fan and supported his Colorado Rockies and Denver Broncos. Throughout the years he had devoted four-legged friends (his dogs and a horse named Sunny Boy).\nWe have many cherished memories of him that we will treasure until we are with him again.\n~ Family of LeeRoy F. Aragon",
  "original_nlines": 7,
  "original_length": 754,
  "language": "en",
  "language_score": 0.99,
  "perplexity": 255.11,
}
```

You can peak at those files using UNIX tools `zcat` and [`jq`](https://stedolan.github.io/jq/manual/), eg:
`zcat data/mined/2019-09/en_head_0000.json.gz | head -1 | jq .`

`jq` can do some complicated filtering.
`jsonql.py` provides a Python API with multiprocess support to do more complicated operations like LM scoring of the document.

## License

By contributing to `cc_net`, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
