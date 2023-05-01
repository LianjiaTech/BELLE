# Makefile to install CC-Net and train the LMs.
# `make` or `make help` to get some help.

# Arguments:
lang?=en
process?=8
servers?=0

# List of languages for LM.
langs?=af,ar,az,be,bg,bn,ca,cs,da,de,el,en,es,et,fa,fi,fr,gu,he,hi,hr,hu,hy,id,\
is,it,ja,ka,kk,km,kn,ko,lt,lv,mk,ml,mn,mr,my,ne,nl,no,pl,pt,ro,ru,uk,zh

# Experiment config
NDOC_FOR_LM=1_000_000
NDOC_FOR_SENTPIECE=400000
VOCAB_SIZE=65536

# Static resources, scripts, ...
KENLM=./bin/lmplz
KENLM_BUILD_BINARY=./bin/build_binary
SPM_TRAIN=./bin/spm_train
SPM_ENCODE=./bin/spm_encode

# DISTRIBUTE will run locally, or on slurm if "servers" is set.
DISTRIBUTE=xargs -L1 -P $(process)
ifneq ($(servers), 0)
	DISTRIBUTE=xargs -L1 -P $(servers) srun -t 240 --mem 5000
endif

# PRIVATE
_SEGMENT=2019-09/CC-MAIN-20190215183319-20190215205319-00000

help:
	# Show help
	grep -i -A1 '^[a-z0-9_]*:' Makefile

# Deletes output files on error (useful when piping output)
SHELL=/bin/bash
.SHELLFLAGS = -o pipefail -c
.DELETE_ON_ERROR:

install: bin/lid.bin $(KENLM) $(SPM_TRAIN)
	# Installs dependencies.
	@if [ -f "data" ]; then\
		echo "Please create/simlink a 'data' directory.";\
	fi
	@if ! python -c "from cc_net import __main__" 2> /dev/null; then\
		pip install . ;\
	fi
	echo " --> All dependencies looks good !"

dl_lm:
	# Download a pretrained language model
	mkdir -p data/lm_sp
	wget -c  -P data/lm_sp http://dl.fbaipublicfiles.com/cc_net/lm/$(lang).arpa.bin
	wget -c  -P data/lm_sp http://dl.fbaipublicfiles.com/cc_net/lm/$(lang).sp.model

lm: data/lm_sp/$(lang).sp.model data/lm_sp/$(lang).arpa.bin
	# Computes a 5-gram LM for the given language -> make lang=it lm
	# Restricted to the first NDOC_FOR_LM documents

sp: data/lm_sp/$(lang).sp.model
	# Train a sentence piece model on Wikipedia -> make lang=it sp

get_lang = $(firstword $(subst ., ,$1))

all_lms:
	# Train a list of language models -> make process=10 langs=en,it,fr all_lms
	# Defaults to the LMs trained in the paper.
	echo "$(langs)" \
		| tr -s ', ' '\n' | awk '{print "data/lm_sp/" $$0 ".arpa.bin"}' \
		| $(DISTRIBUTE) make

dl_all_lms:
	# Download all pretrained language models
	echo "$(langs)" \
		| tr -s ', ' '\n' | awk '{print "lang=" $$0 " dl_lm"}' \
		| $(DISTRIBUTE) make

%.arpa.bin: %.arpa
	# Compress a learned LM to a binary format.
	$(KENLM_BUILD_BINARY) $< $@

%.vocab.txt: %.txt
	# Extracts the vocabulary of a corpus.
	# Restricted to the first NDOC_FOR_LM documents and VOCAB_SIZE top words.
	cat $< | tr ' ' '\n' | sort | uniq -c | sort -rn > $@.tmp_sort
	head -$(VOCAB_SIZE) $@.tmp_sort | sed "s/^ *//" | cut -d' ' -f2 > $@
	rm $@.tmp*
	echo Extracted `wc -l $@` words

data/lm_sp/%.arpa: data/cirrus/sp/%.opening.txt
	mkdir -p $(@D)
	$(KENLM) -o 5 -S 8G -T /tmp --vocab_estimate $(VOCAB_SIZE)  --discount_fallback \
        < $< > $@

data/lm_sp/%.sp.model: data/cirrus/txt/%.opening.txt
	mkdir -p $(@D)
	$(SPM_TRAIN) --input=$< \
		--vocab_size=$(VOCAB_SIZE) --hard_vocab_limit \
		--character_coverage=0.9995 \
		--model_type=unigram \
		--model_prefix=$(basename $@) \
	|| echo "WARNING: Corpus is too small, will train smaller model" && \
	$(SPM_TRAIN) --input=$< \
		--vocab_size=40000 \
		--character_coverage=0.9995 \
		--model_type=unigram \
		--model_prefix=$(basename $@)

	echo "Trained SentencePiece model with `wc -l $(basename $@).vocab` pieces"

data/cirrus/sp/%.opening.txt: data/cirrus/gz/%.json.gz data/lm_sp/%.sp.model
	$(SPM_ENCODE) \
		--model=$(word 2,$^) \
		--output_format=piece \
			< <(python get_wiki_cirrus.py opening --file $< --n_docs $(NDOC_FOR_LM)) \
			> $@

data/cirrus/txt/%.opening.txt: data/cirrus/gz/%.json.gz
	python get_wiki_cirrus.py opening \
		--n_docs $(NDOC_FOR_LM) \
		--file $< --output $@

data/cirrus/gz/%.json.gz:
	mkdir $(@D)
	python get_wiki_cirrus.py dl --lang $(call get_lang,$(@F)) --output_dir $(@D)

clean:
	# Remove intemediary files, dataset, third_party sources
	# We don't need the vocab files nor the text version of the LM.
	rm -r data/cirrus
	rm -r data/lm_sp/*.arpa data/lm_sp/*.vocab
	rm -r third_party

# Installation
bin/lid.bin:
	# DL languages id from Fasttext releases.
	mkdir -p $(@D)
	wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin -O $@

third_party/kenlm:
	# Download kenlm sources: https://kheafield.com/code/kenlm/"
	mkdir -p $(@D)
	wget -O - https://kheafield.com/code/kenlm.tar.gz | tar -xz -C $(@D)

bin/lmplz: third_party/kenlm
	# Compiles kenlm binaries
	mkdir -p $(@D)
	mkdir -p $</build
	(cd $</build && cmake ..)
	make -C $</build -j2
	mv $</build/bin/lmplz $</build/bin/build_binary $(@D)

third_party/sentencepiece:
	# Download sentencepiece sources: https://github.com/google/sentencepiece
	mkdir -p $(@D)
	wget -c -O $(@D)/sentencepiece.zip https://github.com/google/sentencepiece/archive/v0.1.83.zip
	unzip -o -d $(@D) $(@D)/sentencepiece.zip
	rm $(@D)/sentencepiece.zip
	# remove the version id from the folder name
	mv $(@D)/sentencepiece-* $@

bin/spm_train: third_party/sentencepiece
	# Compiles sentencepiece binaries
	mkdir -p $(@D)
	mkdir -p $</build
	(cd $</build && cmake ..)
	make -C $</build -j2
	mv $</build/src/spm_train $</build/src/spm_encode $(@D)
	# Installed SentencePiece locally to install globally do:
	# $ cd $</build
	# $ sudo make install
	# $ sudo ldconfig -v

test:
	python -m cc_net mine --config test
	mkdir -p test_data/mini
	python -m cc_net.minify minify -f test_data/mined/2019-09 -o test_data/mini/2019-09
	mkdir -p test_data/reproduce
	python cc_net/minify.py unminify -f test_data/mini/2019-09 -o test_data/reproduce/2019-09
	diff \
		<(zcat test_data/mined/2019-09/de_head_0000.json.gz | sort | jq -r .raw_content) \
		<(zcat test_data/reproduce/2019-09/de_head_0000.json.gz | sort | jq -r .raw_content)

test2:
	python -m cc_net --config config/test_segment.json
	python -m cc_net --config config/test_reproduce.json
	diff \
		<(zcat test_data/mined/2019-09/fr_head_0000.json.gz | jq -c 'select(.cc_segment == "crawl-data/CC-MAIN-2019-09/segments/1550247479101.30/wet/CC-MAIN-20190215183319-20190215205319-00000.warc.wet.gz") | {url, perplexity}' | sort) \
		<(zcat test_data2/mined_by_segment/2019-09/CC-MAIN-20190215183319-20190215205319-00000.json.gz | jq -c 'select(.bucket == "head" and .language == "fr") | {url, perplexity}' | sort) \
		| head

	diff \
		<(zcat test_data/mined/2019-09/fr_head_0000.json.gz | sort | jq -r .raw_content ) \
		<(zcat test_data2/reproduce/2019-09/fr_head_0000.json.gz | sort | jq -r .raw_content ) \
		| head

test_data/regroup_tr/$(_SEGMENT).json.gz:
	mkdir -p test_data/transpose
	python cc_net/transpose.py transpose -f test_data/mined/2019-09 -o test_data/transpose/2019-09 \
		--ex debug
	mkdir -p test_data/regroup_tr
	python cc_net/transpose.py regroup_tr -i test_data/transpose/2019-09 -o test_data/regroup_tr/2019-09 \
		--ex local --conf test
	mkdir -p test_data/reproduce_tr
	python cc_net/transpose.py unminify -f test_data/regroup_tr/2019-09 -o test_data/reproduce_tr/2019-09 \
		--ex debug --conf test

test_transpose: test_data/regroup_tr/$(_SEGMENT).json.gz
	diff -y -W60 \
		<(zcat test_data/mined/2019-09/*.json.gz | jq -r .language | sort | uniq -c ) \
		<(zcat test_data/reproduce_tr/2019-09/*.json.gz | jq -r .language | sort | uniq -c )
	diff -y -w60 \
		<(zcat test_data/mined/2019-09/*.json.gz | jq -r .raw_content | wc) \
		<(zcat test_data/reproduce_tr/2019-09/*.json.gz | jq -r .raw_content | wc)
	diff \
		<(zcat test_data/mined/2019-09/*.json.gz | jq -r .url | sort) \
		<(zcat test_data/reproduce_tr/2019-09/*.json.gz | jq -r .url | sort) \
		| head
	# zcat test_data/reproduce_tr/2019-09/*.json.gz | sort | head -2 | jq -r .raw_content

test_with_metadata: test_data/regroup_tr/$(_SEGMENT).json.gz
	python -m cc_net mine --config test --metadata test_data/regroup_tr

	diff -y -W60 \
		<(zcat test_data/mined/2019-09/*.json.gz | jq -r .language | sort | uniq -c ) \
		<(zcat test_data/reproduce/2019-09/*.json.gz | jq -r .language | sort | uniq -c )
	diff -y -w60 \
		<(zcat test_data/mined/2019-09/*.json.gz | jq -r .raw_content | wc) \
		<(zcat test_data/reproduce/2019-09/*.json.gz | jq -r .raw_content | wc)
	diff \
		<(zcat test_data/mined/2019-09/*.json.gz | jq -r .url | sort) \
		<(zcat test_data/reproduce/2019-09/*.json.gz | jq -r .url | sort) \
		| head

