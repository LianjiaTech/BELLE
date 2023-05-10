# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Part of the code was adopted from https://github.com/microsoft/Megatron-DeepSpeed/blob/main/megatron/data/dataset_utils.py
"""
import torch
from torch.utils.data import Dataset, Subset, ConcatDataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from datasets import load_dataset
import numpy as np
import os
import copy
from collections import defaultdict, Counter
from tqdm import tqdm
from itertools import chain
from . import raw_datasets

IGNORE_INDEX = -100

def get_raw_dataset(dataset_name, eval_data_file, output_path, seed, local_rank):
    if type(dataset_name)==list:
        dataset_name = dataset_name[0]
    print("dataset_name : ", dataset_name)
    return raw_datasets.BelleOpenSoucreDataset(output_path, seed, local_rank, data_file=dataset_name, eval_data_file=eval_data_file)



def get_shuffle_idx(seed, size):
    np_rng = np.random.RandomState(seed=seed)
    dtype_ = np.uint32
    if size >= (np.iinfo(np.uint32).max - 1):
        dtype_ = np.int64
    shuffle_idx = np.arange(start=0, stop=size, step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx)
    return shuffle_idx


def get_raw_dataset_split_index(local_rank, output_path, dataset_name, seed,
                                split_name, data_split, split_index,
                                data_size):
    index_file_name = f"{output_path}/{dataset_name}_seed{seed}_{split_name}_{data_split}_{split_index}.npy"
    if not os.path.isfile(index_file_name) and local_rank <= 0:
        splits = [float(s) for s in data_split.split(',')]
        splits_sum = sum(splits)
        splits = [split / splits_sum for split in splits]
        splits_index = [0]
        for index, split in enumerate(splits):
            splits_index.append(splits_index[index] +
                                int(round(split * float(data_size))))
        diff = splits_index[-1] - data_size
        for index in range(1, len(splits_index)):
            splits_index[index] -= diff
        assert splits_index[-1] == data_size

        shuffle_idx = get_shuffle_idx(seed, data_size)
        for split_i in range(len(splits)):
            shuffle_idx_split_file_name = f"{output_path}/{dataset_name}_seed{seed}_{split_name}_{data_split}_{split_i}.npy"
            shuffle_idx_split = shuffle_idx[
                splits_index[split_i]:splits_index[split_i + 1]]
            np.save(shuffle_idx_split_file_name,
                    shuffle_idx_split,
                    allow_pickle=True)
    torch.distributed.barrier()
    index = np.load(index_file_name, allow_pickle=True)
    return index.tolist()


class PromptDataset(Dataset):

    def __init__(self, prompt_dataset, chosen_dataset, reject_dataset,
                 pad_token_id, train_phase) -> None:
        super().__init__()
        self.prompt_dataset = prompt_dataset
        self.chosen_dataset = chosen_dataset
        self.reject_dataset = reject_dataset
        self.pad_token_id = pad_token_id
        self.train_phase = train_phase

    def __len__(self):
        length = len(self.chosen_dataset)
        if self.train_phase == 3:
            length = len(self.prompt_dataset)
        return length

    def __getitem__(self, idx):
        if self.train_phase == 1:
            return {
                "input_ids": self.chosen_dataset[idx]["input_ids"],
                "attention_mask": self.chosen_dataset[idx]["attention_mask"],
                "labels": self.chosen_dataset[idx]["labels"]
            }
        elif self.train_phase == 2:
            return self.chosen_dataset[idx]["input_ids"], self.chosen_dataset[idx]["attention_mask"], \
                self.reject_dataset[idx]["input_ids"], self.reject_dataset[idx]["attention_mask"]
        elif self.train_phase == 3:
            return self.prompt_dataset[idx]["input_ids"],self.prompt_dataset[idx]["attention_mask"], \
                self.pad_token_id


def pad_tensors_to_max_length(input_tensor, max_length, pad_token_id):
    padded_tensor = pad_token_id * torch.ones((max_length,), dtype=input_tensor.dtype, device=input_tensor.device)
    padded_tensor[-input_tensor.shape[0]:] = input_tensor
    return padded_tensor

def create_dataset_split(current_dataset, raw_dataset, train_phase, tokenizer,
                         end_of_conversation_token, max_seq_len):

    def _addrole_masklabel_tokenize(source):
        '''
        add speaker and concatenate the sentences
        {
            "id": "uniq_sample_id",
            "conversations": [
                {"from": "human", "value": "你好"},
                {"from": "assistant", "value": "你好，有什么可以帮助你的吗？"},
                {"from": "human", "value": "今天天气怎么样？"},
                {"from": "assistant", "value": "不好意思，我无法回答你的问题，因为我不知道你的位置信息，同时我目前还无法获取到最新的天气信息。"}
            ]
        }
        tokenizer_bloomz.encode("你好，有什么可以帮助你的吗？") == [41381, 355, 37242, 205599, 7336, 10468]
        tokenizer_llama.encode("你好，有什么可以帮助你的吗？") == [1, 29871, 30919, 31076, 30214, 30417, 231, 190, 131, 31882, 30682, 30651, 232, 187, 177, 31931, 30919, 30210, 232, 147, 154, 30882]
        '''

        conversation = ''
        input_ids = []
        labels = []
        for sentence in source:
            sentence_from = sentence["from"].lower()
            sentence_value = 'Human: \n' + sentence["value"] + '\n\nAssistant: \n' if sentence_from == 'human' else sentence["value"] #https://github.com/LianjiaTech/BELLE/issues/337
            conversation += sentence_value
            sentence_ids = tokenizer.encode(sentence_value, add_special_tokens=False)#do not add bos_token_id
            label = copy.deepcopy(sentence_ids) if sentence_from != 'human' else [IGNORE_INDEX] * len(sentence_ids)
            input_ids += sentence_ids
            labels += label
            # add eos at every end of assistant sentence
            if sentence_from != 'human':
                input_ids += [tokenizer.eos_token_id]#make sure eos_token_id is correct
                labels += [tokenizer.eos_token_id]
        return input_ids, labels, conversation


    prompt_dataset = []
    chosen_dataset = []
    reject_dataset = []
    filter_nums = 0
    assert tokenizer.padding_side == "left"#We need add eos_token_id at the last position of input_ids
    print("tokenizer.eos_token_id: ", tokenizer.eos_token_id)
    print("tokenizer.pad_token_id: ", tokenizer.pad_token_id)
    total_num = len(current_dataset)

    if train_phase == 1:
        for i, tmp_data in tqdm(enumerate(current_dataset), total=total_num, unit="example"):
            # tokenize the text
            source = raw_dataset.get_conversations(tmp_data)
            input_ids, labels, conversation = _addrole_masklabel_tokenize(source)
            input_ids = input_ids[:max_seq_len-1]
            labels = labels[:max_seq_len-1]
            if not any(x > -100 for x in labels):
                #All label value is -100, means that no Human inputs
                filter_nums += 1
                # print("conversation: ", source)
                continue

            attention_mask = [1] * len(input_ids)
            input_ids = torch.LongTensor(input_ids)
            attention_mask = torch.LongTensor(attention_mask)
            labels = torch.LongTensor(labels)

            chosen_token = {
                "input_ids": pad_tensors_to_max_length(input_ids, max_seq_len, tokenizer.pad_token_id),
                "attention_mask": pad_tensors_to_max_length(attention_mask, max_seq_len, tokenizer.pad_token_id),
                "labels": pad_tensors_to_max_length(labels, max_seq_len, IGNORE_INDEX)
            }
            chosen_dataset.append(chosen_token)
            if i == 0:
                # print("conversation: ", conversation)
                print("input_ids: ", input_ids)
                print("labels: ", labels)
                print("-"*100)

        print("{} samples were filtered".format(filter_nums))
        print("The total number of samples: {}".format(len(chosen_dataset)))

    else:
        raise ValueError("Only supported SFT")

    return PromptDataset(prompt_dataset, chosen_dataset, reject_dataset,
                         tokenizer.pad_token_id, train_phase)


def create_dataset(local_rank, dataset_name, eval_data_file, data_split, output_path,
                   train_phase, seed, tokenizer, end_of_conversation_token,
                   max_seq_len):
    #dataset_name can be the file path
    print("dataset_name: ", dataset_name)
    raw_dataset = get_raw_dataset(dataset_name, eval_data_file, output_path, seed, local_rank)
    train_dataset = raw_dataset.get_train_data()
    print("length of train_dataset(after get_train_data): ", len(train_dataset))
    train_index = get_raw_dataset_split_index(local_rank, output_path,
                                              raw_dataset.dataset_name_clean,
                                              seed, "train", data_split,
                                              train_phase - 1,
                                              len(train_dataset))
    train_dataset = Subset(train_dataset, train_index)
    train_dataset = create_dataset_split(train_dataset, raw_dataset,
                                         train_phase, tokenizer,
                                         end_of_conversation_token,
                                         max_seq_len)
    print("lenght of train_dataset", len(train_dataset))
    eval_dataset = raw_dataset.get_eval_data()
    eval_index = get_raw_dataset_split_index(local_rank, output_path,
                                             raw_dataset.dataset_name_clean,
                                             seed, "eval",
                                             data_split, train_phase - 1,
                                             len(eval_dataset))
    eval_dataset = Subset(eval_dataset, eval_index)
    eval_dataset = create_dataset_split(eval_dataset, raw_dataset, train_phase,
                                        tokenizer, end_of_conversation_token,
                                        max_seq_len)
    
    # for item in train_dataset:
    #     print(item)
    return train_dataset, eval_dataset



def create_prompt_dataset(local_rank,
                          sft_only_data_path,
                          eval_data_file,
                          data_split,
                          output_path,
                          train_phase,
                          seed,
                          tokenizer,
                          max_seq_len,
                          end_of_conversation_token="</s>"):
    """
    Creates the prompt dataset
    """
    os.makedirs(output_path, exist_ok=True)
    
    train_dataset, eval_dataset = create_dataset(
        local_rank = local_rank, 
        dataset_name = sft_only_data_path, 
        eval_data_file = eval_data_file,
        data_split = data_split, 
        output_path = output_path, 
        train_phase = train_phase,
        seed = seed, 
        tokenizer = tokenizer, 
        end_of_conversation_token = end_of_conversation_token, 
        max_seq_len = max_seq_len
    )
    # if local_rank <= 0:
    #     torch.save(train_dataset, train_fname)
    #     torch.save(eval_dataset, eval_fname)
    return train_dataset, eval_dataset
    

class DataCollatorReward:

    def __call__(self, data):
        batch = {}
        batch["input_ids"] = torch.cat([f[0]
                                        for f in data] + [f[2] for f in data],
                                       dim=0)
        batch["attention_mask"] = torch.cat([f[1] for f in data] +
                                            [f[3] for f in data],
                                            dim=0)
        return batch


class DataCollatorRLHF:

    def __init__(self, max_token_len, inference_tp_size):
        self.max_token_len = max_token_len
        self.inference_tp_size = inference_tp_size

    def __call__(self, data):
        batch = {}
        pad_token_id = data[-1][-1]

        prompt = pad_sequence([f[0] for f in data],
                              padding_value=pad_token_id,
                              batch_first=True)
        prompt_mask = pad_sequence([f[1] for f in data],
                                   padding_value=0,
                                   batch_first=True)

        ### make sure the final ouput is a seqence of 2**?
        length = prompt.size()[-1]
        pad_length = self.max_token_len - length
        if pad_length > 0:
            batch["prompt"] = F.pad(prompt,
                                    pad=(pad_length, 0),
                                    mode='constant',
                                    value=pad_token_id)
            batch["prompt_att_mask"] = F.pad(prompt_mask,
                                             pad=(pad_length, 0),
                                             mode='constant',
                                             value=0)
        else:
            batch["prompt"] = prompt
            batch["prompt_att_mask"] = prompt_mask
        batch["prompt"] = batch["prompt"].flip(1)
        batch["prompt_att_mask"] = batch["prompt_att_mask"].flip(1)
        return batch


def get_unsupervised_data(args, tokenizer):
    unsupervised_raw_datasets = load_dataset(
        args.unsupervised_dataset_name, args.unsupervised_dataset_config_name)
    column_names = unsupervised_raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = unsupervised_raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )

    block_size = args.max_prompt_seq_len + args.max_answer_seq_len

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {
            k: list(chain(*examples[k]))
            for k in examples.keys()
        }
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k:
            [t[i:i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    train_dataset = lm_datasets["train"]

    return train_dataset


class MiniDataset:

    def __init__(self, max_size, small_batch_size):
        self.dataset = []
        self.max_size = max_size
        self.small_batch_size = small_batch_size

    def seperate(self):
        small_dataset = []
        for large_batch in self.dataset:
            if type(large_batch) == list or type(large_batch) == tuple:
                large_size = len(large_batch[0])
            elif type(large_batch) == dict:
                large_size = len(large_batch[list(large_batch.keys())[0]])
            else:
                large_size = len(large_batch)
            for i in range(0, large_size, self.small_batch_size):
                if type(large_batch) == list or type(large_batch) == tuple:
                    small_dataset.append(
                        [x[i:i + self.small_batch_size] for x in large_batch])
                elif type(large_batch) == dict:
                    small_dataset.append({
                        k: v[i:i + self.small_batch_size]
                        for k, v in large_batch.items()
                    })
                else:
                    small_dataset.append(large_batch[i:i +
                                                     self.small_batch_size])
        self.free()

        return small_dataset

    def add(self, data):
        if len(self.dataset) < self.max_size:
            self.dataset.append(data)
            if len(self.dataset) == self.max_size:
                return self.seperate()
            else:
                return None
        else:
            raise ValueError(
                "The dataset is full but we did not stop it. There is a bug in the code."
            )

    def free(self):
        self.dataset = []
