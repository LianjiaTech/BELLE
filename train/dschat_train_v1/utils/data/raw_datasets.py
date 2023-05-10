# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
from datasets import disable_caching
disable_caching()
from datasets import load_dataset
from torch.utils.data import Subset
import re
import os


# The template prompt dataset class that all new dataset porting needs to
# follow in order to have a unified API and unified data format.
class PromptRawDataset(object):

    def __init__(self, output_path, seed, local_rank):
        self.output_path = output_path
        self.seed = seed
        self.local_rank = local_rank

    def get_train_data(self):
        return

    def get_eval_data(self):
        return

    # The prompt should be in the format of: " Human: " + actual_prompt_sentence + " Assistant:"
    def get_prompt(self, sample):
        return

    # The chosen response should be in the format of: " " + actual_response_sentence
    def get_chosen(self, sample):
        return

    # The rejected response should be in the format of: " " + actual_response_sentence
    # If the dataset does not have rejected response, return None
    def get_rejected(self, sample):
        return

    def get_prompt_and_chosen(self, sample):
        return

    def get_prompt_and_rejected(self, sample):
        return


# English dataset. https://huggingface.co/datasets/Dahoas/rm-static
class DahoasRmstaticDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank):
        super().__init__(output_path, seed, local_rank)
        self.dataset_name = "Dahoas/rm-static"
        self.dataset_name_clean = "Dahoas_rm_static"
        self.raw_datasets = load_dataset("Dahoas/rm-static")

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample['prompt']

    def get_chosen(self, sample):
        return sample['chosen']

    def get_rejected(self, sample):
        return sample['rejected']

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return sample['prompt'] + sample['rejected']


# # Belleschool_math0.25K 
# class BelleOpenSoucreDataset(PromptRawDataset):

#     def __init__(self, output_path, seed, local_rank, data_file):
#         eval_data_file = "utils/data/dev1K.json"
#         super().__init__(output_path, seed, local_rank)
#         self.dataset_name = "BelleOpenSoucre"
#         self.dataset_name_clean = "BelleOpenSoucre"
#         dataset_cache_dir = "output/data_files"
#         print("data_file = ", data_file)
#         self.raw_datasets = load_dataset("json", data_files=data_file, cache_dir=dataset_cache_dir)
#         self.raw_datasets.cleanup_cache_files()
#         self.dev_raw_datasets = load_dataset("json", data_files=eval_data_file, cache_dir=dataset_cache_dir)
#         self.dev_raw_datasets.cleanup_cache_files()
#         print(self.raw_datasets["train"])

#     def get_train_data(self):
#         return self.raw_datasets["train"]

#     def get_eval_data(self):
#         return self.dev_raw_datasets["train"]

#     def get_prompt(self, sample):
#         return "Human: "+sample['instruction']+sample['input']+"\n Assistant: "

#     def get_chosen(self, sample):
#         return "Human: "+sample['instruction']+sample['input']+"\n Assistant: "


#     def get_prompt_and_chosen(self, sample):
#         return "Human: "+sample['instruction']+sample['input']+"\n Assistant: "+sample['output']



class BelleOpenSoucreDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, data_file, eval_data_file=None):
        '''
        {
            "id": "uniq_sample_id",
            "conversations": [
                {"from": "human", "value": "你好"},
                {"from": "assistant", "value": "你好，有什么可以帮助你的吗？"},
                {"from": "human", "value": "今天天气怎么样？"},
                {"from": "assistant", "value": "不好意思，我无法回答你的问题，因为我不知道你的位置信息，同时我目前还无法获取到最新的天气信息。"}
            ]
        }
        LlamaTokenizer会自动加上bos_token_id，但是BloomTokenizer不会加上bos_token_id
        两个tokenizer的bos_token_id和eos_token_id是相同的，pad_token_id强制设置为0
        '''
        # eval_data_file = "utils/data/dev1K.json"

        super().__init__(output_path, seed, local_rank)
        self.dataset_name = "BelleOpenSoucre"
        self.dataset_name_clean = "BelleOpenSoucre"
        dataset_cache_dir = "output/data_files"
        print("data_file = ", data_file)
        self.raw_datasets = load_dataset("json", data_files=data_file, cache_dir=dataset_cache_dir)
        self.raw_datasets.cleanup_cache_files()

        if eval_data_file!=None and os.path.exists(eval_data_file):
            print("eval_data_file = ", eval_data_file)
            self.dev_raw_datasets = load_dataset("json", data_files=eval_data_file, cache_dir=dataset_cache_dir)
            self.dev_raw_datasets.cleanup_cache_files()
            self.train_data = self.raw_datasets["train"]
            self.eval_data = self.dev_raw_datasets["train"]
        else:
            train_val = self.raw_datasets["train"].train_test_split(
                test_size=1000, shuffle=True, seed=42
            )
            self.train_data = train_val["train"]
            self.eval_data = train_val["test"]

        print("train_data: ", self.train_data)
        print("eval_data: ", self.eval_data)


    def get_train_data(self):
        return self.train_data

    def get_eval_data(self):
        return self.eval_data

    def get_conversations(self, sample):
        return sample['conversations']
