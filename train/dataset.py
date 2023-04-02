import json

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Union, Optional
from transformers import AutoTokenizer

class TruncateDataset(Dataset):
    """Truncate dataset to certain num"""
    def __init__(self, dataset: Dataset, max_num: int = 100):
        self.dataset = dataset
        self.max_num = min(max_num, len(self.dataset))

    def __len__(self):
        return self.max_num

    def __getitem__(self, item):
        return self.dataset[item]

    def __getattr__(self, item):
        """other dataset func"""
        return getattr(self.dataset, item)

def has_zh(s):
    for c in s:
        if '\u4e00' <= c <= '\u9fa5':
            return True
    return False

def cal_str_length(str):
    words = str.split()
    length = 0
    for word in words:
        if has_zh(word):
            length += len(word)
        else:
            length += 1
    return length

class Tokenizer4GEN():
    def __init__(self, model_name_or_path: str, max_seq_length: int = 512, \
                 max_ans_length: int = 20, padding_side: str = 'left', padding_mode: str = "true", target_only_loss: bool = True):

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)#llama_tokenizer.padding_side==right 
        self.padding_side = padding_side
        self.tokenizer.padding_side = padding_side
        self.target_only_loss = target_only_loss
        self.vocab_size = self.tokenizer.vocab_size
        self.all_special_tokens = self.tokenizer.all_special_tokens
        self.pad_token_id = self.tokenizer.pad_token_id
        self.batch_decode = self.tokenizer.batch_decode
        self.save_pretrained = self.tokenizer.save_pretrained
        assert padding_mode in ['max_length', "true"], "Uncorrect padding mode: {}".format(padding_mode)
        self.padding_mode = True if padding_mode == "true" else "max_length"

        self.max_seq_length = max_seq_length
        self.max_ans_length = max_ans_length
        self.eos_token = self.tokenizer.eos_token if self.tokenizer.eos_token != None else "</s>"
        self.bos_token = self.tokenizer.bos_token


    def __call__(self, queries, answers, sample_ids):
        if self.target_only_loss:
            if self.padding_side == 'left':
                contexts = []
                for i, q in enumerate(queries):
                    contexts.append(q + answers[i])
                answer_features = self.tokenizer(text_target = answers, padding = self.padding_mode, truncation=True, return_tensors="pt", max_length=self.max_seq_length, return_offsets_mapping=False)
                features = self.tokenizer(contexts, padding = self.padding_mode, truncation=True, return_tensors="pt", max_length=self.max_seq_length, return_offsets_mapping=False)

                answer_mask = answer_features['attention_mask']
                q_a_mask = features['attention_mask']
                pad_len = q_a_mask.shape[1] - answer_mask.shape[1]
                pad_mask = answer_mask.new_zeros(answer_mask.shape[0], pad_len)
                label_mask = torch.cat([pad_mask, answer_mask], 1)

                labels = features['input_ids'].detach().clone()
                labels[label_mask==0]=-100
            else:
                contexts = []
                for i, q in enumerate(queries):
                    contexts.append(q + answers[i])
                query_features = self.tokenizer(queries, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_seq_length, return_offsets_mapping=False)
                features = self.tokenizer(contexts, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_seq_length, return_offsets_mapping=False)

                q_mask = query_features['attention_mask']
                q_a_mask = features['attention_mask']
                pad_len = q_a_mask.shape[1] - q_mask.shape[1]
                pad_mask = q_mask.new_zeros(q_mask.shape[0], pad_len)
                target_only_mask = -(torch.cat([q_mask, pad_mask], 1)-1)
                label_mask = q_a_mask * target_only_mask

                labels = features['input_ids'].detach().clone()
                labels[label_mask==0]=-100

        else:
            contexts = []
            for i, q in enumerate(queries):
                contexts.append(q + answers[i])
            
            features = self.tokenizer(contexts, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_seq_length, return_offsets_mapping=False)
            labels = features['input_ids'].detach().clone()
            pad_token_id = self.tokenizer.pad_token_id
            labels[labels==pad_token_id] = -100

        return {
            "sample_ids": sample_ids,
            "whole_input_ids": features['input_ids'],
            "whole_attention_mask": features['attention_mask'],
            "whole_labels": labels,
            "target_only_labels": labels
        }


class GEN4ALLDataset(Dataset):
    def __init__(self,
                 json_path,
                 tokenizer,
                 bos_token=None,
                 eos_token="</s>",
                 max_length = 512,
                 data_tokenized = False,
                 data_size = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.data_tokenized = data_tokenized
        self.data_size = data_size

        if self.data_tokenized:
            self.all_data = self._load_tokenized(json_path)
        else:
            self.all_data = self._load(json_path=json_path)

        print("{}: the number of examples = {}".format(json_path, len(self.all_data)))

    def __len__(self):
        return len(self.all_data)

    def _load(self, json_path):
        data_lines = []
        max_input_length = 0
        max_target_length = 0
        skip_nums = 0
        with open(json_path, mode="r", encoding="utf-8") as reader:
            json_data = []
            for line in reader:
                json_data.append(json.loads(line.strip()))

            for data_part in tqdm(json_data, total=len(json_data), unit="example"):
                try:
                    instruction = data_part['instruction']
                    input_text = data_part["input"]
                    input_text = self.bos_token + instruction + input_text if self.bos_token!=None else instruction + input_text
                    target_text = data_part["output"] + self.eos_token
                    sample_id = data_part.get("id", "placeholder")
                    
                except Exception as e:
                    print(e)
                    continue

                max_input_length = max(max_input_length, cal_str_length(input_text))
                max_target_length = max(max_target_length, cal_str_length(target_text))
                data_lines.append((sample_id, input_text, target_text))

        print("max input length:", max_input_length)
        print("max target length:", max_target_length)
        print (f"Skiped examples: {skip_nums}")
        return data_lines

    def _load_tokenized(self, json_path):
        data_lines = []
        with open(json_path, mode="r") as reader:
            for line in tqdm(reader):
                data = json.loads(line)
                if type(data) != dict:
                    continue
                input_ids = data['input']
                target_ids = data['target']
                id = data['id']
                if len(input_ids) + len(target_ids) > self.max_length - 2:
                    continue
                data_lines.append((id, input_ids, target_ids))
        print (f'Total examples: {len(data_lines)}')
        return data_lines


    def __getitem__(self, item):
        return self.all_data[item]
