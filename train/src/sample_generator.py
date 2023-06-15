import pudb
import copy
from transformers import PreTrainedTokenizer
import json
IGNORE_INDEX = -100


def generate_and_tokenize_prompt(model_max_length: int, tokenizer: PreTrainedTokenizer, data_point):
    input_ids = []
    labels = []
    source = data_point["conversations"]
    for sentence in source:
        sentence_from = sentence["from"].lower()
        sentence_value = (
            "Human: \n" + sentence["value"] + "\n\nAssistant: \n"
            if sentence_from == "human"
            else sentence["value"]
        )  # https://github.com/LianjiaTech/BELLE/issues/337
        # conversation += sentence_value
        sentence_ids = tokenizer.encode(
            sentence_value, add_special_tokens=False
        )  # do not add bos_token_id
        label = (
            copy.deepcopy(sentence_ids)
            if sentence_from != "human"
            else [IGNORE_INDEX] * len(sentence_ids)
        )
        input_ids += sentence_ids
        labels += label
        # add eos at every end of assistant sentence
        if sentence_from != "human":
            input_ids += [
                tokenizer.eos_token_id
            ]  # make sure eos_token_id is correct
            labels += [tokenizer.eos_token_id]

    input_ids = input_ids[: model_max_length - 1]
    labels = labels[: model_max_length - 1]
    if all(x == IGNORE_INDEX for x in labels):
        labels[18:24] = input_ids[
            18:24
        ]  # labels can not have all values being -100. 18 and 24 are just random numbers

    attention_mask = [1] * len(input_ids)
    tokenized_full_prompt = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
    return tokenized_full_prompt


def pretrain_generate(model_max_length: int, tokenizer: PreTrainedTokenizer, data_point):
    input_ids = tokenizer.encode(data_point['text'])
    labels = copy.deepcopy(input_ids)
    input_ids += [tokenizer.eos_token_id]
    labels += [tokenizer.eos_token_id]
    input_ids = input_ids[: model_max_length]
    labels = labels[: model_max_length]
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def exam_generate(model_max_length: int, tokenizer: PreTrainedTokenizer, data_point):
    template = 'Human: \n{human}\n\nAssistant: \n'
    # pudb.set_trace()
    input_str = template.format(
        human=f'回答下面的{data_point["type"]}题，用json返回答案，包括原因和答案，如{{"reason":..., "answer":...}}\n{data_point["question"]}\n选项：{" ".join(data_point["candidates"])}'
    )
    input_ids = tokenizer.encode(
        input_str,
        add_special_tokens=False
    )
    labels = [IGNORE_INDEX] * len(input_ids)
    bot_ids = tokenizer.encode(
        json.dumps(
            {
                'reason': data_point['reason'],
                'answer': data_point['answer']
            }, ensure_ascii=False
        ),
        add_special_tokens=False
    )
    input_ids += bot_ids
    labels += bot_ids

    input_ids += [tokenizer.eos_token_id]
    labels += [tokenizer.eos_token_id]

    input_ids = input_ids[: model_max_length - 1]
    labels = labels[: model_max_length - 1]
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }
