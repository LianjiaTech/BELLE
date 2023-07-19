from itertools import chain
from typing import Any, Dict, List
import pudb
import copy
from transformers import PreTrainedTokenizer
import json

IGNORE_INDEX = -100


def generate_and_tokenize_prompt(
    model_max_length: int,
    tokenizer: PreTrainedTokenizer,
    data_point: Dict[str, Any],
    fix_length=False,
    padding_side="left",
):
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
            input_ids += [tokenizer.eos_token_id]  # make sure eos_token_id is correct
            labels += [tokenizer.eos_token_id]

    input_ids = input_ids[:model_max_length]
    labels = labels[:model_max_length]

    if all(x == IGNORE_INDEX for x in labels):
        labels[18:24] = input_ids[
            18:24
        ]  # labels can not have all values being -100. 18 and 24 are just random numbers
    attention_mask = [1] * len(input_ids)

    if fix_length:
        if padding_side == "left":
            input_ids = [tokenizer.pad_token_id] * (
                model_max_length - len(input_ids)
            ) + input_ids
            labels = [tokenizer.pad_token_id] * (
                model_max_length - len(labels)
            ) + labels
            attention_mask = [0] * (
                model_max_length - len(attention_mask)
            ) + attention_mask
        else:
            input_ids = input_ids + [tokenizer.pad_token_id] * (
                model_max_length - len(input_ids)
            )
            labels = labels + [tokenizer.pad_token_id] * (
                model_max_length - len(labels)
            )
            attention_mask = attention_mask + [0] * (
                model_max_length - len(attention_mask)
            )

    tokenized_full_prompt = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
    return tokenized_full_prompt


def batch_grouped_pretrain_generate(
    model_max_length: int,
    tokenizer: PreTrainedTokenizer,
    examples: Dict[str, List[str]],
) -> Dict[str, List[List[int]]]:
    # build grouped texts with format `X1 X2 X3 ... <eos> X1 X2 X3 ... [<eos>]`
    token_ids_list: List[List[int]] = tokenizer(
        examples["text"], add_special_tokens=False
    )["input_ids"]
    token_ids_list = [
        token_ids + [tokenizer.eos_token_id] for token_ids in token_ids_list
    ]
    concatenated_ids = list(chain(*token_ids_list))
    # we drop the small remainder, and if the total_length < block_size, we exclude this batch
    total_length = (len(concatenated_ids) // model_max_length) * model_max_length
    result = [
        concatenated_ids[i : i + model_max_length]
        for i in range(0, total_length, model_max_length)
    ]
    return {"input_ids": result, "labels": result.copy()}


def exam_generate(model_max_length: int, tokenizer: PreTrainedTokenizer, data_point):
    template = "Human: \n{human}\n\nAssistant: \n"
    # pudb.set_trace()
    input_str = template.format(
        human=f'回答下面的{data_point["type"]}题，用json返回答案，包括原因和答案，如{{"reason":..., "answer":...}}\n{data_point["question"]}\n选项：{" ".join(data_point["candidates"])}'
    )
    input_ids = tokenizer.encode(input_str, add_special_tokens=False)
    labels = [IGNORE_INDEX] * len(input_ids)
    bot_ids = tokenizer.encode(
        json.dumps(
            {"reason": data_point["reason"], "answer": data_point["answer"]},
            ensure_ascii=False,
        ),
        add_special_tokens=False,
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
