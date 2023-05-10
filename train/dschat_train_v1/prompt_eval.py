import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, AutoConfig
import argparse
from tqdm import tqdm
import json, os
parser = argparse.ArgumentParser()

parser.add_argument('--model_name_or_path',required=True,type=str)
parser.add_argument('--finetuned_model_name_or_path',required=True,type=str)
parser.add_argument('--test_file',required=True,type=str)
parser.add_argument('--predictions_file', default='./predictions.json', type=str)
args = parser.parse_args()

print("test_file: " + args.test_file)
print("model_name_or_path: " + args.model_name_or_path)
print("finetuned_model_name_or_path: " + args.finetuned_model_name_or_path)

max_new_tokens = 1024
generation_config = dict(
    temperature=0.001,
    top_k=30,
    top_p=0.85,
    do_sample=True,
    num_beams=1,
    repetition_penalty=1.2,
    max_new_tokens=max_new_tokens
    )


def read_data(filename):
    res = []
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            res.append(json.loads(line.strip()))
    return res


input_items = read_data(args.test_file)
output_items = []

def write_data(filename, examples):
    with open(filename, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

print("predictions will be written at {}".format(args.predictions_file))

def get_input_text(input_item):
    conversations = input_item['conversations']
    conv_turn = len(conversations)
    for i, sentence in conversations:
        sentence_from = sentence["from"].lower()
        sentence_value = 'Human: ' + sentence["value"] + '\n\nAssistant: ' if sentence_from == 'human' else sentence["value"]
        conversation += sentence_value
        sentence_ids = tokenizer.encode(sentence_value, add_special_tokens=False)#do not add bos_token_id
        label = copy.deepcopy(sentence_ids) if sentence_from != 'human' else [IGNORE_INDEX] * len(sentence_ids)
        input_ids += sentence_ids

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
    for sentence in source[:-1]:
        sentence_from = sentence["from"].lower()
        sentence_value = 'Human: ' + sentence["value"] + '\n\nAssistant: ' if sentence_from == 'human' else sentence["value"]
        conversation += sentence_value
        sentence_ids = tokenizer.encode(sentence_value, add_special_tokens=False)#do not add bos_token_id
        input_ids += sentence_ids
        if sentence_from != 'human':
            input_ids += [tokenizer.eos_token_id]#make sure eos_token_id is correct

    return input_ids, conversation

if __name__ == '__main__':
    load_type = torch.float16
    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device('cpu')

    if "llama" in args.model_name_or_path:
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 2
    model_config = AutoConfig.from_pretrained(args.model_name_or_path)
    print("Loading model")
    model = AutoModelForCausalLM.from_pretrained(
        args.finetuned_model_name_or_path, 
        torch_dtype=load_type,
        config=model_config,
        )

    model.to(device)
    model.eval()
    print("Load model successfully")

    index = 0

    for i in tqdm(range(len(input_items)), total=len(input_items), unit="item"):
        batch_input_items = input_items[i]

        input_ids, conversation = _addrole_masklabel_tokenize(source=input_items[i]['conversations'])
        input_ids = input_ids[:2048]
        if "Human" not in conversation:
            continue
        attention_mask = [1] * len(input_ids)
        input_ids = torch.LongTensor(input_ids).unsqueeze(0)
        attention_mask = torch.LongTensor(attention_mask).unsqueeze(0)
        #(1, max_seq_len)
        generation_output = model.generate(
            input_ids = input_ids.to(device), 
            attention_mask = attention_mask.to(device),
            **generation_config
        )

        generate_text = tokenizer.decode(generation_output[0].cpu().tolist(),skip_special_tokens=True)
        output_items.append({"generate_text": generate_text})
        if index%10 == 0:
            print("conversation: ", conversation)
            print("generate_text: ", generate_text)
            print("-"*100)
        index += 1
        
    write_data(args.predictions_file, output_items)
