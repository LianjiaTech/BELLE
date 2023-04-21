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

dev_batch_size = 4

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

    model_config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.finetuned_model_name_or_path, 
        torch_dtype=load_type,
        config=model_config,
        )

    model.to(device)
    model.eval()
    print("Load model successfully")

    index = 0

    for i in tqdm(range(0, len(input_items), dev_batch_size), total=len(input_items)//dev_batch_size, unit="item"):
        batch_input_items = input_items[i:i+dev_batch_size]
        batch_input_text = ["Human: "+input_item['instruction']+"\nAssistant: " for input_item in batch_input_items]

        batch_inputs = tokenizer(batch_input_text, max_length=max_new_tokens, padding=True, truncation=True,return_tensors="pt")  #add_special_tokens=False ?
        batch_generation_output = model.generate(
            input_ids = batch_inputs["input_ids"].to(device), 
            attention_mask = batch_inputs['attention_mask'].to(device),
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            **generation_config
        )

        batch_generate_text = tokenizer.batch_decode(batch_generation_output,skip_special_tokens=True)

        for generate_text, input_item in zip(batch_generate_text, batch_input_items):
            output_items.append({"instruction": input_item['instruction'],"generate_text":generate_text})
            if index%100 == 0:
                print(generate_text)
            index += 1
        
    write_data(args.predictions_file, output_items)
