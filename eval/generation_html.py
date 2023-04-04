#!/usr/bin/env python
# -*- coding:utf-8 -*-
import pandas as pd
import json
import argparse

def read_data(path):
    datas = []
    with open(path) as f:
        for l in f.readlines():
            datas.append(eval(l))  # json.loads(l) 在加载某些行数据存在问题，这里使用eval
    return datas


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task"
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default="eval_prompt.json",
    )
    parser.add_argument(
        "--eval_set_path",
        type=str,
        default="eval_set.json",
    )
    parser.add_argument(
        "--html_path",
        type=str,
        default="template_html/ChatGPT_Score.html.temp",
    )
    parser.add_argument(
        "--output_html_path",
        type=str,
        default="ChatGPT_Score.html",
    )


    args = parser.parse_args()
    prompt_path = args.prompt_path
    eval_set_path = args.eval_set_path
    output_html_path = args.output_html_path
    html_path = args.html_path

    prompt_data = read_data(prompt_path)
    eval_set_data = read_data(eval_set_path)
    eval_set_data = json.dumps(eval_set_data, ensure_ascii=False)
    prompt_data = json.dumps(prompt_data, ensure_ascii=False)
    eval_set_str = f"const eval_set = {eval_set_data}"
    eval_prompt_str = f"const eval_prompt = {prompt_data}"
    with open(html_path, "r") as f:
        text = f.read()
        text = text.replace("const eval_set = []", eval_set_str)
        text = text.replace("const eval_prompt = []", eval_prompt_str)

    with open(output_html_path, "w") as f:
        f.write(text)
