import time

import torch
import torch.nn as nn

from gptq import *
from modelutils import *
from quant import *

from transformers import AutoTokenizer

DEV = torch.device('cuda:0')

def get_llama(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = 2048
    return model

def load_quant(model, checkpoint, wbits, groupsize):
    from transformers import LlamaConfig, LlamaForCausalLM 
    config = LlamaConfig.from_pretrained(model)
    def noop(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = noop 
    torch.nn.init.uniform_ = noop 
    torch.nn.init.normal_ = noop 

    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = LlamaForCausalLM(config)
    torch.set_default_dtype(torch.float)
    model = model.eval()
    layers = find_layers(model)
    for name in ['lm_head']:
        if name in layers:
            del layers[name]
    make_quant(model, layers, wbits, groupsize)

    print('Loading model ...')
    if checkpoint.endswith('.safetensors'):
        from safetensors.torch import load_file as safe_load
        model.load_state_dict(safe_load(checkpoint))
    else:
        model.load_state_dict(torch.load(checkpoint))
    model.seqlen = 2048
    print('Done.')

    return model

if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='llama model to load'
    )
    parser.add_argument(
        '--wbits', type=int, default=16, choices=[2, 3, 4, 8, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--groupsize', type=int, default=-1,
        help='Groupsize to use for quantization; default uses full row.'
    )
    parser.add_argument(
        '--load', type=str, default='',
        help='Load quantized model.'
    )

    parser.add_argument(
        '--text', type=str,
        help='input text'
    )
    
    parser.add_argument(
        '--min_length', type=int, default=10,
        help='The minimum length of the sequence to be generated.'
    )
    
    parser.add_argument(
        '--max_length', type=int, default=1024,
        help='The maximum length of the sequence to be generated.'
    )
    
    parser.add_argument(
        '--top_p', type=float , default=0.95,
        help='If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.'
    )
    
    parser.add_argument(
        '--temperature', type=float, default=0.8,
        help='The value used to module the next token probabilities.'
    )
    
    args = parser.parse_args()

    if type(args.load) is not str:
        args.load = args.load.as_posix()
    
    if args.load:
        model = load_quant(args.model, args.load, args.wbits, args.groupsize)
    else:
        model = get_llama(args.model)
        model.eval()
        
    model.to(DEV)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print("Human:")
    line = input()
    while line:
        inputs = 'Human: ' + line.strip() + '\n\nAssistant:'
        input_ids = tokenizer.encode(inputs, return_tensors="pt").to(DEV)

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids,
                do_sample=True,
                min_length=args.min_length,
                max_length=args.max_length,
                top_p=args.top_p,
                temperature=args.temperature,
            )
        print("Assistant:\n") 
        print(tokenizer.decode([el.item() for el in generated_ids[0]])[len(inputs)+4:]) # generated_ids开头加上了bos_token,需要将inpu的内容截断,只输出Assistant 
        print("\n-------------------------------\n")
        print("Human:") #每次终端用户输入前，加上Human提示。
        line = input()
