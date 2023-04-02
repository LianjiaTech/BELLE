import os
import json
import torch
import torch.nn as nn
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoTokenizer, AutoModelForCausalLM

class GEN4ALL(nn.Module):
    """
    Fine-tune Causal Model For Text Generation
    """

    def __init__(self, args):
        super(GEN4ALL, self).__init__()
        self.args = args
        self.model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)


    def forward(self, input_ids, attention_mask, labels, decoder_input_ids = None, decoder_attention_mask = None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs


    def save_model(self, output_dir):
        self.model.config.use_cache = False
        if self.args.use_lora:
            old_state_dict = model.state_dict
            self.model.state_dict = (
                lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
            ).__get__(model, type(model))

        self.model.save_pretrained(output_dir)


