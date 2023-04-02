import sentencepiece
import pytorch_lightning
import os
import sys
import json
import random
import logging
import argparse
import datetime
import traceback
from collections import namedtuple
from typing import Union, List

import time
import torch
import numpy as np
from torch.optim import SGD
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.plugins import DeepSpeedPlugin
from pytorch_lightning.strategies import DeepSpeedStrategy, DDPStrategy
from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_info
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
import transformers
from transformers import AdamW, get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
from transformers.optimization import  get_scheduler
from lightning_transformers.utilities.deepspeed import enable_transformers_pretrained_deepspeed_sharding

from dataset import GEN4ALLDataset, Tokenizer4GEN, TruncateDataset
from utils import get_logger, set_random_seed
from model import GEN4ALL
from functools import partial



print('Current time: ', time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
print('Cuda device: ', torch.cuda.device_count())

set_random_seed(0)

class GEN4ALLTrainer(pl.LightningModule):
    def __init__(self, args: Union[argparse.Namespace, dict]):
        """Initialize a model, tokenizer and config."""
        super().__init__()
        log_format = '%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'

        if isinstance(args, argparse.Namespace):
            self.save_hyperparameters(args)
            self.args = args
            logging.basicConfig(format=log_format,
                                filename=os.path.join(self.args.default_root_dir, "eval_result_log.txt"),
                                level=logging.INFO)
        elif isinstance(args, dict):
            # eval mode
            TmpArgs = namedtuple("tmp_args", field_names=list(args.keys()))
            self.args = TmpArgs(**args)
            logging.basicConfig(format=log_format,
                                filename=os.path.join(self.args.default_root_dir, "eval_test.txt"),
                                level=logging.INFO)
        else:
            raise NotImplementedError("undefined args type for [{}] !!!".format(type(args)))

        # target loss only
        self.label_keyword = "target_only_labels" if self.args.target_loss_only is True else "whole_labels"

        self.data_dir = self.args.data_dir

        if self.args.notrain:
            self.model = GEN4ALL(self.args)

        self.result_logger = logging.getLogger(__name__)
        self.result_logger.setLevel(logging.INFO)
        self.result_logger.info(
            str(self.args.__dict__ if isinstance(self.args, argparse.ArgumentParser) else self.args))

        self.result_logger.info("transformers.__version__ : {}".format(transformers.__version__))
        self.result_logger.info("torch.__version__ : {}".format(torch.__version__))
        self.result_logger.info("pytorch_lighting.__version__ : {}".format(pl.__version__))
        self.result_logger.info("numpy.__version__ : {}".format(np.__version__))
        self.result_logger.info('Cuda device: {}'.format(torch.cuda.device_count()))
        self.optimizer = self.args.optimizer
        self.result_logger.info("self.optimizer : {}".format(self.optimizer))
        self.result_logger.info("self.strategy : {}".format(self.args.speedup))
        self.t_total = -1
        self.every_n_train_steps_to_save = -1
        
        self.tokenizer = Tokenizer4GEN(model_name_or_path = self.args.model_name_or_path, max_seq_length = self.args.max_length, max_ans_length = self.args.max_ans_length, padding_side = "left", padding_mode = "true")


    def setup(self, stage):
        if not self.args.notrain:
            logging.info("Initializing model...")
            enable_transformers_pretrained_deepspeed_sharding(self)
            self.model = GEN4ALL(self.args)

    def configure_optimizers(self):
        print('deepspeed_offload', self.deepspeed_offload)
        no_decay = ["bias", "LayerNorm.weight"]
        params_decay = [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)]
        params_nodecay = [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)]
        optim_groups = [
            {"params": params_decay, "weight_decay": self.hparams.weight_decay},
            {"params": params_nodecay, "weight_decay": 0.0},
        ]
        # todo: need to enable deepspeed cpu adam only if offloading

        if self.deepspeed_offload:
            print ('\noptimizer: DeepSpeedCPUAdam\n')
            optimizer = DeepSpeedCPUAdam(optim_groups, lr=self.hparams.lr, betas=self.hparams.betas)
        else:
            print ('\noptimizer: FusedAdam\n')
            optimizer = FusedAdam(optim_groups, lr=self.hparams.lr, betas=self.hparams.betas)

        t_total = (len(self.train_dataloader()) // (
                    self.args.accumulate_grad_batches * num_gpus) + 1) * self.args.max_epochs if self.t_total == -1 else self.t_total
        warmup_steps = int(t_total * self.args.warmup_rate)

        scheduler = get_scheduler(name=self.args.lr_scheduler, optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        return {
            "optimizer": optimizer,
            "lr_scheduler":{
                "scheduler": scheduler,
                "interval": "step"
            }
        }


    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            config = strategy.config['zero_optimization']
            return config.get('offload_optimizer') or config.get('offload_param')
        return False

    def training_step(self, batch, batch_idx):

        tf_board_logs = {"lr": self.trainer.optimizers[0].param_groups[0]['lr']}

        outputs = self.model.forward(
            input_ids=batch["whole_input_ids"],
            attention_mask=batch["whole_attention_mask"],
            labels=batch[self.label_keyword]
            )
        train_loss = outputs.loss
        tf_board_logs[f"train_loss"] = train_loss
        self.log('train_loss', train_loss)
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'])

        return {'loss': train_loss, 'log': tf_board_logs}

    def validation_step(self, batch, batch_idx):
        output = dict()
        outputs = self.model.forward(
            input_ids=batch["whole_input_ids"],
            attention_mask=batch["whole_attention_mask"],
            labels=batch[self.label_keyword]
        )
        val_loss = outputs.loss
        output["val_loss"] = val_loss

        # output["mrc_result"] = {"sample_ids": batch["sample_ids"], "predict_text": output_texts}
        return output

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss.cpu().numpy().tolist()}
        self.log('val_loss', avg_loss, sync_dist=True)
        return {"val_loss": avg_loss, "log": tensorboard_logs}
    

    def train_dataloader(self) -> DataLoader:
        print ('Loading train data')
        return self.get_dataloader("train")

    def val_dataloader(self) -> DataLoader:
        print('Loading val data')
        return self.get_dataloader("dev")

    def test_dataloader(self) -> DataLoader:
        print('Loading test data')
        return self.get_dataloader("test")

    def predict_dataloader(self) -> DataLoader:
        print('Loading predict data')
        return self.get_dataloader("test")


    def get_dataloader(self, prefix, limit: int = None) -> DataLoader:
        """
        load_mmap_dataset
        """
        if prefix == 'train':
            file_name = self.args.train_file
        elif prefix == 'dev':
            file_name = self.args.dev_file
        elif prefix == 'test':
            file_name = self.args.test_file
        else:
            raise NotImplementedError("undefined prefix, which should be ['train', 'dev', 'test'].")

        if self.args.data_size <= 0 and not self.args.iter_dataset:
            json_path = os.path.join(self.data_dir, file_name)
            if not os.path.exists(json_path):
                raise NotImplementedError("File [{}] does not exists !!!".format(json_path))
        else:
            json_path = self.data_dir

        if not self.args.iter_dataset:
            dataset = GEN4ALLDataset(
                json_path=json_path,
                tokenizer=self.tokenizer,
                bos_token = self.tokenizer.bos_token,
                eos_token = self.tokenizer.eos_token,
                max_length=self.args.max_length,
                data_tokenized=self.args.data_tokenized,
                data_size=self.args.data_size
            )
        else:
            dataset = GEN4ALLDatasetIter(
                json_path=json_path,
                tokenizer=self.tokenizer,
                bos_token_id=self.tokenizer.bos_token_id if self.args.tokenizer_type == 'bert' else None,
                eos_token_id=self.tokenizer.eos_token_id,
                max_length=self.args.max_length,
                data_tokenized=self.args.data_tokenized,
                data_size=self.args.data_size
            )

        if limit is not None:
            dataset = TruncateDataset(dataset, limit)

        def _smart_batching_collate_gen(batch):
            sample_ids = []
            queries, answers = [], []
            for sample_id, input_text, target_text in batch:
                sample_ids.append(sample_id)
                queries.append(input_text)
                answers.append(target_text)
            
            feats = self.tokenizer(queries, answers, sample_ids)
            return feats

        dataset_length = len(dataset)
        for _ in range(5):
            idx = random.randint(0, dataset_length - 1)
            self.result_logger.info("Example {}:  {}/{}".format(idx, dataset.all_data[idx], dataset_length))

        loader_batch_size = self.args.batch_size if prefix == "train" else self.args.dev_batch_size
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=loader_batch_size,
            num_workers=4,
            shuffle=True if prefix == "train" else False,
            collate_fn=_smart_batching_collate_gen
        )

        return dataloader


def pad_tensors_to_max_length(input_tensor: torch.Tensor, max_length: int, pad_token_id: int):
    padded_tensor = pad_token_id * torch.ones((max_length,), dtype=input_tensor.dtype, device=input_tensor.device)
    padded_tensor[-input_tensor.shape[0]:] = input_tensor
    return padded_tensor


def main():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True, help="data dir")
    parser.add_argument("--tokenizer_type", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--dev_file", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--max_keep_ckpt", default=3, type=int, help="the number of keeping ckpt max.")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="bert config dir")
    parser.add_argument("--max_length", type=int, default=128, help="max length of dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--dev_batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--workers", type=int, default=0, help="num workers for dataloader")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="warmup steps used for scheduler.")
    parser.add_argument("--warmup_rate", default=0.1, type=float, help="rate of warmup steps used for scheduler.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--seed", default=42, type=int, help="set random seed for reproducing results.")
    parser.add_argument('--target_loss_only', action='store_true', help="calculate target loss only")
    parser.add_argument("--speedup", type=str, required=False)
    parser.add_argument("--betas", nargs='+', type=float, required=True)
    parser.add_argument("--notrain", action='store_true')
    parser.add_argument("--data_tokenized", action='store_true')
    parser.add_argument("--data_size", type=int, default=-1, help="Specified when data is loaded from disk.")
    parser.add_argument("--use_lora", action="store_true", help="use lora")
    parser.add_argument("--iter_dataset", action='store_true', help="use iter datasets")
    parser.add_argument("--optimizer", choices=["adamw", "sgd", "torch.adam"], default="adamw", help="loss type")
    parser.add_argument("--final_div_factor", type=float, default=1e4, help="final div factor of linear decay scheduler")
    parser.add_argument("--lr_scheduler", type=str, default="linear")
    parser.add_argument("--lr_mini", type=float, default=-1)
    parser.add_argument("--max_ans_length", type=int, default=12)

    parser = Trainer.add_argparse_args(parser)

    # parsing args from cli
    args = parser.parse_args()
    if not os.path.exists(args.default_root_dir):
        os.makedirs(args.default_root_dir, exist_ok=True)

    model = GEN4ALLTrainer(args)
    train_dataloader = model.get_dataloader('train')

    t_total = (len(train_dataloader) // (model.args.accumulate_grad_batches * int(args.devices) + 1)) * model.args.max_epochs
    model.t_total = t_total
    model.every_n_train_steps_to_save = model.t_total//args.max_keep_ckpt

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.default_root_dir,  
        save_top_k=args.max_keep_ckpt,
        verbose=True,
        monitor="val_loss", 
        mode="min",
        save_weights_only=True,
        # every_n_train_steps=args.save_steps,
        # every_n_epochs=0.5
        save_last=True
    )

    logging.info('From args:\n')
    for key, value in args.__dict__.items():
        logging.info(' %s: %s' % (key, value))

    strategy = args.speedup
    logging.info('Speedup strategy : %s' % strategy)
    trainer = Trainer.from_argparse_args(
        args,
        callbacks=[checkpoint_callback],
        deterministic=True,
        strategy=strategy,
        accelerator='gpu',
        log_every_n_steps = model.every_n_train_steps_to_save
    )
    if not args.notrain:
        trainer.fit(model, train_dataloaders = train_dataloader)
        model.model.save_model(output_dir=args.default_root_dir)
        model.tokenizer.save_pretrained(args.default_root_dir)
    else:
        trainer.validate(model)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
        print(traceback.print_exc())
