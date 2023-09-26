import copy
import time
import types
from typing import Any, Dict, List, Optional, Union
from gradio_client import Client
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer
import deepspeed
from accelerate import Accelerator
from transformers.deepspeed import is_deepspeed_zero3_enabled
from deepspeed.runtime.engine import DeepSpeedEngine
import torch


def get_ds_state_dict(ds_engine: DeepSpeedEngine):
    """
    如果是zero stage 3，要对所有rank调用，无视掉stage3_gather_16bit_weights_on_model_save参数
    """
    if ds_engine.zero_optimization_partition_weights():
        # consolidation is expensive in time and memory and therefore isn't a default
        state_dict = ds_engine._zero3_consolidated_16bit_state_dict()
    else:
        state_dict = ds_engine.module.state_dict()
    return state_dict


def get_model_param_count(
    model: Union[DeepSpeedEngine, torch.nn.Module], trainable_only=False
):
    """
    Calculate model's total param count. If trainable_only is True then count only those requiring grads
    """
    if is_deepspeed_zero3_enabled() and isinstance(model, DeepSpeedEngine):

        def numel(p):
            return p.ds_numel

    else:

        def numel(p):
            return p.numel()

    return sum(
        numel(p) for p in model.parameters() if not trainable_only or p.requires_grad
    )


def bind_methods_from_class_to_instance(
    to_instance, from_class, include: Optional[List[str]] = None
):
    for method_name, method in vars(from_class).items():
        if callable(method) and (include is None or method_name in include):
            setattr(to_instance, method_name, types.MethodType(method, to_instance))


def prepare_deepspeed(accelerator: Accelerator, model: PreTrainedModel):
    # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
    deepspeed_plugin = accelerator.state.deepspeed_plugin
    config_kwargs = deepspeed_plugin.deepspeed_config
    if model is not None:
        if hasattr(model, "config"):
            hidden_size = (
                max(model.config.hidden_sizes)
                if getattr(model.config, "hidden_sizes", None)
                else getattr(model.config, "hidden_size", None)
            )
            if (
                hidden_size is not None
                and config_kwargs["zero_optimization"]["stage"] == 3
            ):
                # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                config_kwargs.update(
                    {
                        "zero_optimization.reduce_bucket_size": hidden_size
                        * hidden_size,
                        "zero_optimization.stage3_param_persistence_threshold": 10
                        * hidden_size,
                        "zero_optimization.stage3_prefetch_bucket_size": 0.9
                        * hidden_size
                        * hidden_size,
                    }
                )

    # If ZeRO-3 is used, we shard both the active and reference model.
    # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
    if config_kwargs["zero_optimization"]["stage"] != 3:
        config_kwargs["zero_optimization"]["stage"] = 0
    model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
    model.eval()
    return model


def zero_infer(
    accelerator: Accelerator,
    model: DeepSpeedEngine,
    tokenizer: PreTrainedTokenizer,
    texts: List[str],
    batch_size: int,
    pad_to_multiple_of=8,
):
    """
    必须要在所有进程同时调用，否则会卡住
    """
    model.eval()
    num_samples = len(texts)
    text_token_ids: Dict[str, List[List[int]]] = tokenizer(
        texts,
        add_special_tokens=False,
        pad_to_multiple_of=pad_to_multiple_of,
        padding=True,
    )
    input_ids_batches: List[List[int]] = []
    attention_mask_batches: List[List[int]] = []
    for i in range(0, num_samples, batch_size):
        input_ids_batches.append(text_token_ids["input_ids"][i : i + batch_size])
        attention_mask_batches.append(
            text_token_ids["attention_mask"][i : i + batch_size]
        )
    outputs: List[Dict[str, Any]] = []
    with torch.no_grad():
        for input_ids, attention_mask in zip(input_ids_batches, attention_mask_batches):
            outputs.append(
                model(
                    input_ids=torch.tensor(
                        input_ids, dtype=torch.int64, device=accelerator.device
                    ),
                    attention_mask=torch.tensor(
                        attention_mask, dtype=torch.int64, device=accelerator.device
                    ),
                )
            )
    return outputs


class MultiClient(object):
    def __init__(self, worker_addrs, synced_worker=False) -> None:
        self.clients = [Client(addr) for addr in worker_addrs]
        self.synced_worker = synced_worker

    def predict(self, tasks: List[List], max_retries: int = 3) -> List[Any]:
        assert len(tasks) >= 1, "No predict tasks!"
        num_tasks = len(tasks)
        if self.synced_worker and len(tasks) % len(self.clients) != 0:
            num_dummy_tasks = len(self.clients) - len(tasks) % len(self.clients)
            tasks.extend([copy.deepcopy(tasks[-1]) for _ in range(num_dummy_tasks)])

        pbar = tqdm(total=len(tasks))
        jobs = {
            client: (i, client.submit(*(tasks[i]), api_name="/predict"))
            for i, client in enumerate(self.clients)
            if i < len(tasks)
        }
        results = {}
        retries = {i: 0 for i in range(len(tasks))}

        while jobs:
            for client, (i, job) in list(jobs.items()):
                if job.done():
                    pbar.update(1)
                    del jobs[client]
                    try:
                        result = job.result()
                        results[i] = result
                    except Exception as e:
                        print("Job failed with error:", e)
                        if retries[i] < max_retries:
                            print("Retrying job...")
                            retries[i] += 1
                            new_job = client.submit(*tasks[i], api_name="/predict")
                            jobs[client] = (i, new_job)
                            continue  # Skip the rest of the loop
                        else:
                            results[i] = None

                    new_i = len(results) + len(jobs)
                    if new_i < len(tasks):
                        new_task = tasks[new_i]
                        new_job = client.submit(*new_task, api_name="/predict")
                        jobs[client] = (new_i, new_job)
            time.sleep(1)
        pbar.close()

        predicts = [results[i] for i in range(num_tasks)]

        return predicts
