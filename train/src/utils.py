import time
from typing import Any, List, Union
from gradio_client import Client
from tqdm import tqdm
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


def get_model_param_count(model: Union[DeepSpeedEngine, torch.nn.Module], trainable_only=False):
    """
    Calculate model's total param count. If trainable_only is True then count only those requiring grads
    """
    if is_deepspeed_zero3_enabled() and isinstance(model, DeepSpeedEngine):
        def numel(p):
            return p.ds_numel

    else:
        def numel(p):
            return p.numel()

    return sum(numel(p) for p in model.parameters() if not trainable_only or p.requires_grad)


class MultiClient(object):
    def __init__(self, worker_addrs) -> None:
        self.clients = [Client(addr) for addr in worker_addrs]

    def predict(self, tasks: List[List], max_retries: int = 3) -> List[Any]:
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
                            new_job = client.submit(
                                *tasks[i], api_name="/predict")
                            jobs[client] = (i, new_job)
                            continue  # Skip the rest of the loop
                        else:
                            results[i] = None

                    if tasks:
                        new_i = len(results) + len(jobs)
                        if new_i < len(tasks):
                            new_task = tasks[new_i]
                            new_job = client.submit(
                                *new_task, api_name="/predict")
                            jobs[client] = (new_i, new_job)
            time.sleep(0.1)
        pbar.close()

        predicts = [results[i] for i in sorted(results)]

        return predicts
