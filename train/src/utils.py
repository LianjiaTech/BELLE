import copy
import time
import types
from typing import Any, List, Optional, Union
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


def bind_methods_from_class_to_instance(to_instance, from_class, include: Optional[List[str]] = None):
    for method_name, method in vars(from_class).items():
        if callable(method) and (include is None or method_name in include):
            setattr(to_instance, method_name, types.MethodType(method, to_instance))


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
                            new_job = client.submit(
                                *tasks[i], api_name="/predict")
                            jobs[client] = (i, new_job)
                            continue  # Skip the rest of the loop
                        else:
                            results[i] = None

                    new_i = len(results) + len(jobs)
                    if new_i < len(tasks):
                        new_task = tasks[new_i]
                        new_job = client.submit(
                            *new_task, api_name="/predict")
                        jobs[client] = (new_i, new_job)
            time.sleep(1)
        pbar.close()

        predicts = [results[i] for i in range(num_tasks)]

        return predicts
