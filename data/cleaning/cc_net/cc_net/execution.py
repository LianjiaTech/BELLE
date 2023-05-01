# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import functools
import itertools
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Sized

import submitit
from typing_extensions import Protocol


class Executor(Protocol):
    def __call__(self, function: Callable[..., str], *args: Iterable) -> None:
        ...


class SubmititRetryOnTimeout(submitit.helpers.Checkpointable):
    def __init__(self, fn: Callable):
        self.fn = fn
        self.__name__ = fn.__name__

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


def get_executor(
    name: str,
    log_dir: Path,
    execution: str,
    timeout_hour: float = 1.0,
    mem_gb: int = 1,
    cpus: int = 1,
    task_parallelism: int = -1,
    options: dict = {},
) -> Executor:

    execution_mode = execution.split(",")[0]
    options.update(
        {kv.split("=", 1)[0]: kv.split("=", 1)[1] for kv in execution.split(",")[1:]}
    )

    if execution_mode == "mp":
        warnings.warn("Execution mode 'mp' is deprecated, use 'local'.")
        execution_mode = "local"

    cluster = None if execution_mode == "auto" else execution_mode
    # use submitit to detect which executor is available
    ex = submitit.AutoExecutor(log_dir, cluster=cluster)

    if ex.cluster == "local":
        # LocalExecutor doesn't respect task_parallelism
        return functools.partial(custom_map_array, ex, task_parallelism)
    if ex.cluster == "debug":
        return debug_executor

    # We are on slurm
    if task_parallelism == -1:
        task_parallelism = 500

    ex.update_parameters(
        name=name,
        timeout_min=int(timeout_hour * 60),
        mem_gb=mem_gb,
        cpus_per_task=cpus,
        slurm_array_parallelism=task_parallelism,
        **options,
    )
    return functools.partial(map_array_and_wait, ex)


def map_array_and_wait(
    ex: submitit.AutoExecutor, function: Callable[..., str], *args: Iterable
):
    f_name = function.__name__

    assert len(args) > 0, f"No arguments passed to {f_name}"
    approx_length = _approx_length(*args)

    print(f"Submitting {f_name} in a job array ({approx_length} jobs)")
    jobs = ex.map_array(function, *args)
    if not jobs:
        return
    failed_jobs = []
    done = 0
    total = len(jobs)
    job_array_id = jobs[0].job_id.split("_")[0]
    print(f"Started {f_name} in job array {job_array_id} ({len(jobs)} jobs).")
    for job in submitit.helpers.as_completed(jobs):
        done += 1
        e = job.exception()
        if not e:
            print(f"Finished job {job.job_id} ({done} / {total}).", job.result())
            continue

        print(f"Failed job {job.job_id} ({done} / {total}):", e)
        failed_jobs.append(job)

    if failed_jobs:
        n_failures = 10
        message = f"{len(failed_jobs)} / {done} jobs failed while running {f_name}"
        print(message)
        for job in failed_jobs[:n_failures]:
            print(f"Failed {job.job_id} -> {job.paths.stderr}")
        if len(failed_jobs) > n_failures:
            print(f"... ({len(failed_jobs) - n_failures} failed job skipped)")
        raise Exception(message)


def debug_executor(function: Callable[..., Optional[str]], *args: Iterable) -> None:
    logging.getLogger().setLevel(logging.DEBUG)
    approx_length = _approx_length(*args)
    for i, x in enumerate(zip(*args)):
        try:
            message = function(*x)
        except Exception:
            try:
                import ipdb as pdb  # type: ignore
            except ImportError:
                import pdb  # type: ignore
            import traceback

            traceback.print_exc()
            print("")
            pdb.post_mortem()
            sys.exit(1)
        if message is not None:
            print(message, f"({i + 1} / {approx_length})")


def _approx_length(*args: Iterable):
    for a in args:
        if isinstance(a, Sized):
            return len(a)
    return -1


def custom_map_array(
    ex: submitit.AutoExecutor,
    parallelism: int,
    function: Callable[..., Optional[str]],
    *args: Iterable,
) -> None:
    f_name = function.__name__
    assert len(args) > 0, f"No arguments passed to {f_name}"

    jobs_args = list(zip(*args))
    total = len(jobs_args)
    if parallelism < 0:
        parallelism = os.cpu_count() or 0
    assert parallelism >= 0, f"Can't run any jobs with task_parallelism={parallelism}"
    print(f"Submitting {total} jobs for {f_name}, with task_parallelism={parallelism}")
    enqueued = 0
    done = 0
    running_jobs: List[submitit.Job] = []
    failed_jobs: List[submitit.Job] = []

    while done < len(jobs_args):
        # Try to queue more job if we have some bandwidth.
        if enqueued < total and len(running_jobs) < parallelism:
            running_jobs.append(ex.submit(function, *jobs_args[enqueued]))
            enqueued += 1
            continue

        # Else wait for some job to finish
        if not running_jobs:
            warnings.warn(
                f"No more running jobs, yet we submitted only {enqueued} / {total} and finished {done} / {total}"
            )
            break

        job = get_next_job(running_jobs)
        running_jobs.remove(job)
        done += 1
        e = job.exception()
        if not e:
            print(f"Finished job {job.job_id} ({done} / {total}).", job.result())
            continue

        print(f"Failed job {job.job_id} ({done} / {total}):", e)
        failed_jobs.append(job)

    if failed_jobs:
        n_failures = 10
        message = f"{len(failed_jobs)} / {done} jobs failed while running {f_name}"
        print(message)
        for job in failed_jobs[:n_failures]:
            print(f"Failed {job.job_id} -> {job.paths.stderr}")
        if len(failed_jobs) > n_failures:
            print(f"... ({len(failed_jobs) - n_failures} failed job skipped)")
        raise Exception(message)


def get_next_job(
    jobs: Sequence[submitit.Job], poll_frequency: float = 10
) -> submitit.Job:
    """
    Waits for any of the job to finish and returns it.

    jobs: list of jobs
    poll_frequency: frequency in second at which we check job status
    """
    start = time.time()
    waiting = False
    while True:
        for job in jobs:
            if job.done():
                return job
        if not waiting:
            job_ids = [j.job_id for j in jobs[:4]]
            suffix = "..." if len(jobs) > 4 else ""
            print(
                f"Waiting on {len(jobs)} running jobs. Job ids: {','.join(job_ids)}{suffix}"
            )
            waiting = True
        time.sleep(poll_frequency)
