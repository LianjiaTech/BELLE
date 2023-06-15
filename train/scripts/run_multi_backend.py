import fcntl
import subprocess
import os
import signal
import argparse
import sys
import time
import select
from transformers import HfArgumentParser
from dataclasses import dataclass, field

cuda_devices = os.getenv("CUDA_VISIBLE_DEVICES", "")
if cuda_devices == "":
    num_process = 1
else:
    num_process = len([int(idx) for idx in cuda_devices.split(",")])

@dataclass
class Arguments:
    model_name_or_path: str = field(metadata={'help': 'raw ckpt'})
    ckpt_path: str = field(metadata={'help': 'lora or raw ckpt'})
    use_lora: bool = field(default=False)


args = HfArgumentParser((Arguments)).parse_args_into_dataclasses()[0]

processes = []
outputs = {}  # 用于存储子进程的输出

# ANSI 转义序列颜色代码
COLORS = [
    "\033[31m",  # 红色
    "\033[32m",  # 绿色
    "\033[33m",  # 黄色
    "\033[34m",  # 蓝色
    "\033[35m",  # 紫色
    "\033[36m",  # 青色
    "\033[91m",  # 浅红色
    "\033[92m",  # 浅绿色
    "\033[93m",  # 浅黄色
    "\033[94m",  # 浅蓝色
    "\033[95m",  # 浅紫色
    "\033[96m",  # 浅青色
]

def set_non_blocking(file):
    fd = file.fileno()
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

def terminate_processes_and_exit(exit_code=0):
    print("终止子进程...")
    for process in processes:
        process.terminate()

    # 等待子进程终止
    for process in processes:
        process.wait()

    # 打印子进程的输出
    for local_rank, output in outputs.items():
        color_code = COLORS[local_rank % len(COLORS)]  # 根据进程编号选择颜色
        print(f"{color_code}rank: {local_rank} stdout: {output['stdout'].decode()}\033[0m")
        print(f"{color_code}rank: {local_rank} stderr: {output['stderr'].decode()}\033[0m")

    # 退出主进程
    sys.exit(exit_code)

def handle_termination(signal, frame):
    print("收到终止信号，终止子进程...")
    terminate_processes_and_exit()

signal.signal(signal.SIGINT, handle_termination)
signal.signal(signal.SIGTERM, handle_termination)

for local_rank in range(num_process):
    process = subprocess.Popen(        
        [
            "python",
            "src/entry_point/interface.py",
            "--local_rank", f"{local_rank}",
            "--model_name_or_path", args.model_name_or_path,
            "--ckpt_path", args.ckpt_path,
            "--llama",      
        ] + (["--use_lora"] if args.use_lora else []),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    processes.append(process)
    outputs[local_rank] = {
        "stdout": b"",
        "stderr": b"",
    }

for process in processes:
    set_non_blocking(process.stdout)
    set_non_blocking(process.stderr)

try:
    while True:
        for local_rank, process in enumerate(processes):
            stdout = process.stdout.read()
            stderr = process.stderr.read()

            if stdout:
                new_output = stdout[len(outputs[local_rank]["stdout"]) :]
                outputs[local_rank]["stdout"] += new_output
                color_code = COLORS[local_rank % len(COLORS)]  # 根据进程编号选择颜色
                print(f"{color_code}rank: {local_rank} stdout: {new_output.decode()}\033[0m")

            if stderr:
                new_output = stderr[len(outputs[local_rank]["stderr"]) :]
                outputs[local_rank]["stderr"] += new_output
                color_code = COLORS[local_rank % len(COLORS)]  # 根据进程编号选择颜色
                print(f"{color_code}rank: {local_rank} stderr: {new_output.decode()}\033[0m")

        processes = [process for process in processes if process.poll() is None]

        if not processes:
            break

        time.sleep(0.1)
except Exception as e:
    print("主脚本发生异常:", str(e))
    terminate_processes_and_exit(1)

