# ZeRO Inference

## 1.1 什么是ZeRO Inference
[Zero Inference](https://www.deepspeed.ai/2022/09/09/zero-inference.html)利用ZeRO stage 3的数据并行特性，能够将模型分布到多张GPU上，或者Offload到内存或者NVMe上，推理单GPU无法加载的模型

## 1.2 Zero Inference注意事项
Zero Inference是数据并行的推理，因此需要在各个GPU同时启动推理进程并进行`model.forward`，否则会卡住

## 1.3 和其他并行策略的比较
- 张量并行（对模型手动切分）
    - 通信量：
        - $O(batch \times len \times layer \times hidden)$
    - 计算和通讯不能同时进行
- 流水线并行（`AutoModelForCausalLM.from_pretrained(device_map="auto")`）
    - 通信量：
        - $O(batch \times len \times layer \times hidden)$
    - 计算和通讯不能同时进行
- 数据并行（ZeRO Inference）
    - 通信量：
        - $O(hidden \times hidden)$
    - 计算和通讯可以同时进行

## 2 运行代码
本仓库实现了LLaMA和LLaMA 2的flash attention，可在推理时启动，降低显存占用。
由于flash attention不支持自定义的attention mask，启动flash attention时，batch size必须设为1并关闭任何padding。
### 2.1 批量推理（推荐）

#### 2.1.1 准备数据
```
{"text": xxx}
{"text": xxx}
```

#### 2.1.2 运行
```
bash scripts/run_zero_inference.sh
```
可传入的参数有`max_new_tokens`、`min_new_tokens`、`do_sample`、`num_beams`、`temperature`、`top_k`、`top_p`、`repetition_penalty`。
具体说明见[huggingface文档](https://huggingface.co/docs/transformers/main/main_classes/text_generation)

### 2.2 推理后端（前后端分离）

#### 2.2.1 运行后端
```
bash scripts/run_zero_inference_backend_without_trainer.sh
```
- `devices`：指令使用哪几个显卡，格式同`CUDA_VISIBLE_DEVICES`
- `base_port`：后端服务监听端口，打开[`base_port`, `base_port` + `num_devices` - 1]的端口

#### 2.2.2 运行前端
运行`src/evaluation.ipynb`，由于ZeRO Inference要求多个`model.forward`必须同时运行，必须设置`synced_worker=True`，同时保证客户端连接上了每个后端进程
