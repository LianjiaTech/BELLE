# FAQ

这里给出一些实验过程中遇到的问题以及供参考的解决方案，同时对问题大致地分类

**解决方案仅供参考，未必能彻底解决对应问题！！！**

**解决方案仅供参考，未必能彻底解决对应问题！！！**

**解决方案仅供参考，未必能彻底解决对应问题！！！**

### Deepspeed相关

| 报错信息                                                     | 参考                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| RuntimeError: CUDA error: an illegal memory access was encountered.CUDA kernel errors might be asynchronously reported at some other API call,so the stacktrace below might be incorrect | https://github.com/pytorch/pytorch/issues/21819              |
| RuntimeError: Error building extension 'fused_adam'          | sudo ln -s /usr/local/cuda/lib64/libcudart.so /usr/lib/libcudart.so |
| RuntimeError: expected scalar type Float but found Half      | use_int8_training和deepspeed不能同时指定                     |
| RuntimeError: expected scalar type Float but found Half      | V100显卡上 use_int8_training和fp16不能同时指定               |

### transformers相关

| 报错信息                                                     | 参考                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| AutoTokenizer.from_pretrained("llama_model_path")出现递归error<br />RecursionError: maximum recursion depth exceeded | 有可能是transformers版本的问题，对于LlamaModel，可采用LlamaTokenizer加载 |
| torch.distributed.distributed_c10d.init_process_group() got multiple values for keyword argument 'backend' | transformers降低版本至4.28.1                                 |
|                                                              |                                                              |

### 其他问题

| 报错信息                                                     | 参考                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| V100机器上8bit量化训练失败或loss不正常                       | https://github.com/Facico/Chinese-Vicuna/issues/39<br />https://github.com/TimDettmers/bitsandbytes/issues/100<br />https://github.com/mymusise/ChatGLM-Tuning/issues/19<br />https://github.com/tloen/alpaca-lora/issues/170 |
|                                                              |                                                              |
| huggingface_hub.utils._validators.HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': . Use `repo_type` argument if needed. | 这是因为docker容器内访问不到model_name_or_path，需要挂载到物理机对应的目录。 |



这里给出一些实验建议：

1. 不开deepspeed会占用更多显存，建议全量参数finetune模式尽可能采用deepspeed
2. LoRA训练如果采用8bit量化，就不能使用deepspeed；如果使用deepspeed，就不能指定use_int8_training

关于deepspeed的配置可参考：

1. https://github.com/microsoft/DeepSpeed/issues/2187
2. https://www.deepspeed.ai/tutorials/advanced-install/
3. https://github.com/pyg-team/pytorch_geometric/issues/1001

