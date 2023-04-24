# FAQ

我们的机器配置是8卡A100 40G，所以下面的问题也只针对8卡A100 40G的配置。max_seq_len设置为1024

## <a name="1"></a> 1. 单机单卡可以训练多大参数量的模型

现在的deepspeed-chat项目还不支持offload，所以目前实验来看，在max_seq_len设置为1024的情况下，单机单卡(40G)上微调的模型参数量应该最多也就是bloomz-1b1。（总结的并不严谨，欢迎大家指正）

如果想要微调bloomz-1b7模型，可能需要将max_seq_len设置的很小。



## <a name="2"></a> 2. 单机多卡可以训练多大参数量的模型

目前来看，可以跑起来7b1的bloomz和7B的LLaMA。



## <a name="3"></a> 3. 单机单卡采用LoRA可以训练多大参数量的模型

可以训练7B的LLaMA。而对于7b1的bloom，需要将max_seq_len设置小一些。



## <a name="4"></a> 4. 单机多卡采用LoRA可以训练多大参数量的模型

可以训练13B的模型。目前还没在8卡A100 40G上尝试过更大参数量的模型。欢迎大家一起交流。

## <a name="Others"></a>Others

这里我们提供了在实验过程中遇到的一些报错的情况，并提供了参考的解决方案。（**注：参考方案未必一定能够解决对应的问题**）

| 报错信息                                                     | 参考                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| RuntimeError: CUDA error: an illegal memory access was encountered.CUDA kernel errors might be asynchronously reported at some other API call,so the stacktrace below might be incorrect | https://github.com/pytorch/pytorch/issues/21819              |
| AutoTokenizer.from_pretrained("llama_model_path")出现递归error<br />RecursionError: maximum recursion depth exceeded | 有可能是transformers版本的问题，对于LlamaModel，可采用LlamaTokenizer加载 |
| xx>=0.11.0 is required for a normal functioning of this module, but found xx==0.10.0 | 这是因为版本不匹配导致的问题，可按照报错信息安装要求的版本即可 |
| torch.distributed.distributed_c10d.init_process_group() got multiple values for keyword argument 'backend' | transformers降低版本至4.28.1                                 |
| RuntimeError: Error building extension 'fused_adam'          | sudo ln -s /usr/local/cuda/lib64/libcudart.so /usr/lib/libcudart.so |
| huggingface_hub.utils._validators.HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': . Use `repo_type` argument if needed. | 这是因为docker容器内访问不到model_name_or_path，需要挂载到物理机对应的目录。 |