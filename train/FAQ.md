# FAQ

**我们的机器配置是8卡A100 40G，所以下面的问题也只针对8卡A100 40G的配置**

LLaMA模型的max_seq_len通常设置为1024或2048

Bloomz模型的max_seq_len通常设置为512或1024

## <a name="1"></a> 1. 单机单卡可以训练多大参数量的模型

现在的deepspeed-chat项目还不支持offload，目前实验来看，在max_seq_len设置为1024的情况下，单机单卡(40G)上微调的模型参数量应该最多也就是bloomz-1b1。（总结的并不严谨，欢迎大家指正）

如果想要微调bloomz-1b7模型，可能需要将max_seq_len设置的很小。



## <a name="2"></a> 2. 单机多卡可以训练多大参数量的模型

目前来看，可以跑起来7b1的bloomz和7B的LLaMA。



## <a name="3"></a> 3. 单机单卡采用LoRA可以训练多大参数量的模型

可以训练7B的LLaMA。而对于7b1的bloom，需要将max_seq_len设置小一些。



## <a name="4"></a> 4. 单机多卡采用LoRA可以训练多大参数量的模型

可以训练13B的模型。目前还没在8卡A100 40G上尝试过更大参数量的模型。欢迎大家一起交流。

## <a name="5"></a> 5. 加载Llama tokenizer时存在的问题

Llama的tokenizer初始没有pad_token_id，需要赋值

实验过程中发现，不同的transformers版本在加载Llama词表时会出现一些问题，记录如下：

| transformers版本 | 问题                                                         |
| ---------------- | ------------------------------------------------------------ |
| 4.28.0.dev0      | 当前版本可正常加载tokenizer，unk_token_id=0, bos_token_id=1, eos_token_id=2 |
| 4.28.1           | AutoTokenizer.from_pretrained会出现RecursionError: maximum recursion depth exceeded，需要用LlamaTokenizer<br />eos_token_id，bos_token_id，unk_token_id都是0 |
| 4.29.0.dev0      | 该版本与4.28.1存在同样的问题                                 |

目前统一解决办法是，如果模型是llama，则做如下赋值操作：

```bash
tokenizer.pad_token_id = 0
tokenizer.bos_token_id = 1
tokenizer.eos_token_id = 2
```

## <a name="6"></a> 6. 加载2M的数据量需要多大的内存和多长时间

对于200万的数据量，通过观察，大概要350G的内存，加载时长大概在25min左右(这是Bloom的时长，如果是Llama，tokenize的时间会加长)

我们目前尚未对加载数据部分的代码做优化，包括内存和时长。

## <a name="7"></a> 7. 训练模型的生成结果非常糟糕

这里的糟糕指的是生成的结果形如：“我们用减法计算出小明还剩下多少个鸡蛋多少个鸡蛋多少个鸡蛋多少个鸡蛋个鸡蛋个鸡蛋减法计算减法计算蛋个鸡蛋”

就目前我们的实验经验来看，出现这个问题的主要原因出在这几个特殊的token_id上，尤其是pad_token_id和eos_token_id的值，要确保两者不相等，而且pad_token_id=0, eos_token_id=2。（不区分LLaMA和Bloom）

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