# BELLE训练
| [English](https://github.com/LianjiaTech/BELLE/blob/main/train/docs/README_en.md) | [中文](https://github.com/LianjiaTech/BELLE/blob/main/train/README.md)

当前仓库的代码属于BELLE项目训练代码v2版，上一版基于deepspeed-chat的代码放在dschat_train_v1目录下，未做任何改动。

考虑到如下因素和目前大家提出的issues，我们更新了仓库的训练代码

1. 没有deepspeed环境时无法使用仓库代码训练模型
2. deepspeed-chat没有集成peft包，对参数高效微调这一块的可扩展性不高

当前v2版本的代码对环境的依赖性较低，而且更加简洁。

## 1. 准备环境

### 1.1 Docker镜像

我们提供了一个完整可运行的Docker镜像，Dockerfile写在docker文件夹下。

考虑到build存在一定的困难，我们提供了镜像下载，你可以使用下面命令从dockerhub拉取我们的镜像，然后在镜像中运行代码，详见[docker环境说明](../docker/README.md)。

```shell
sudo docker pull tothemoon/belle:latest
git clone https://github.com/LianjiaTech/BELLE.git
```
```
sudo docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    --network host \
    --privileged \
    [--env https_proxy=$https_proxy \]
    [--env http_proxy=$http_proxy \]
    [--env all_proxy=$all_proxy \]
    --env HF_HOME=$hf_home \
    -it [--rm] \
    --name belle \
    -v $belle_path:$belle_path \
    -v $hf_home:$hf_home \
    -v $ssh_pub_key:/root/.ssh/authorized_keys \
    -w $workdir \
    $docker_user/belle:$tag \
    [--sshd_port 2201 --cmd "echo 'Hello, world!' && /bin/bash"]
```
`[]`中内容可忽略
- `--rm`：容器退出时销毁，如果长期在容器中工作，可忽略
- `--sshd_port`：sshd监听端口，默认是22001
- `--cmd`：容器要执行的命令`"echo 'Hello, world!' && /bin/bash"`，可忽略
- `hf_home`：huggingface缓存目录
- `$ssh_pub_key`：sshd公钥目录

上述命令实现了以下几点：

1. 拉取docker镜像
2. clone BELLE仓库
3. 将BELLE目录挂载
4. 将huggingface目录挂载。其中huggingface_models代表预训练模型的保存路径，该目录下存放所有需要的预训练语言模型，如llama-7b, bloomz-7b1-mt等
5. 注意：上述挂载的目录必须是绝对路径

### 1.2 conda（不推荐）

由于部分包依赖系统环境编译，推荐使用docker。假如由于机器等原因不能使用docker，也可以通过conda创建环境，然后pip安装需要的包，需自行解决依赖问题

```bash
pip install -r requirements.txt
```

但是通过pip安装deepspeed很有可能安装或者运行失败，[FAQ](https://github.com/LianjiaTech/BELLE/blob/main/train/docs/FAQ.md) 中给出了一些安装deepspeed的教程以及可能遇到的问题

## 2. 模型训练
- 微调参见[README_FT.md](README_FT.md)
- RLHF参见[README_RLHF.md](README_RLHF.md)