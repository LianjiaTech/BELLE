# docker镜像
## 国内环境自行替换代理
```bash
export https_proxy=...
export http_proxy=...
export all_proxy=...
```

## 构建镜像
```bash
sudo bash build_dockerfile_upon_transfermers.sh
```

## 上传镜像到dockerhub
```bash
sudo bash upload_image.sh
```

## 下载镜像
已经构建好镜像，无需自行构建
```bash
sudo docker tothemoon/belle:20230728
```
belle镜像中包含sshd，可以远程连接到容器内部

## 运行镜像
### 1. 参考[nvidia安装说明](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)安装container-toolkit
### 2. 自行创建ssh密钥
### 3. 运行容器
```bash
sudo bash docker_run.sh
```
`docker_run.sh`文件主要内容如下
```
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    --network host \
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