# 需要先安装container-toolkit
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

export https_proxy=...
export http_proxy=...
export all_proxy=...

belle_path=...
docker_user=...
tag=...
hf_home="/.../.cache"
ssh_pub_key="/home/.../.ssh/id_rsa.pub"
workdir="$belle_path/train"
chown root:root $ssh_pub_key

# docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
#     --network host \
#     --env HTTP_PROXY=$http_proxy \
#     --env HF_HOME=$hf_home \
#     -it --rm \
#     -v $belle_path:$belle_path \
#     -v $hf_home:$hf_home \
#     -v $ssh_pub_key:/root/.ssh/authorized_keys \
#     -w $workdir \
#     $docker_user/transformers:ds_$tag \
#     /bin/bash

docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    --network host \
    --env https_proxy=$https_proxy \
    --env http_proxy=$http_proxy \
    --env all_proxy=$all_proxy \
    --env HF_HOME=$hf_home \
    -it --rm \
    --name belle \
    -v $belle_path:$belle_path \
    -v $hf_home:$hf_home \
    -v $ssh_pub_key:/root/.ssh/authorized_keys \
    -w $workdir \
    $docker_user/belle:$tag \
    --sshd_port 2201 --cmd "echo 'Hello, world!' && /bin/bash"