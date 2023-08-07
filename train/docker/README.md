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
sudo docker pull tothemoon/belle:20230804
```
belle镜像中包含sshd，可以远程连接到容器内部

## 容器环境
```
Package                 Version
----------------------- -------------------------------
absl-py                 1.3.0
accelerate              0.22.0.dev0
aiofiles                23.1.0
aiohttp                 3.8.5
aiosignal               1.3.1
alembic                 1.11.1
altair                  5.0.1
anyio                   3.7.1
apex                    0.1
appdirs                 1.4.4
argon2-cffi             21.3.0
argon2-cffi-bindings    21.2.0
arrow                   1.2.3
asttokens               2.2.1
astunparse              1.6.3
async-timeout           4.0.2
attrs                   22.1.0
audioread               3.0.0
backcall                0.2.0
backoff                 2.2.1
beautifulsoup4          4.11.1
binaryornot             0.4.4
bitsandbytes            0.41.1
black                   23.7.0
bleach                  5.0.1
blis                    0.7.9
cachetools              5.2.0
catalogue               2.0.8
certifi                 2022.12.7
cffi                    1.15.1
chardet                 5.2.0
charset-normalizer      2.1.1
click                   8.1.3
cloudpickle             2.2.0
cmaes                   0.10.0
cmake                   3.24.1.1
colorlog                6.7.0
comm                    0.1.2
confection              0.0.3
contourpy               1.0.6
cookiecutter            1.7.3
cuda-python             11.7.0+0.g95a2041.dirty
cudf                    22.10.0a0+316.gad1ba132d2.dirty
cugraph                 22.10.0a0+113.g6bbdadf8.dirty
cuml                    22.10.0a0+56.g3a8dea659.dirty
cupy-cuda118            11.0.0
cycler                  0.11.0
cymem                   2.0.7
Cython                  0.29.32
dask                    2022.9.2
dask-cuda               22.10.0a0+23.g62a1ee8
dask-cudf               22.10.0a0+316.gad1ba132d2.dirty
datasets                2.14.3
dbus-python             1.2.16
debugpy                 1.6.4
decorator               5.1.1
deepspeed               0.10.0
defusedxml              0.7.1
dill                    0.3.4
distributed             2022.9.2
distro                  1.4.0
einops                  0.6.1
entrypoints             0.4
evaluate                0.4.0
exceptiongroup          1.0.4
execnet                 1.9.0
executing               1.2.0
expecttest              0.1.3
faiss-cpu               1.7.4
fastapi                 0.100.1
fastjsonschema          2.16.2
fastrlock               0.8.1
ffmpy                   0.3.1
filelock                3.12.2
flash-attn              2.0.4
fonttools               4.38.0
frozenlist              1.4.0
fsspec                  2022.11.0
gitdb                   4.0.10
GitPython               3.1.18
google-auth             2.15.0
google-auth-oauthlib    0.4.6
gql                     3.4.1
gradio                  3.39.0
gradio_client           0.3.0
graphql-core            3.2.3
graphsurgeon            0.4.6
greenlet                2.0.2
grpcio                  1.51.1
h11                     0.14.0
HeapDict                1.0.1
hf-doc-builder          0.4.0
hjson                   3.1.0
httpcore                0.17.3
httpx                   0.24.1
huggingface-hub         0.16.4
hypothesis              5.35.1
idna                    3.4
importlib-metadata      5.1.0
importlib-resources     5.10.1
iniconfig               1.1.1
intel-openmp            2021.4.0
ipykernel               6.19.2
ipython                 8.7.0
ipython-genutils        0.2.0
jedi                    0.18.2
Jinja2                  3.1.2
jinja2-time             0.2.0
joblib                  1.2.0
json5                   0.9.10
jsonschema              4.17.3
jupyter_client          7.4.8
jupyter_core            5.1.0
jupyter-tensorboard     0.2.0
jupyterlab              2.3.2
jupyterlab-pygments     0.2.2
jupyterlab-server       1.2.0
jupytext                1.14.4
kiwisolver              1.4.4
langcodes               3.3.0
librosa                 0.9.2
linkify-it-py           1.0.3
lit                     16.0.6
llvmlite                0.39.1
locket                  1.0.0
Mako                    1.2.4
Markdown                3.4.1
markdown-it-py          2.1.0
MarkupSafe              2.1.1
matplotlib              3.6.2
matplotlib-inline       0.1.6
mdit-py-plugins         0.3.3
mdurl                   0.1.2
mistune                 2.0.4
mkl                     2021.1.1
mkl-devel               2021.1.1
mkl-include             2021.1.1
mock                    4.0.3
mpmath                  1.2.1
msgpack                 1.0.4
multidict               6.0.4
multiprocess            0.70.12.2
murmurhash              1.0.9
mypy-extensions         1.0.0
nbclient                0.7.2
nbconvert               7.2.6
nbformat                5.7.0
nest-asyncio            1.5.6
networkx                2.6.3
ninja                   1.11.1
nltk                    3.8.1
notebook                6.4.10
numba                   0.56.4
numpy                   1.22.2
nvidia-dali-cuda110     1.20.0
nvidia-pyindex          1.0.9
nvtx                    0.2.5
oauthlib                3.2.2
onnx                    1.12.0
opencv                  4.6.0
optuna                  3.2.0
orjson                  3.9.2
packaging               22.0
pandas                  1.5.2
pandocfilters           1.5.0
parameterized           0.9.0
parso                   0.8.3
partd                   1.3.0
pathspec                0.11.2
pathy                   0.10.1
peft                    0.4.0
pexpect                 4.8.0
pickleshare             0.7.5
Pillow                  9.2.0
pip                     23.2.1
pkgutil_resolve_name    1.3.10
platformdirs            2.6.0
pluggy                  1.0.0
polygraphy              0.43.1
pooch                   1.6.0
portalocker             2.0.0
poyo                    0.5.0
preshed                 3.0.8
prettytable             3.5.0
prometheus-client       0.15.0
prompt-toolkit          3.0.36
protobuf                3.20.1
psutil                  5.9.4
ptyprocess              0.7.0
pudb                    2022.1.3
pure-eval               0.2.2
py-cpuinfo              9.0.0
pyarrow                 9.0.0
pyasn1                  0.4.8
pyasn1-modules          0.2.8
pybind11                2.10.1
pycocotools             2.0+nv0.7.1
pycparser               2.21
pydantic                1.10.12
pydub                   0.25.1
Pygments                2.13.0
PyGObject               3.36.0
pylibcugraph            22.10.0a0+113.g6bbdadf8.dirty
pylibraft               22.10.0a0+81.g08abc72.dirty
pynvml                  11.4.1
pyparsing               3.0.9
pyre-extensions         0.0.29
pyrsistent              0.19.2
pytest                  7.2.0
pytest-rerunfailures    10.3
pytest-shard            0.1.2
pytest-timeout          2.1.0
pytest-xdist            3.1.0
python-dateutil         2.8.2
python-hostlist         1.22
python-multipart        0.0.6
python-slugify          8.0.1
pytorch-quantization    2.1.2
pytz                    2022.6
PyYAML                  6.0
pyzmq                   24.0.1
raft-dask               22.10.0a0+81.g08abc72.dirty
regex                   2022.10.31
requests                2.28.1
requests-oauthlib       1.3.1
requests-toolbelt       0.10.1
resampy                 0.4.2
responses               0.18.0
rjieba                  0.1.11
rmm                     22.10.0a0+38.ge043158.dirty
rouge-score             0.1.2
rsa                     4.9
sacrebleu               1.5.1
sacremoses              0.0.53
safetensors             0.3.1
scikit-learn            0.24.2
scipy                   1.6.3
semantic-version        2.10.0
Send2Trash              1.8.0
sentencepiece           0.1.99
setuptools              59.5.0
six                     1.16.0
smart-open              6.3.0
smmap                   5.0.0
sniffio                 1.3.0
sortedcontainers        2.4.0
soundfile               0.11.0
soupsieve               2.3.2.post1
spacy                   3.4.4
spacy-legacy            3.0.10
spacy-loggers           1.0.4
sphinx-glpi-theme       0.3
SQLAlchemy              2.0.19
srsly                   2.4.5
ssh-import-id           5.10
stack-data              0.6.2
starlette               0.27.0
sympy                   1.11.1
tbb                     2021.7.1
tblib                   1.7.0
tensorboard             2.9.0
tensorboard-data-server 0.6.1
tensorboard-plugin-wit  1.8.1
tensorrt                8.5.1.7
terminado               0.17.1
text-unidecode          1.3
thinc                   8.1.5
threadpoolctl           3.1.0
timeout-decorator       0.5.0
tinycss2                1.2.1
tokenizers              0.13.3
toml                    0.10.2
tomli                   2.0.1
toolz                   0.12.0
torch                   2.0.1+cu118
torchaudio              2.0.2+cu118
torchtext               0.13.0a0+fae8e8c
torchvision             0.15.2+cu118
tornado                 6.1
tqdm                    4.64.1
traitlets               5.7.1
transformers            4.32.0.dev0
treelite                2.4.0
treelite-runtime        2.4.0
triton                  2.0.0
typer                   0.7.0
typing_extensions       4.7.1
typing-inspect          0.9.0
uc-micro-py             1.0.2
ucx-py                  0.27.0a0+29.ge9e81f8
uff                     0.6.9
urllib3                 1.26.13
urwid                   2.1.2
urwid-readline          0.13
uvicorn                 0.23.2
wasabi                  0.10.1
wcwidth                 0.2.5
webencodings            0.5.1
websockets              11.0.3
Werkzeug                2.2.2
wheel                   0.38.4
xdoctest                1.0.2
xformers                0.0.20
xgboost                 1.6.2
xxhash                  3.3.0
yarl                    1.9.2
zict                    2.2.0
zipp                    3.11.0
```

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