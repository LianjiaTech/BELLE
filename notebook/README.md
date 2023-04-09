 
<!---
Copyright 2023 The BELLE Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

#  BELLE Notebooks

你可以在这里找到BELLE的官方Notebooks实现列表


### Documentation notebooks
你可以在colab里面打开这里任何一个notebooks (notebook最上面提供了打开colab的按钮) ，你也可以在下面链接打开它们:

| Notebook     |      Description      |  Colab |
|:----------|:-------------|:-------------|
| [BELLE-GPTQ Colab推理](https://github.com/LianjiaTech/BELLE/blob/main/notebook/BELLE_INFER_COLAB.ipynb)  | 提供了BELLE 4 bit量化的LLAMA7B模型在Colab运行的示例代码 ，该代码运行时内存只需要7G，GPU显存只需要6G即可运行，所以Colab普通账号即可完成BELLE模型推理。当然你本地机器如果满足上述硬件条件，也可以将Notebook下载到本地运行。|[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LianjiaTech/BELLE/blob/main/notebook/BELLE_INFER_COLAB.ipynb)|

### The hardware requirements 
BELLE不同模型运行时，需要满足的最低硬件配置条件

| Model     |       Description      |     RAM      |  GPU |
|:----------|:-------------|:-------------|:-------------|
|llama7b-2m-4bit-128g.pt | BELLE llama7B 200万训练数据版4bit量化后权重  |7G| 6G|
|llama7b-2m-8bit-128g.pt | BELLE llama7B 200万训练数据版8bit量化后权重  |16G| 11G|
|BELLE-LLAMA-7B-2M | BELLE llama7B 200万训练数据版  |32G| 28G|



