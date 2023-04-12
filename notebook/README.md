 
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
| [BELLE-GPTQ Colab推理](https://github.com/LianjiaTech/BELLE/blob/main/notebook/BELLE_INFER_COLAB.ipynb)  | 提供了BELLE 8bit量化的BLOOM模型在Colab运行的示例代码(由于模型对机器配置要求，暂时只有Colab Pro账户才能成功运行,正在进行优化使普通账户也能运行） ，该代码运行时内存最高消费20G，GPU显存只需要10G即可运行，当然你本地机器如果满足上述硬件条件，也可以将Notebook下载到本地运行。|[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LianjiaTech/BELLE/blob/main/notebook/BELLE_INFER_COLAB.ipynb)|


### The hardware requirements 
BELLE不同模型运行时，需要满足的最低硬件配置条件

| Model     |       Description      |     RAM      |  GPU |
|:----------|:-------------|:-------------|:-------------|
|bloom7b-2m-8bit-128g.pt | BELLE BLOOM 7B 200万训练数据版8bit量化后权重  |20G| 11G|




