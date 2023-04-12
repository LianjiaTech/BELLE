# ChatBELLE 跨平台应用

[English Version](README_en.md)

基于[BELLE](https://github.com/LianjiaTech/BELLE)模型的跨平台离线大语言模型交谈App。使用量化后的离线端上模型配合Flutter，可在macOS（已支持）、Windows、Android、iOS(参考[Known Issues](#known-issues))等设备上运行。

下图是一个可以使用App在设备端本地运行4bit量化的BELLE-7B模型，在M1 Max CPU上实时运行的效果（未加速）：

<img src="./chatbelle-demo.gif"></img>


## App下载

请见[Releases](https://github.com/LianjiaTech/BELLE/releases/tag/v0.95)。

各平台对应下载&使用说明请见[使用说明](#使用说明)。

目前仅支持macOS。更多平台即将发布！

## 模型下载

[ChatBELLE-int4] [正在更新中，第一时间同步]

**需要先首先执行ChatBELLE app，会建好一个文件夹```~/Library/Containers/com.barius.chatbelle```。然后将下载好的模型重命名并移动至app显示的路径。默认为~/Library/Containers/com.barius.chatbelle/Data/belle-model.bin。**

## 模型量化
使用[llama.cpp的4bit量化](https://github.com/ggerganov/llama.cpp)优化设备端离线推理的速度和内存占用。量化会带来计算精度的损失，影响模型的生成效果。4bit是比较激进的量化方式，目前的4bit模型效果相比fp32和fp16还有明显差距，仅供尝试。随着模型算法的发展和设备端算力的演进，我们相信离线推理的效果会有很大改善，我们也会持续跟进。

### GPTQ
[GPTQ](https://github.com/IST-DASLab/gptq)使用one-shot量化方式来获得更小的量化损失或更高的压缩率。我们将持续跟进基于GPTQ的设备端量化模型。


## 路线图
* 更多设备
* 多轮对话
* 模型选择
* 聊天历史
* 聊天列表


## 使用说明

### macOS
建议使用M1/M2系列芯片配合16G RAM以获得最佳体验。如果推理速度过慢，可能是内存不足，可以尝试关闭其他app以释放内存。8G内存会非常慢。
Intel芯片理论上也可以跑，但是速度较慢。

* 下载[Releases](https://github.com/LianjiaTech/BELLE/releases/tag/v0.95)中的[chatbelle.dmg](https://github.com/LianjiaTech/BELLE/releases/download/v0.95/chatbelle.dmg)，双击打开，把`Chat Belle.dmg`左键拖进`应用程序`文件夹中。
* 右键`应用程序`文件夹中的`Chat Belle`App，按住Ctrl并左键单击`打开`，点`打开`。
* App会显示模型加载失败，并显示模型路径。关闭App。
* 下载量化后的模型[ChatBELLE-int4][正在更新中，第一时间同步]。
* 移动并重命名模型至app显示的路径。默认为`~/Library/Containers/com.barius.chatbelle/Data/belle-model.bin`。
* 重新打开App（直接双击）。

### Windows
* 敬请期待

### Android
* 敬请期待

### iOS
* 敬请期待


## 已知问题
* 推理在8GB内存的macOS设备上会非常慢，原因是内存不足导致疯狂swapping。16GB内存的设备在内存占用较高的情况下也可能遇到同样状况。
* 推理在Intel芯片的Mac设备上比较慢。
* iOS的3GB App内存限制导致最小模型(~4.3G)也无法加载。[参考](https://github.com/mikeger/llama-ios)


## 免责声明
本程序仅供学习、研究使用，因使用、传播本程序带来的任何损害，本程序的开发者不负任何责任。


## 致谢
* LLaMa模型设备端推理 [llama.cpp](https://github.com/ggerganov/llama.cpp)
* Flutter聊天UI [flyer.chat](https://github.com/flyerhq/flutter_chat_ui)

