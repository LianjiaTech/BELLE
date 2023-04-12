# ChatBELLE 跨平台应用

[中文README](README.md)


# ChatBELLE Cross-platform App
A minimal, cross-platform LLM chat app with [BELLE](https://github.com/LianjiaTech/BELLE) using quantized on-device offline models and Flutter UI, running on macOS (done), Windows, Android, iOS(see [Known Issues](#known-issues)) and more.


## App Downloading
Please refer to [Releases](https://github.com/LianjiaTech/BELLE/releases/tag/v0.95).

Downloading and usage for different platforms: [Usage](#Usage).

Only macOS supported by now. More platforms coming soon!


## Model Downloading
[ChatBELLE-int4][please waiting, we will release asap]


## Model Quantization
Utilizes [llama.cpp's 4bit quantization](https://github.com/ggerganov/llama.cpp) to optimize on-device inferencing speed and RAM occupation. Quantization leads to accuracy loss and model performance degradation. 4-bit quantization trades accuracy for model size, our current 4-bit model sees significant performance gap compared with fp32 or fp16 ones and is just for users to take a try. With better algorithms being developed and more powerful chips landing on mobile devices, we believe on-device model performance will thrive and will keep a close track on this.

### GPTQ
[GPTQ](https://github.com/IST-DASLab/gptq) employs one-shot quantization to achieve lower accuracy loss or higher model compression rate. We will keep track of this line of work.


## Roadmap
* More devices
* Multiround chat
* Model selection
* Chat history
* Chat list


## Usage

### macOS
Recommend using M1/M2 series CPU with 16GB RAM to have the best experience. If you encounter slow inference, try closing other apps to release more memory. Inference on 8G RAM will be very slow.
Intel CPUs could possibly run as well (not tested) but could be very slow.

* Download [chatbelle.dmg](https://github.com/LianjiaTech/BELLE/releases/download/v0.95/chatbelle.dmg) from [Releases](https://github.com/LianjiaTech/BELLE/releases/tag/v0.95) page, double click to open it, then drag `Chat Belle.dmg` into `Applications` folder.
* Open the `Chat Belle` app in `Applications` folder by right click then Ctrl-click `Open`, then click `Open`.
* The app will prompt the intended model file path and fail to load the model. Close the app.
* Download quantized model from [ChatBELLE-int4](https://huggingface.co/BelleGroup/ChatBELLE-int4/blob/main/belle-model.bin).
* Move and rename the model to the path prompted by the app. Defaults to `~/Library/Containers/com.barius.chatbelle/Data/belle-model.bin` .
* Reopen the app again (double clicking is now OK).

### Windows
* Stay tuned

### Android
* Stay tuned

### iOS
* Stay tuned


## Known Issues
* On macOS devices with 8GB RAM, inference is really slow due to constant swapping. 16GB RAM devices might see the same slowdown if RAM occupation by other applications is high.
* Inferencing on Macs with Intel chips is slow.
* The 3GB App RAM constraint on iOS devices won't allow even the smallest model (~4.3G) from loading. [Reference](https://github.com/mikeger/llama-ios)


## Disclaimer
This program is for learning and research purposes only. The devs take no responsibilities in any damage caused by using or distributing this program.


## Thanks
* LLaMa model inferencing code uses [llama.cpp](https://github.com/ggerganov/llama.cpp)
* Flutter chat UI uses [flyer.chat](https://github.com/flyerhq/flutter_chat_ui)
