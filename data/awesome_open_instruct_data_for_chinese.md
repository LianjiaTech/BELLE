# awesome-open-instruct-data-for-chinese [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

这是一个用于中文指令精调的 AWESOME 数据集合集。

通过使用指令进行微调，以提高 LLM（大型语言模型）的性能成为了一个趋势。随着以数据为中心的 AI 越来越受欢迎，我们需要更高质量的数据集来训练我们的模型。

在这里，你可以找到一些开源的中文指令数据的AWESOME 列表。

## 目录

- [(JosephusCheung/GuanacoDataset)|534K](https://huggingface.co/datasets/JosephusCheung/GuanacoDataset)
- [(COIG)|191K](https://huggingface.co/datasets/BAAI/COIG/tree/main)
- [(Firefly)|1.1M](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M)
- [(HC3-Chinese)|13K](https://huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese)
- [(alpaca_gpt4_zh)|52K](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/blob/main/data/alpaca_gpt4_data_zh.json)
- [(pCLUE)|1.2M](https://github.com/CLUEbenchmark/pCLUE)
- [(CSL)|396K](https://github.com/ydli-ai/CSL)
- [(MOSS)|0.6M](https://github.com/OpenLMLab/MOSS/tree/main/SFT_data)
- [(Safety-Prompts)|100K](https://github.com/thu-coai/Safety-Prompts)
- [(oa_leet10k)|10K](https://huggingface.co/datasets/ehartford/oa_leet10k)
- [(RefGPT-Fact-zh)|50K](https://huggingface.co/datasets/Mutonix/RefGPT-Fact)

## Datasets

 ## [(JosephusCheung/GuanacoDataset)|534K](https://huggingface.co/datasets/JosephusCheung/GuanacoDataset)

 - The dataset for the Guanaco model is designed to enhance the multilingual capabilities and address various linguistic tasks. It builds upon the 175 tasks from the Alpaca model by providing rewrites of seed tasks in different languages and adding new tasks specifically designed for English grammar analysis, natural language understanding, cross-lingual self-awareness, and explicit content recognition. The dataset comprises a total of 534,530 entries. 

 ## [(COIG)|191K](https://huggingface.co/datasets/BAAI/COIG/tree/main)

 - COIG project provides diverse Chinese instruction corpora. Researchers can contribute to the corpus set and collaborate. COIG releases first chip to aid Chinese LLMs' development and encourages more researchers to join. Includes translated, exam, human value alignment, counterfactual correction, and leetcode instruction corpora.

 ## [(Firefly)|1.1M](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M)

 - Summary: they have constructed a large collection of data related to Chinese culture, comprising 23 commonly used Chinese datasets. For each task, multiple instruction templates were manually written by them to ensure the quality and richness of the data, resulting in a training set of 1.15 million Chinese language samples. The tasks covered include couplet writing, poetry composition, Classical Chinese translation, prose writing, Jin Yong's novels, and others. For each task, multiple human-written instruction templates were used by them to ensure the high quality and diversity of the data.
- Lincense: [Apache 2.0]
- [Firefly Dataset](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M)

 ## [(InstructWild)|52K](https://github.com/XueFuzhao/InstructionWild)

 - Summary:  the project aims to establish a more diverse and rich dataset of instructional commands. To do so, they collected 429 commands from ChatGPT screenshots and released English and Chinese versions. Following Alpaca's method, they generated 52,000 commands and their responses. They crawled over 700 noisy commands from Twitter and screened out useless ones. As a result, they selected 429 clean commands to ensure high quality. They collected the commands using a method similar to Alpaca's, but without the need for human intervention. Therefore, the generated prompts are more diverse and cover a wider range of topics. They provide 5 sample prompts to generate new commands using the OpenAI API. After collecting these prompts, they collected their respective instructions from the OpenAI API. The English and Chinese datasets were generated independently and cost a total of $880. The English dataset has 52K commands (about 24 million tokens), and the Chinese dataset has 52K commands as well.
 - Lincense: [Apache 2.0]
 - [InstructWild Dataset](https://drive.google.com/file/d/1OqfOUWYfrK6riE9erOx-Izp3nItfqz_K/view)



 ## [(HC3-Chinese)|13K](https://huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese)

 - Summary:  In this work, we collected tens of thousands of comparison responses from both human experts and ChatGPT, with questions ranging from open-domain, financial, medical, legal, and psychological areas. The HC3 dataset is a valuable resource to analyze the linguistic and stylist characteristics of both humans and ChatGPT, which helps to investigate the future improvement directions for LLMs. They construct the comparison dataset mainly from two sources: Publicly available question-answering datasets, where answers are given by experts in specific domains or the high-voted answers by web users; the second is Wiki text; Based on the collected human question-answering datasets, they use ChatGPT to generate answers to these questions. To make the answer more aligned with human answers, they add additional instructions to ChatGPT for specific datasets. 
 - Lincense: [cc-by-sa-4.0]
 - [HC3 Dataset](https://huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese)



 ## [(alpaca_gpt4_zh)|52K](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/blob/main/data/alpaca_gpt4_data_zh.json)

 - Summary: This project present the first attempt to use GPT-4 to generate instruction following data for LLM finetuning. They also collect feedback and comparison data from GPT-4 to enable a comprehensive evaluation and reward model training. The dataset contains 52K instruction-following data generated by GPT-4 with Alpaca prompts translated into Chinese by ChatGPT. Experiments on instruction-tuned LLaMA models show that the 52K English and Chinese instruction-following data generated by GPT-4 leads to superior zero-shot performance on new tasks to the instruction-following data generated by previous state-of-the-art models.
 - Lincense: [cc-by-nc-4.0]
 - [alpaca_gpt4_zh Dataset](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/blob/main/data/alpaca_gpt4_data_zh.json)



 ## [(pCLUE)|1.2M](https://github.com/CLUEbenchmark/pCLUE)

 - Summary:  This is a large-scale pre-training dataset based on prompts for multi-task and zero-shot learning. The dataset contains 1.2 million training samples and 73 prompts, covering 9 datasets: Text classification (tnews), Text classification (iflytek), Natural language inference (ocnli), Semantic matching (afqmc), Coreference resolution (cluewsc2020), Keyword recognition (csl), Reading comprehension - free-form (c3), Reading comprehension - extractive (cmrc2018), Reading comprehension - idiom filling (chid).

 - [pCLUE Dataset](https://github.com/CLUEbenchmark/pCLUE)



 ## [(CSL)|396K](https://github.com/ydli-ai/CSL)

 - Summary:  Scientific literature serves as a high-quality corpus, supporting a lot of Natural Language Processing (NLP) research. However, existing datasets are centered around the English language, which restricts the development of Chinese scientific NLP. In this work, we present CSL, a large-scale Chinese Scientific Literature dataset, which contains the titles, abstracts, keywords and academic fields of 396k papers. To our knowledge, CSL is the first scientific document dataset in Chinese. The CSL can serve as a Chinese corpus. Also, this semi-structured data is a natural annotation that can constitute many supervised NLP tasks. Based on CSL, we present a benchmark to evaluate the performance of models across scientific domain tasks, i.e., summarization, keyword generation and text classification

 - [CSL Dataset](https://drive.google.com/file/d/1xEDgtqHU4qm0Sp-dKjc5KerAmWydmh3-/view)

 ## [(MOSS)|0.6M](https://github.com/OpenLMLab/MOSS/tree/main/SFT_data)

 - Summary: MOSS is an open-sourced plugin-augmented conversational language model. The multi-turn conversational data (moss-002-sft-data) used to train MOSS-002, covering helpfulness, honesty, and harmlessness. The data is consisting of 570K English and 590K Chinese conversations generated by text-davinci-003. We open-sourced a small portion of other data(moss-003-sft-data, moss-003-sft-plugin-data, moss-003-pm-data) and will make public the full data in the near future. 

 - Lincense: [cc-by-nc 4.0]

 - [(MOSS SFT Dataset)](https://github.com/OpenLMLab/MOSS/tree/main/SFT_data)

 ## [(Safety-Prompts)|100K](https://github.com/thu-coai/Safety-Prompts)

 - Summary: Chinese safety prompts for evaluating and improving the safety of LLMs. The benchmark explores the comprehensive safety performance of LLMs from two perspectives: 8 kinds of typical safety scenarios and 6 types of more challenging instruction attacks.
 - Lincense: [apache-2.0]
 - [(Safety-Prompts Dataset)](https://github.com/thu-coai/Safety-Prompts)

 ## [(oa_leet10k)|10K](https://huggingface.co/datasets/ehartford/oa_leet10k)

 - Summary: LeetCode problems solved in multiple programming languages
 - Lincense: [apache-2.0]
 - [(oa_leet10k Dataset)](https://huggingface.co/datasets/ehartford/oa_leet10k)



 ## [(RefGPT-Fact-zh)|50K](https://huggingface.co/datasets/Mutonix/RefGPT-Fact/viewer/Mutonix--RefGPT-Fact/zh)

 - Summary:  RefGPT-Fact is a datasets containing 100k multi-turn dialogues about factual knowledge with 50k English and 50k Chinese. The English version uses the English Wikipedia as the reference and the Chinese version uses the frequently-used Chinese online encyclopedia website, Baidu Baike.
 - Lincense: [apache-2.0]
 - [(RefGPT-Fact-zh Dataset)](https://huggingface.co/datasets/Mutonix/RefGPT-Fact/viewer/Mutonix--RefGPT-Fact/zh)

