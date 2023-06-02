# Awesome-open-instruct-data-for-english [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

这是一个用于英文指令精调的 AWESOME 数据集合集。

通过使用指令进行微调，以提高 LLM（大型语言模型）的性能成为了一个趋势。随着以数据为中心的 AI 越来越受欢迎，我们需要更高质量的数据集来训练我们的模型。

在这里，你可以找到一些开源的英文指令数据的AWESOME 列表。

注：中英文混合的指令数据集算作为中文指令数据，详见中文指令数据的AWESOME 列表

## 目录
- [(InstructWild)|52K](https://github.com/XueFuzhao/InstructionWild)
- [(HC3)|24K](https://huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese)
- [(prosocial-dialog)|58K](https://huggingface.co/datasets/allenai/prosocial-dialog)
- [(ChatAlpaca)|20K](https://github.com/cascip/ChatAlpaca/tree/main)
- [(gpt4all)|437K](https://huggingface.co/datasets/nomic-ai/gpt4all_prompt_generations)
- [(gpt4all-j)|800K](https://huggingface.co/datasets/nomic-ai/gpt4all-j-prompt-generations)
- [(Anthropic/hh-rlhf)|170K](https://huggingface.co/datasets/Anthropic/hh-rlhf)
- [(LaMini-instruction)|2.58M](https://huggingface.co/datasets/MBZUAI/LaMini-instruction)
- [(oa_leet10k)|10K](https://huggingface.co/datasets/ehartford/oa_leet10k)
- [(dolly-v2)|15K](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
- [(sharegpt)|90K](https://huggingface.co/datasets/RyokoAI/ShareGPT52K/tree/main)
- [(stanford_alpaca)|52K](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json)
- [(UltraChat)|1.57M](https://huggingface.co/datasets/stingning/ultrachat)
- [(RefGPT-Fact)|50K](https://huggingface.co/datasets/Mutonix/RefGPT-Fact)
- [(OIG-small-chip2)|44M](https://github.com/LAION-AI/Open-Instruction-Generalist/tree/main/40M)
- [(OpenAssistant/oasst1)|88K](https://huggingface.co/datasets/OpenAssistant/oasst1)

## Datasets

 ## [(InstructWild)|52K](https://github.com/XueFuzhao/InstructionWild)

 - Summary:  the project aims to establish a more diverse and rich dataset of instructional commands. To do so, they collected 429 commands from ChatGPT screenshots and released English and Chinese versions. Following Alpaca's method, they generated 52,000 commands and their responses. They crawled over 700 noisy commands from Twitter and screened out useless ones. As a result, they selected 429 clean commands to ensure high quality. They collected the commands using a method similar to Alpaca's, but without the need for human intervention. Therefore, the generated prompts are more diverse and cover a wider range of topics. They provide 5 sample prompts to generate new commands using the OpenAI API. After collecting these prompts, they collected their respective instructions from the OpenAI API. The English and Chinese datasets were generated independently and cost a total of $880. The English dataset has 52K commands (about 24 million tokens), and the Chinese dataset has 52K commands as well.
 - Lincense: [Apache 2.0]
 - [InstructWild Dataset](https://drive.google.com/file/d/1OqfOUWYfrK6riE9erOx-Izp3nItfqz_K/view)

 ## [(HC3)|24K](https://huggingface.co/datasets/Hello-SimpleAI/HC3)

 - Summary:  In this work, we collected tens of thousands of comparison responses from both human experts and ChatGPT, with questions ranging from open-domain, financial, medical, legal, and psychological areas. The HC3 dataset is a valuable resource to analyze the linguistic and stylist characteristics of both humans and ChatGPT, which helps to investigate the future improvement directions for LLMs. They construct the comparison dataset mainly from two sources: Publicly available question-answering datasets, where answers are given by experts in specific domains or the high-voted answers by web users; the second is Wiki text; Based on the collected human question-answering datasets, they use ChatGPT to generate answers to these questions. To make the answer more aligned with human answers, they add additional instructions to ChatGPT for specific datasets. 
 - Lincense: [cc-by-sa-4.0]
 - [HC3 Dataset](https://huggingface.co/datasets/Hello-SimpleAI/HC3)

 ## [(prosocial-dialog)|58K](https://huggingface.co/datasets/allenai/prosocial-dialog)

 - Summary: ProsocialDialog is the first large-scale multi-turn English dialogue dataset to teach conversational agents to respond to problematic content following social norms. Covering diverse unethical, problematic, biased, and toxic situations, ProsocialDialog contains responses that encourage prosocial behavior, grounded in commonsense social rules (i.e., rules-of-thumb, RoTs). Created via a human-AI collaborative framework, ProsocialDialog consists of 58K dialogues, with 331K utterances, 160K unique RoTs, and 497K dialogue safety labels accompanied by free-form rationales.

 - Lincense: [cc-by-4.0]

 - [(prosocial-dialog Dataset)](https://huggingface.co/datasets/allenai/prosocial-dialog)

 ## [(ChatAlpaca)|20K](https://github.com/cascip/ChatAlpaca/tree/main)

 - Summary: ChatAlpaca is a chat dataset that aims to help researchers develop models for instruction-following in multi-turn conversations. The dataset is an extension of the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) data, which contains multi-turn instructions and their corresponding responses. ChatAlpaca is developed by Chinese Information Processing Laboratory at the Institute of Software, Chinese Academy of Sciences ([www.icip.org.cn](http://www.icip.org.cn/)).
 - Lincense: [apache-2.0]
 - [(ChatAlpaca Dataset)](https://github.com/cascip/ChatAlpaca/tree/main)

 ## [(gpt4all)|437K](https://huggingface.co/datasets/nomic-ai/gpt4all_prompt_generations)

 - Summary: We collected roughly one million promptresponse pairs using the GPT-3.5-Turbo OpenAI API between March 20, 2023 and March 26th, 2023. To do this, we first gathered a diverse sample of questions/prompts by leveraging three publicly available datasets. We chose to dedicate substantial attention to data preparation and curation based on commentary in the Stanford Alpaca project. Upon collection of the initial dataset of promptgeneration pairs, we loaded data into Atlas for data curation and cleaning. With Atlas, we removed all examples where GPT-3.5-Turbo failed to respond to prompts and produced malformed output. This reduced our total number of examples to 806,199 high-quality prompt-generation pairs. They use some filtering mechanisms exclusion produces a final subset containing 437,605 prompt-generation pairs.
 - Lincense: [apache-2.0]
 - [(gpt4all Dataset)](https://huggingface.co/datasets/nomic-ai/gpt4all_prompt_generations)

 ## [(gpt4all-j)|800K](https://huggingface.co/datasets/nomic-ai/gpt4all-j-prompt-generations)

 - Summary: GPT4All-J is an Apache-2 licensed chatbot trained over a massive curated corpus of assistant interactions including word problems, multi-turn dialogue, code, poems, songs, and stories. It builds on the March 2023 GPT4All. Building on the GPT4All dataset, we curated the GPT4All-J dataset by augmenting the original 400k GPT4All examples with new samples encompassing additional multi-turn QA samples and creative writing such as poetry, rap, and short stories.
 - Lincense: [apache-2.0]
 - [(gpt4all-j Dataset)](https://huggingface.co/datasets/nomic-ai/gpt4all-j-prompt-generations)

 ## [(Anthropic/hh-rlhf)|170K](https://huggingface.co/datasets/Anthropic/hh-rlhf)

 - Summary: This repository provides access to two different kinds of data:
   1. Human preference data about helpfulness and harmlessness from [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862). These data are meant to train preference (or reward) models for subsequent RLHF training. These data are *not* meant for supervised training of dialogue agents. Training dialogue agents on these data is likely to lead to harmful models and this shold be avoided.
   2. Human-generated and annotated red teaming dialogues from [Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned](https://www.anthropic.com/red_teaming.pdf). These data are meant to understand how crowdworkers red team models and what types of red team attacks are succesful or not. The data are *not* meant for fine-tuning or preference modeling (use the data above for preference modeling). These data are entire transcripts of conversations that are derived from the harmlessness preference modeling data described above, where only the chosen response is incorporated into the overall transcript. Furthermore, the transcripts are annotated with human and automated measurements of how harmful the overall dialogues are.
 - Lincense: [mit]
 - [(Anthropic/hh-rlhf Dataset)](https://huggingface.co/datasets/Anthropic/hh-rlhf)

 ## [(LaMini-instruction)|2.58M](https://huggingface.co/datasets/MBZUAI/LaMini-instruction)

 - Summary: The work distills the knowledge from large language models by performing sentence/offline distillation. We generate a total of 2.58M pairs of instructions and responses using [`gpt-3.5-turbo`](https://openai.com/api/) based on several existing resources of prompts, including [self-instruct](https://github.com/yizhongw/self-instruct), [P3](https://huggingface.co/datasets/bigscience/P3) , [FLAN](https://github.com/google-research/FLAN)  and [Alpaca](https://github.com/tatsu-lab/stanford_alpaca). They include only three random examples and a limited number of constraints within each prompt. Instead of explicitly specifying language restrictions, output length limitations, or instruction types, our instruction to gpt-3.5-turbo is to generate a variety of examples that align with the provided examples and adhere to the desired output format.  It is of concern that gpt-3.5-turbo may not possess the ability to generate diverse text without explicit guidance. To address this concern, we collect several common topics from Wikipedia to guide the generation process.
 - Lincense: [cc-by-nc 4.0]
 - [(LaMini-instruction Dataset)](https://huggingface.co/datasets/MBZUAI/LaMini-instruction)

 ## [(oa_leet10k)|10K](https://huggingface.co/datasets/ehartford/oa_leet10k)

 - Summary: LeetCode problems solved in multiple programming languages
 - Lincense: [apache-2.0]
 - [(oa_leet10k Dataset)](https://huggingface.co/datasets/ehartford/oa_leet10k)

 ## [(dolly-v2)|15K](https://huggingface.co/datasets/databricks/databricks-dolly-15k)

 - Summary: databricks-dolly-15k is an open source dataset of instruction-following records generated by thousands of Databricks employees in several of the behavioral categories outlined in the [InstructGPT](https://arxiv.org/abs/2203.02155) paper, including brainstorming, classification, closed QA, generation, information extraction, open QA, and summarization. Databricks employees were invited to create prompt / response pairs in each of eight different instruction categories, including the seven outlined in the InstructGPT paper, as well as an open-ended free-form category. The contributors were instructed to avoid using information from any source on the web with the exception of Wikipedia (for particular subsets of instruction categories), and explicitly instructed to avoid using generative AI in formulating instructions or responses. Examples of each behavior were provided to motivate the types of questions and instructions appropriate to each category.
 - Lincense: [cc-by-sa-3.0]
 - [(dolly-v2 Dataset)](https://huggingface.co/datasets/databricks/databricks-dolly-15k)

 ## [(sharegpt)|90K](https://huggingface.co/datasets/RyokoAI/ShareGPT52K/tree/main)

 - Summary:  This dataset is a collection of approximately 90,000 conversations scraped via the ShareGPT API before it was shut down. These conversations include both user prompts and responses from OpenAI's ChatGPT. This repository now contains the new 90K conversations version. The previous 52K may be found in the old directory. This dataset is expected to primarily consist of messages in English and other Western languages.
 - Lincense: [cc0-1.0]
 - [(ShareGPT Dataset)](https://huggingface.co/datasets/RyokoAI/ShareGPT52K/tree/main)

 ## [(stanford_alpaca)|52K](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json)

 - Summary:  They built on the data generation pipeline from [self-instruct](https://github.com/yizhongw/self-instruct) and made the following modifications:
   - Used `text-davinci-003` to generate the instruction data instead of `davinci`.
   - Wrote a new prompt (`prompt.txt`) that explicitly gave the requirement of instruction generation to `text-davinci-003`. 
   - Adopted much more aggressive batch decoding, i.e., generating 20 instructions at once, which significantly reduced the cost of data generation.
   - Simplified the data generation pipeline by discarding the difference between classification and non-classification instructions.
   - Only generated a single instance for each instruction
 - Lincense: [cc-by-nc 4.0]
 - [(stanford_alpaca Dataset)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json)

 ## [(UltraChat)|1.57M](https://huggingface.co/datasets/stingning/ultrachat)

 - Summary:  An open-source, large-scale, and multi-round dialogue data powered by Turbo APIs. In consideration of factors such as safeguarding privacy, we do not directly use any data available on the Internet as prompts. To ensure generation quality, two separate ChatGPT Turbo APIs are adopted in generation, where one plays the role of the user to generate queries and the other generates the response. We instruct the user model with carefully designed prompts to mimic human user behavior and call the two APIs iteratively. 
 - Lincense: [cc-by-nc 4.0]
 - [(UltraChat Dataset)](https://huggingface.co/datasets/stingning/ultrachat)

 ## [(RefGPT-Fact)|50K](https://huggingface.co/datasets/Mutonix/RefGPT-Fact)

 - Summary:  RefGPT-Fact is a datasets containing 100k multi-turn dialogues about factual knowledge with 50k English and 50k Chinese. The English version uses the English Wikipedia as the reference and the Chinese version uses the frequently-used Chinese online encyclopedia website, Baidu Baike.
 - Lincense: [apache-2.0]
 - [(RefGPT-Fact Dataset)](https://huggingface.co/datasets/Mutonix/RefGPT-Fact)

 ## [(OIG-small-chip2)|44M](https://github.com/LAION-AI/Open-Instruction-Generalist/tree/main/40M)

 - Summary:  The purpose of the larger dataset is to perform continued pre-training, followed by a finetune on the smaller high quality dataset. The purpose of the smaller OIG-small-chip2 dataset is to make it easy to convert a language model pretrained on large amounts of text into an instruction following model using a small amount of additional compute via finetuning or softprompt tuning.Many additional datasets are being prepared by various community members and will be incorporated into this dataset as we are able to verify the quality and formatting of the data. Our goal is to make helpful and non-toxic instruction tuned models available to everyone.OIG is currently at 44M. We will continue to publish ever larger diverse instruction datasets with the goal of creating 1 trillion tokens of diverse instructions - enough to pretrain an LLM from scratch.
 - Lincense: [apache-2.0]
 - [(OIG-small-chip2 Dataset)](https://github.com/LAION-AI/Open-Instruction-Generalist/tree/main/40M)

 ## [(OpenAssistant/oasst1)|88K](https://huggingface.co/datasets/OpenAssistant/oasst1)

 - Summary:  In an effort to democratize research on large-scale alignment, we release OpenAssistant Conversations (OASST1), a human-generated, human-annotated assistant-style conversation corpus consisting of 161,443 messages in 35 different languages, annotated with 461,292 quality ratings, resulting in over 10,000 fully annotated conversation trees. The corpus is a product of a worldwide crowd-sourcing effort involving over 13,500 volunteers. This dataset contains message trees. Each message tree has an initial prompt message as the root node, which can have multiple child messages as replies, and these child messages can have multiple replies. All messages have a role property: this can either be "assistant" or "prompter". The roles in conversation threads from prompt to leaf node strictly alternate between "prompter" and "assistant".
 - Lincense: [apache-2.0]
 - [(OpenAssistant/oasst1 Dataset)](https://huggingface.co/datasets/OpenAssistant/oasst1)

