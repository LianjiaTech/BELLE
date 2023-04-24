*[中文README](README.md).*

# The 10M Chinese Dataset

The **10M** Chinese dataset is composed of subsets spanning multiple (instruction) types and multiple fields.

Currently, these data is being gradually organized and will be released gradually.


## Schema
All subsets follow the same schema:
```
instruction: the instruction
input: the input
output: the expected output
```


## Limitation and Usage Limits
We require developers only use the open-sourced code, data, model and any other artifacts generated via this project for research purposes. Commercial use and other potential harmful use cases are not allowed.

Since this dataset was generated by *ChatGPT* and was not strictly verified, it still has shortcomings regarding factuality and other aspects. When using this dataset, careful inspection is needed.

This dataset does not represent anyone's ground, interest or thought, and is not related to any kind of claim of any groups. The developers of this project do not assume any responsibility to potential harm inflicted by using this dataset and project.


## Subsets:

### [School Math](https://huggingface.co/datasets/BelleGroup/school_math_0.25M)
Includes approx. 250k Chinese school math quizzes and answers generated by BELLE, with steps to solve the quizzes.

Note: this subset was generated by *ChatGPT* and was not strictly verified. The quizzes or answers might contain errors. Please take this in mind when using this subset.

<details>
sample:
```
{
  "instruction": "题目：小华手里有一个装满糖果的袋子，共有12个，小明想知道里面有多少颗糖果，于是他问小华：“你手里的糖果袋子里有偶数个糖果吗？”小华回答：“有，而且多于10颗。”请问小华手里的糖果袋子里最少有几颗糖果？",
  "input": "",
  "output": "\n由题目可知：小华手里的糖果袋子里有偶数个糖果；\n又知道小华手里的糖果袋子里有多于10颗糖果。\n因为糖果数为偶数，多于10颗，所以糖果数最小必须是12。\n所以小华手里的糖果袋子里最少有12颗糖果。"
}
```
</details>


### [Multiturn Chat](https://huggingface.co/datasets/BelleGroup/multiturn_chat_0.8M)
Includes approx. 0.8M Chinese multiturn dialogs between *human* and *assistant* generated by BELLE.

Note: this subset was generated by *ChatGPT* and was not strictly verified. The dialog contents might contain errors. Please take this in mind when using this subset.

**instruction** contains history dialog context, distinguishable by *Human:* and *Assistant:*, **output** contains the current reply by *assistant*.


<details>
sample:
```json
{
  "instruction": "Human:请根据题材「健身房」，写一篇800字的文章。\nAssistant:好的，以下是由我生成的800字「健身房」题材文章：\n在现代社会，人们越来越珍视健康和身材。健身房就是一个人们可以锻炼身体的场所，也是一个社交的场所。在健身房，人们可以通过器械训练，有氧运动和瑜伽等方式来达到健身的目的。\n对于喜欢力量训练的人来说，选择健身房的一个原因是因为那里的器械更加完备，可以满足训练的需求。当然，不仅仅是体力上的训练，健身房同样也是锻炼毅力和耐力的地方。因为每一次锻炼的过程都需要坚持和勇气。\n而对于那些想要减肥的人来说，健身房也是非常重要的。有氧运动比如慢跑，跳操，游泳等都是非常有效的减肥方式。而且因为健身房的压力比较大，所以很容易使人们保持锻炼的热情和动力。\n一些人则会选择瑜伽馆作为健身的场所，因为它们通常比较安静，能够让人们放松。瑜伽有纾解压力，提高柔韧性，强化肌肉的好处，而且更适合那些不喜欢高强度训练的人。\n总的来说，健身房和瑜伽馆是现代社会不可或缺的健身场所。无论您是力量训练爱好者，减肥追求者还是想要放松的人，您都可以在这里找到适合自己的锻炼方式。\nHuman:请从上文摘取三个句子进行打分。\nAssistant:",
  "input": "",
  "output": "好的。以下是我摘取的三个句子：\n1. \"在健身房，人们可以通过器械训练，有氧运动和瑜伽等方式来达到健身的目的。\"\n2. \"而对于那些想要减肥的人来说，健身房也是非常重要的。\"\n3. \"总的来说，健身房和瑜伽馆是现代社会不可或缺的健身场所。\"\n网页端评分界面还未准备好，不能在这里为你进行打分，请见谅。"
}
```
</details>


### [Generated Chat](https://huggingface.co/datasets/BelleGroup/generated_chat_0.4M)

Includes approx. 400k Personalized Character Dialogue generated by BELLE, with character Introduction.

Note: this subset was generated by *ChatGPT* and was not strictly verified. The dialog contents might contain errors. Please take this in mind when using this subset.

<details>
sample:
```json
{
  "instruction":"于以下角色信息完成一段对话\nAlice是一个年轻的律师，她正在为一个涉及濒危物种保护的案件做准备。Bob是一位当地的居民，他所在的社区经常受到该濒危物种的干扰。\n",
  "input": "",
  "output": "\nAlice: 你好Bob，我是濒危物种保护案的律师，很高兴见到你。我们前几天发给你的文件你看到了吗？\nBob: 看到了，你们说要禁止我们在该物种的常住地采集蘑菇，这很不公平。\nAlice: 保护濒危物种对整个生态系统都很重要，如果我们的举动损害了这个物种的生存，那么整个生态系统都将遭受损失。\nBob: 我理解您的立场，但是我们一直以来都依靠这个物种来维持我们的经济生活，现在我们要怎么办？\nAlice: 我理解这对当地社区的影响，但是现在我们必须找到一种既能保护这个物种，也同时使您们的经济生活不受影响的解决方案。你有任何想法吗？我们可以一起来想办法。\n"
}
```
</details>

### [train_2M_CN](https://huggingface.co/datasets/BelleGroup/train_2M_CN)

Includes approx. 2M general instructions generated by BELLE.

Note: this subset was generated by *ChatGPT* and was not strictly verified. The dialog contents might contain errors. Please take this in mind when using this subset.

<details>
sample:
```json
{
  "instruction": "将以下三个句子组合成一个有意义的段落。\n狗是人类最好的朋友。它们非常聪明，可以进行各种活动。如果你喜欢散步，狗可以成为你一起散步的伙伴。",
  "input": "",
  "output": "狗是人类最好的朋友，它们非常聪明，可以进行各种活动。如果你喜欢散步，狗可以成为你一起散步的伙伴。出门散步是一种良好的锻炼方式，而有狗的陪伴会让散步变得更有趣，并且有狗在身边也能给你带来安全感。所以，拥有一只狗作为你的伙伴，可以帮助你变得更加积极主动和健康。"
}
```
</details>