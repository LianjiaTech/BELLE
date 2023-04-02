import os,json
import random
os.makedirs("belle_open_source_data", exist_ok=True)

def split_data(file_name):
    data = []
    with open(f"belle_open_source_data/{file_name}.json") as f:
        lines = f.readlines()
        for line in lines:
            data.append(json.loads(line))

    print(len(data))
    random.shuffle(data)
    dev_rate = 0.01
    dev_num = int(len(data) * dev_rate)
    train_data = data[:-dev_num]
    dev_data = data[-dev_num:]
    print(len(train_data), len(dev_data))
    with open(f"belle_open_source_data/{file_name}.train.json", "w") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(f"belle_open_source_data/{file_name}.dev.json", "w") as f:
        for item in dev_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if not os.path.exists("belle_open_source_data/Belle_open_source_1M.json"):
    os.system("wget -P belle_open_source_data/ https://huggingface.co/datasets/BelleGroup/train_1M_CN/resolve/main/Belle_open_source_1M.json")
    split_data("Belle_open_source_1M")


if not os.path.exists("belle_open_source_data/Belle_open_source_0.5M.json"):
    os.system("wget -P belle_open_source_data/ https://huggingface.co/datasets/BelleGroup/train_0.5M_CN/resolve/main/Belle_open_source_0.5M.json")
    split_data("Belle_open_source_0.5M")