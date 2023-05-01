import os
import os
import glob
import sys
import json

from OSCAR2301 import Oscar2301


OUTPUT_FILE_EACH_ITEMS = 100000
OUTPUT_DIR = 'OSCAR-2301/unpack'


dataset = Oscar2301(language='zh')
doc_files = glob.glob('OSCAR-2301/zh_meta/*.zst')
print(dataset)

fp_out = None
for idx, item in dataset._generate_examples(doc_files):
    # print(item)
    # if idx > 100:
    #     break
    if fp_out is None:
        out_path = os.path.join(OUTPUT_DIR, f'output_{int(idx/OUTPUT_FILE_EACH_ITEMS):05d}.jsonl')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        print(out_path)
        fp_out = open(out_path, 'a')
    fp_out.write(json.dumps(item, ensure_ascii=False) + '\n')
    if (idx + 1) % OUTPUT_FILE_EACH_ITEMS == 0:
        fp_out.close()
        fp_out = None