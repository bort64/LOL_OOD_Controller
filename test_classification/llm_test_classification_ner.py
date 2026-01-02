import json
import random
import re

import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
import math
from tqdm import tqdm
import torch.nn.functional as F
from openicl.icl_dataset_reader import DatasetReader
import requests
import time


def calculate_ppl(logits, entity_start, entity_end):
    # 确保 entity_end 不超过 logits 的最大长度
    entity_end = min(entity_end, logits.size(1))  # 如果实体部分超出，限制为 logits 的长度
    loss = torch.tensor(0.0, device=logits.device)  # 初始化为 Tensor 类型

    for t in range(entity_start, entity_end):  # 只计算Entity部分的tokens
        token_logits = logits[:, t, :]  # 获取当前token的logits
        token_id = torch.argmax(token_logits, dim=-1)  # 获取当前token的预测id

        # 计算交叉熵损失，使用mean减少损失的积累
        loss += F.cross_entropy(token_logits, token_id, reduction='mean')

    # 计算困惑度，使用平均损失
    ppl = math.exp(loss.item()/(entity_end - entity_start))  # 困惑度 = e^avg_loss

    return ppl  # 返回困惑度
# 计算置信度
def calculate_confidence(logits, entity_start, entity_end):
    # 计算Entity部分的置信度
    entity_logits = logits[:, entity_start:entity_end, :]
    probabilities = F.softmax(entity_logits, dim=-1)
    confidence = probabilities.max(dim=-1).values.mean()  # 计算平均置信度
    return confidence

def format_predictions(idx_list, train_dataset):
    formatted_predictions = []
    for idx in idx_list:
        text = train_dataset[idx]["text"]
        entity = train_dataset[idx]["Entity"]

        # 动态构建实体部分，只有存在的实体才会加入
        entity_parts = []

        if entity["organization"]:
            organization = ','.join(entity["organization"])
            entity_parts.append(f"Organization: {organization}")
        else:
            entity_parts.append(f"Organization: None")

        if entity["person"]:
            person = ','.join(entity["person"])
            entity_parts.append(f"Person: {person}")
        else:
            entity_parts.append(f"Person: None")

        if entity["location"]:
            location = ','.join(entity["location"])
            entity_parts.append(f"Location: {location}")
        else:
            entity_parts.append(f"Location: None")

        # 如果没有实体，则Entity部分为空；否则只包括存在的实体
        if entity_parts:
            formatted_predictions.append(f"Text: {text} Entities: {', '.join(entity_parts)}\n")
        else:
            formatted_predictions.append(f"Text: {text} Entities: \n")

    return ''.join(formatted_predictions)

# 主逻辑
def main(test_path, input_columns_name, output_columns_name, ice_num, candidate_num,
         select_time, batch_size, seed,task_name):
    # 加载数据集
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    combined_dataset = load_dataset("json", data_files={"test": test_path})
    test_dataset = combined_dataset["test"]
    data = DatasetReader(combined_dataset, input_columns=input_columns_name, output_column=output_columns_name)
    total_confidence = 0.0
    count = 0.0
    results = []
    unpredict_predictions = []
    predict_rate = 0
    predict_cnt = 0
    unpredict_rate = 0
    unpredict_cnt = 0

    # 遍历测试集
    for idx in tqdm(range(len(test_dataset)), desc="Processing test samples", unit="sample"):
        # torch.cuda.empty_cache()

        text = test_dataset[idx]["text"]
        entity = test_dataset[idx]["Entity"]
        ice_item = test_dataset[idx]["ice"]
        predictable = test_dataset[idx]["predictable"]

        entities = []
        for item in entity:
            entities.append(item)
        prompt = f"Solve the NER task, identifying the Organization, Person, Location entities from given text.\n{ice_item}Text: {text} Entities:"
        entity_content = chat(prompt)
        result_last = []
        matches = list(re.finditer(r'(Person|Organization|Location)\s*:', entity_content))

        for i, match in enumerate(matches):
            key = match.group(1)
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(entity_content)
            value_str = entity_content[start:end].strip()
            values = re.split(r'[，,;]', value_str)
            values = [v.strip() for v in values if v.strip()]
            if key == 'Organization':
                prefix = 'O'
            elif key == 'Person':
                prefix = 'P'
            elif key == 'Location':
                prefix = 'L'
            else:
                continue

            for v in values:
                if v and v.lower() != 'none':
                    result_last.append(f"{prefix}-{v}")


        set1 = set(entities)
        set2 = set(result_last)
        print(set1)
        print(set2)
        common_elements = {s.lower().replace(' ','').replace('.','').replace('_', '') for s in set1} & {s.lower().replace(' ','').replace('.','').replace('_', '') for s in set2}
        repetition_rate = len(common_elements) / max(len(set1), len(set2))
        count += repetition_rate
        print(repetition_rate)
        results.append({
                    "ice": ice_item,
                    "text": text,
                    "Entity": entities,
                    "Predicted Entity": result_last,
                    "Repetition Rate": repetition_rate,
                    "Confidence": 0.9,
                    "predictable": predictable
                })
        if predictable == 1:
                predict_rate += repetition_rate
                predict_cnt += 1
        else:
                unpredict_rate += repetition_rate
                unpredict_cnt += 1

    # 计算平均PPL和置信度
    with open(f'', 'w') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    accuracy = count / len(test_dataset)
    pre_avg = predict_rate / predict_cnt
    unpre_avg = unpredict_rate / unpredict_cnt
    print(f"Accuracy: {accuracy:.7}")
    print(f"Predict Accuracy: {pre_avg:.7}")
    print(f"Unpredict Accuracy: {unpre_avg:.7}")
    # print(f"Average Confidence: {avg_confidence:.2f}")
    print(len(test_dataset))
    print(predict_cnt)
    print(unpredict_cnt)


if __name__ == '__main__':
    task = ["wnut","ener","music"]
    for task_name in task:
        test_path = f''
        input_columns_name = ['Text']
        output_columns_name = 'Entity'
        ice_num = 3
        candidate_num = 30
        select_time = 10
        batch_size = 8
        seed = 42

        main(test_path, input_columns_name, output_columns_name, ice_num, candidate_num,
         select_time, batch_size, seed,task_name)
