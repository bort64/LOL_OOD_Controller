import json
import random

import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
import math
from tqdm import tqdm
import torch.nn.functional as F
from openicl.icl_dataset_reader import DatasetReader
from retriever.ner_bert_retriever import NerBERTRetriever

model_name = "/root/autodl-tmp/meta-llama/Llama-3.2-3B"
# model_name = "/root/autodl-tmp/meta-llama/Llama-3.1-8B"
# model_name = "/root/autodl-tmp/meta-llama/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(model_name, ignore_mismatched_sizes=True).half()

# 设置填充标记为结束标记
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

# 检查GPU是否可用
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def calculate_ppl(logits, labels, entity_start, entity_end):
    batch_size, seq_len, vocab_size = logits.size()
    entity_end = min(entity_end, seq_len)  # 防止越界

    shift_logits = logits[:, entity_start:entity_end - 1, :].contiguous()
    shift_labels = labels[:, entity_start + 1:entity_end].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        reduction='mean'
    )

    ppl = math.exp(loss.item())
    return ppl

def calculate_confidence(logits, entity_start, entity_end):
 
    entity_logits = logits[:, entity_start:entity_end, :]
    probabilities = F.softmax(entity_logits, dim=-1)
    confidence = probabilities.max(dim=-1).values.mean()  # 计算平均置信度
    return confidence

def format_predictions(idx_list, train_dataset):
    formatted_predictions = []
    for idx in idx_list:
        text = train_dataset[idx]["Text"]
        entity = train_dataset[idx]["Entity"]

        # 动态构建实体部分，只有存在的实体才会加入
        entity_parts = []

        if entity["organization"]:
            organization = ','.join(entity["organization"])
            entity_parts.append(f"{organization}")

        if entity["person"]:
            person = ','.join(entity["person"])
            entity_parts.append(f"{person}")

        if entity["location"]:
            location = ','.join(entity["location"])
            entity_parts.append(f"{location}")

        # 如果没有实体，则Entity部分为空；否则只包括存在的实体
        if entity_parts:
            formatted_predictions.append(f"Text: {text} Entities: {', '.join(entity_parts)}\n")
        else:
            formatted_predictions.append(f"Text: {text} Entities: \n")

    return ''.join(formatted_predictions)



# 主逻辑
def main(train_path, test_path, model_path, input_columns_name, output_columns_name, ice_num, candidate_num,
         select_time, batch_size, seed,task):
    # 加载数据集
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    combined_dataset = load_dataset("json", data_files={"train": train_path, "test": test_path})
    train_dataset = combined_dataset["train"]
    test_dataset = combined_dataset["test"]
    data = DatasetReader(combined_dataset, input_columns=input_columns_name, output_column=output_columns_name)
    bert_retriever = NerBERTRetriever(data, ice_num=ice_num, index_split='train', test_split='test')
    ice_idx_list = bert_retriever.retrieve()

    predictions = []
    total_ppl = 0.0
    total_confidence = 0.0
    count = 0.0
    results = []
    count_0 = 0

    # 遍历测试集
    for idx in tqdm(range(len(test_dataset)), desc="Processing test samples", unit="sample"):
        # 获取ICE样本
        torch.cuda.empty_cache()
        ice_item = format_predictions(ice_idx_list[idx], train_dataset)
        torch.cuda.empty_cache()

        text = test_dataset[idx]["Text"]
        entity = test_dataset[idx]["Entity"]
        entities = []
        for item in entity["organization"]:
            entities.append(item)
        for item in entity["person"]:
            entities.append(item)
        for item in entity["location"]:
            entities.append(item)

        prompt = f"Solve the NER task, identifying the Organization, Person, Location entities from given text.\n{ice_item}Text: {text} Entities:"
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        print(prompt)
        del prompt_ids

        # Tokenize inputs
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=1024)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        outputs = model(**inputs)
        logits = outputs.logits
        gen_ids = inputs["input_ids"]
        max_length = 12
        for _ in range(max_length):
            last_token_logits = logits[:, -1, :]
            next_token_id = torch.argmax(last_token_logits, dim=-1).unsqueeze(-1)
            gen_ids = torch.cat((gen_ids, next_token_id), dim=1)
            outputs = model(input_ids=gen_ids)
            logits = outputs.logits

        generated_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        print(generated_text)

        # 提取实体部分
        entity_start = generated_text.find("Entities:")
        if entity_start != -1:
            # 使用tokenizer来确定实体部分的start和end
            entity_text = generated_text[entity_start:]
            entity_lines = [line.strip() for line in entity_text.split('\n') if line.strip()]
            entity_lines = [line for line in entity_lines if "Entities:" in line]

            if len(entity_lines) >= 4:
                fourth_entity = entity_lines[3]  # 获取第四个实体
                entity_content = fourth_entity.split("Entities:", 1)[1].strip()
                entity_content_ids = tokenizer.encode(entity_content, add_special_tokens=False)
                print("entity_content",entity_content)
                # 计算重复率
                result_last = []
                parts = [part.strip() for part in entity_content.split(',')]
                result_last = parts

                set1 = set(entities)
                print(set1)

                set2 = set(result_last)
                print(set2)
                common_elements = set1 & set2
                repetition_rate = len(common_elements) / max(len(set1), len(set2))
                count += repetition_rate
                print(f"Repetition Rate: {repetition_rate:.2%}")
                results.append({
                    "ice": ice_item,
                    "text": text,
                    "Entity": entities,
                    "Predicted Entity": result_last,
                    "Repetition Rate": repetition_rate,
                })

    with open(f'random_ice.jsonl', 'w') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    accuracy = count / len(test_dataset)
    print(f"Accuracy: {accuracy:.2%}")


if __name__ == '__main__':
    model_path = "/root/autodl-tmp/meta-llama/Llama-3.2-3B"
    # model_path = "/root/autodl-fs/transformer/Qwen3-1.7B"
    # model_path = "/root/autodl-tmp/meta-llama/Llama-3.1-8B"
    # model_path = "/root/autodl-tmp/meta-llama/Qwen3-8B"
    input_columns_name = ['Text']
    output_columns_name = 'Entity'
    ice_num = 3
    candidate_num = 30
    select_time = 10
    batch_size = 8
    seed = 42
    task = ["wnut","ener","conll2003"]
    for task_name in task:
        train_path = f'train_100.jsonl'

        test_path = f'test.jsonl'

        main(train_path, test_path, model_path, input_columns_name, output_columns_name, ice_num, candidate_num,
         select_time, batch_size, seed, task_name)

