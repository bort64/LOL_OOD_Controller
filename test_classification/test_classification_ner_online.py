import json
import random
from datasets import Dataset, DatasetDict
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
import math
from tqdm import tqdm
import torch.nn.functional as F
from datasets import Dataset, Features, Sequence, Value
from datasets import load_dataset, concatenate_datasets, Dataset
from openicl.icl_dataset_reader import DatasetReader
from retriever.ner_bert_retriever import NerBERTRetriever
from retriever.ner_bert_retriever_online import NerBERTRetriever_unpre_online
from retriever.ner_bert_retriever_unpredict import NerBERTRetriever_unpre

# 加载模型
# model_name = "/root/autodl-tmp/meta-llama/Llama-3.2-3B"
model_name = "/root/autodl-fs/transformer/Qwen3-1.7B"
# model_name = "/root/autodl-tmp/meta-llama/Llama-3.1-8B"
# model_name = "/root/autodl-fs/transformer/Qwen3-1.7B"
# model_name = "/root/autodl-tmp/meta-llama/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(model_name, ignore_mismatched_sizes=True).half()

# 设置填充标记为结束标记
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

# 检查GPU是否可用
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
file =''
input_file = f"{file}knn3_ppl_none_0.1_classify.jsonl"
train_samples = []
test_samples = []


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


import re
def format_predictions_train(idx_list, train_dataset):
    formatted_predictions = []
    for idx in idx_list:
        sample = train_dataset[idx]
        text = sample["Text"]
        entity_dict = sample["Entity"]
        if idx>1000:
            print("entity_dict:",entity_dict)

        # 从字典中提取各类实体
        organization_entities = entity_dict.get("organization", [])
        person_entities = entity_dict.get("person", [])
        location_entities = entity_dict.get("location", [])

        # 构建实体部分（允许为空）
        entity_parts = []
        entity_parts.append(f"Location: {', '.join(location_entities) if location_entities else 'None'}")
        entity_parts.append(f"Organization: {', '.join(organization_entities) if organization_entities else 'None'}")
        entity_parts.append(f"Person: {', '.join(person_entities) if person_entities else 'None'}")
        # entity_parts.append(f"Location: {', '.join(location_entities) if location_entities else 'None'}")

        # 拼接最终格式
        formatted_predictions.append(f"Text: {text} Entities: {'; '.join(entity_parts)}\n")

    return ''.join(formatted_predictions)
def clean_list(x):
    if x is None:
        return []
    # 过滤掉 None，确保全是字符串
    return [str(i) for i in x if i not in (None, "None")]

def format_predictions(idx_list, train_dataset):
    formatted_predictions = []
    for idx in idx_list:
        text = train_dataset[idx]["Text"]
        entities = train_dataset[idx]["Entity"]  # 获取实体

        # 确保 entities 是字符串列表，并进行分类
        organization_entities = []
        person_entities = []
        location_entities = []

        for entity in entities:
            if isinstance(entity, str):  # 确保是字符串
                if entity.startswith("O-"):  # 以 "O-" 开头为 Organization
                    organization_entities.append(re.sub(r"^O-", "", entity))  # 去掉 "O-" 前缀
                elif entity.startswith("P-"):  # 以 "P-" 开头为 Person
                    person_entities.append(re.sub(r"^P-", "", entity))  # 去掉 "P-" 前缀
                elif entity.startswith("L-"):  # 以 "L-" 开头为 Location
                    location_entities.append(re.sub(r"^L-", "", entity))  # 去掉 "L-" 前缀

        # 动态构建实体部分
        entity_parts = []

        if organization_entities:
            organization = ','.join(organization_entities)
            entity_parts.append(f"Organization: {organization}")
        else :
            entity_parts.append(f"Organization: None")

        if person_entities:
            person = ','.join(person_entities)
            entity_parts.append(f"Person: {person}")
        else :
            entity_parts.append(f"Person: None")

        if location_entities:
            location = ','.join(location_entities)
            entity_parts.append(f"Location: {location}")
        else :
            entity_parts.append(f"Location: None")

        # 如果没有实体，则Entity部分为空；否则只包括存在的实体
        if entity_parts:
            formatted_predictions.append(f"Text: {text} Entities: {'; '.join(entity_parts)}\n")
        else:
            formatted_predictions.append(f"Text: {text} Entities: \n")

    return ''.join(formatted_predictions)

def convert_list_entities_to_dict(entity_list):
    org, person, loc = [], [], []
    for ent in entity_list:
        if ent.startswith("O-"):
            org.append(ent[2:])
        elif ent.startswith("P-"):
            person.append(ent[2:])
        elif ent.startswith("L-"):
            loc.append(ent[2:])
    return {
        "organization": org,
        "person": person,
        "location": loc
    }

embed_model = SentenceTransformer("/root/autodl-fs/sentence-transformer/all-mpnet-base-v2", device=device)

def retrieve_most_similar(query_text, corpus_embeddings, train_dataset, top_k=3):
    """
    从训练集检索与 query_text 最相似的样本，要求选中的样本实体非空。
    """
    query_emb = embed_model.encode(query_text, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_emb, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=len(cos_scores))  # 先排序所有样本

    selected_ids = []
    for idx in top_results.indices.tolist():
        entity_dict = train_dataset[idx]["Entity"]
        # 检查实体是否非空
        has_entity = any(len(v) > 0 for v in entity_dict.values())
        if has_entity:
            selected_ids.append(idx)
        if len(selected_ids) >= top_k:
            break

    # 如果所有都空，则随便选前几个防止报错
    if not selected_ids:
        selected_ids = top_results.indices[:top_k].tolist()

    return selected_ids
# 主逻辑
def main(train_path,test_path, model_path, input_columns_name, output_columns_name, ice_num, candidate_num,
         select_time, batch_size, seed):
    # 加载数据集
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    combined_dataset = load_dataset("json", data_files={"test": test_path})
    train_dataset = load_dataset("json", data_files={"train": train_path})
    train_dataset = train_dataset["train"]
    corpus = [sample["text"] if "text" in sample else sample["Text"] for sample in train_dataset]
    corpus_embeddings = embed_model.encode(corpus, convert_to_tensor=True, show_progress_bar=True)
    test_dataset = combined_dataset["test"]
    candidate_num = 1000
    candidate_indices = random.sample(range(len(train_dataset)), candidate_num)
    candidate_pool = train_dataset.select(candidate_indices)
    data = DatasetReader(candidate_pool, input_columns=input_columns_name, output_column=output_columns_name)
    bert_retriever = NerBERTRetriever_unpre_online(data, ice_num=ice_num, index_split='train')

    count = 0.0
    predict_rate = 0
    predict_cnt = 0
    unpredict_rate = 0
    unpredict_cnt = 0
    # 遍历测试集
    for idx in tqdm(range(len(test_dataset)), desc="Processing test samples", unit="sample"):
        # 获取ICE样本
        # torch.cuda.empty_cache()
        text = test_dataset[idx]["text"]
        entity = test_dataset[idx]["Entity"]
        ice_item = test_dataset[idx]["ice"]
        predictable = test_dataset[idx]["predictable"]

        entities = []
        for item in entity:
            entities.append(item)
        print("++++++++++++++++++++++++++++++++++++++++++")
        if predictable == 1:
            ice = ice_item
        else:
            sim_ids = retrieve_most_similar(text, corpus_embeddings,train_dataset, top_k=1)
            ice_from_train = format_predictions_train(sim_ids, train_dataset)
            ice_list = bert_retriever.retrieve_multi_all_single(text)
            print(ice_list)
            ice = format_predictions_train(ice_list, candidate_pool)
            ice = ice+ice_from_train
        prompt = f"Solve the NER task, identifying the Organization, Person, Location entities from given text.\n{ice}Text: {text} Entities:"
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        l1 = len(prompt_ids)
        del prompt_ids
        if predictable==0:
            print("prompt",prompt)
        # Tokenize inputs
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=1024)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
                base_outputs = model(**inputs)
                base_logits = base_outputs.logits  # [batch, prompt_len, vocab]

        # 用 generate 生成新tokens，同时保存生成部分的scores
        with torch.no_grad():
                generation_output = model.generate(
                    **inputs,
                    max_new_tokens=30,
                    do_sample=False,  # 贪心解码，不用采样
                    temperature=None,  # 确保不触发 sampling 参数检查
                    top_p=None,
                    pad_token_id=model.config.pad_token_id,  # 显式指定 pad_token_id
                    eos_token_id=tokenizer.eos_token_id,  # 显式指定结束符
                    return_dict_in_generate=True,
                    output_scores=True
                )


        gen_ids = generation_output.sequences
        generated_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        # print("generated_text:", generated_text)

        # 把 scores 转成 logits 并拼接到 base_logits 后面
        if generation_output.scores is not None:
            step_logits = torch.stack(generation_output.scores, dim=1)  # [batch, step, vocab]
            logits_full = torch.cat([base_logits, step_logits], dim=1)  # [batch, prompt+gen, vocab]
        else:
            logits_full = base_logits
        # print(logits_full)
        # print(generated_text)

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
                l2 = len(entity_content_ids)
                entity_start_token_id = l1
                entity_end_token_id = l1 + l2

                # confidence = calculate_confidence(logits, entity_start_token_id, entity_end_token_id)
                confidence = calculate_confidence(logits_full, entity_start_token_id, entity_end_token_id)
                print("entity_content:",entity_content)
                # 计算重复率
                result_last = []
                matches = list(re.finditer(r'(Person|Organization|Location)\s*:', entity_content))

                for i, match in enumerate(matches):
                    key = match.group(1)
                    start = match.end()
                    end = matches[i + 1].start() if i + 1 < len(matches) else len(entity_content)

                    # 截取本类实体的值部分
                    value_str = entity_content[start:end].strip()

                    # 按逗号 / 中文逗号切分
                    # values = re.split(r'[，,;]', value_str)
                    values = re.split(r'[，,;]|\.\sand\s', value_str)
                    values = [v.strip() for v in values if v.strip()]

                    # 映射前缀
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
                print(set1)

                set2 = set(result_last)
                print(set2)
                # common_elements = set1 & set2
                # common_elements = {s.lower() for s in set1} & {s.lower() for s in set2}
                common_elements = {s.lower().replace(' ','').replace('.','').replace('_', '').replace(' \'', '') for s in set1} & {s.lower().replace(' ','').replace('.','').replace('_', '').replace(' \'', '') for s in set2}
                # common_elements = {s.lower().replace(' ','').replace('.','') for s in set1} & {s.lower().replace(' ','').replace('.','')for s in set2}

                # common_elements = {normalize(s) for s in set1} & {normalize(s) for s in set2}

                repetition_rate = len(common_elements) / max(len(set1), len(set2))

                count += repetition_rate
                print("repetition_rate_pre", test_dataset[idx]["Repetition Rate"])
                print("repetition_rate_unpre", repetition_rate)
                print("++++++++++++++++++++++++++++++++++++++++++")
                if predictable == 1:
                    if confidence >= 0.9:
                        entity_dict = convert_list_entities_to_dict(result_last)
                        new_item = {
                            "Text": text,
                            "Entity": {
                                "organization": clean_list(entity_dict.get("organization")),
                                "person": clean_list(entity_dict.get("person")),
                                "location": clean_list(entity_dict.get("location")),
                            }
                        }
                        print(new_item)
                        train_schema = Features({
                            "Text": Value("string"),
                            "Entity": {
                                "organization": Sequence(Value("string")),
                                "person": Sequence(Value("string")),
                                "location": Sequence(Value("string")),
                            }
                        })

                        new_ds = Dataset.from_list([new_item], features=train_schema)
                        # new_ds = Dataset.from_list([new_item])
                        print(new_ds)
                        bert_retriever.add_to_index(new_item)

                        candidate_pool = concatenate_datasets([candidate_pool, new_ds])
                        print(len(candidate_pool))
                    predict_rate += repetition_rate
                    predict_cnt += 1
                else:
                    unpredict_rate += repetition_rate
                    unpredict_cnt += 1

    accuracy = count / len(test_dataset)
    pre_avg = predict_rate / predict_cnt
    unpre_avg = unpredict_rate / unpredict_cnt
    print(f"Accuracy: {accuracy/100:.7%}")
    print(f"Predict Accuracy: {pre_avg/100:.7%}")
    print(f"Unpredict Accuracy: {unpre_avg/100:.7%}")
    # print(f"Average Confidence: {avg_confidence:.2f}")


if __name__ == '__main__':
    train_path = f'train_100.jsonl'
    test_path = f'{file}knn3_ppl_none_0.1_classify_qwen.jsonl'
    # model_path = "/root/autodl-tmp/meta-llama/Llama-3.2-3B"
    model_path = "/root/autodl-fs/transformer/Qwen3-1.7B"
    # model_path = "/root/autodl-tmp/meta-llama/Llama-3.1-8B"
    # model_path = "/root/autodl-tmp/meta-llama/Qwen3-8B"
    input_columns_name = ['Text']
    output_columns_name = 'Entity'
    ice_num = 2
    candidate_num = 30
    select_time = 10
    batch_size = 8
    seed = 42

    main(train_path,test_path, model_path, input_columns_name, output_columns_name, ice_num, candidate_num,
         select_time, batch_size, seed)
