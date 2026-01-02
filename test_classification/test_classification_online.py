import random
from collections import defaultdict
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, concatenate_datasets, Dataset
import torch
from openicl.icl_dataset_reader import DatasetReader
from openicl.icl_prompt_template import PromptTemplate
import numpy as np
from retriever.bert_retriever_online import ColBERTRetriever_online

model_name = "/root/autodl-tmp/meta-llama/Llama-3.2-3B"
# model_name = "/root/autodl-fs/transformer/Qwen3-1.7B"
# model_name = "/root/autodl-tmp/meta-llama/Llama-3.1-8B"
# model_name = "/root/autodl-tmp/meta-llama/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name,padding_side='left')
model = AutoModelForCausalLM.from_pretrained(model_name, ignore_mismatched_sizes=True).half()

# 设置填充标记为结束标记
tokenizer.pad_token = tokenizer.eos_token  # 设置填充标记为结束标记（eos_token）
model.config.pad_token_id = tokenizer.eos_token_id  # 设置pad_token_id为eos_token_id

# 检查GPU是否可用
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)  # 将模型移到GPU或CPU


def clean_text(text: str) -> str:
    # 替换换行符为空格
    text = text.replace("\n", " ")
    # 去掉下划线、长横线、破折号等无效符号
    text = re.sub(r"[_\-–—….]+", " ", text)
    # 合并多个空格为一个空格
    text = re.sub(r"\s+", " ", text)
    # 去掉首尾空格
    return text.strip()


def calculate_accuracy(predictions, references):
    correct = sum(p == r for p, r in zip(predictions, references))
    total = len(references)
    return correct / total if total > 0 else 0.0

from tqdm import tqdm
def format_predictions(idx_list, train_dataset):
    # 情感分析标签映射
    # label_map = {0: 'negative', 1: 'positive', 2: 'neutral'}
    # label_map = {0: 'benign', 1: 'toxic'}
    label_map = {0: 'entailment', 1: 'contradiction', 2: 'neutral'}
    # 按标签分组样本
    label_groups = defaultdict(list)
    for idx in idx_list:
        # text = train_dataset[idx]["Text"].split(" ")[:80]
        # text = " ".join(text)
        Premise = clean_text(train_dataset[idx]["Premise"]).split(" ")[:64]
        Premise = " ".join(Premise)
        Hypothesis = clean_text(train_dataset[idx]["Hypothesis"]).split(" ")[:64]
        Hypothesis = " ".join(Hypothesis)
        label = train_dataset[idx]["Label"]
        text = f"Premise: {Premise} Hypothesis: {Hypothesis}"
        label_groups[label].append((text, label))

    # 交错不同标签的样本
    formatted_predictions = []
    max_count = max(len(group) for group in label_groups.values())

    for i in range(max_count):
        for label in sorted(label_groups.keys()):  # 按标签顺序排列
            if i < len(label_groups[label]):
                text, label_val = label_groups[label][i]
                label_str = label_map[label_val]
                # formatted_predictions.append(f"Text: {text} Prediction: {label_str}\n")
                formatted_predictions.append(f"{text} Prediction: {label_str}\n")

    return ''.join(formatted_predictions)
def main(task_item,train_path,test_path, input_columns_name, output_columns_name,ice_num,file,seed1):
    # load dataset
    test_dataset = load_dataset("json", data_files={"test": test_path})
    train_dataset = load_dataset("json", data_files={"train": train_path})
    train_dataset = train_dataset["train"]
    test_dataset = test_dataset["test"]
    labels = [item["label"] for item in test_dataset]
    candidate_num = 100
    random.seed(seed1)  # 42
    candidate_indices = random.sample(range(len(train_dataset)), candidate_num)
    candidate_pool = train_dataset.select(candidate_indices)

    print(candidate_indices)

    # 计算每类平均选取的数量

    data = DatasetReader(candidate_pool, input_columns=input_columns_name, output_column=output_columns_name)
    bert_retriever = ColBERTRetriever_online(data, ice_num=ice_num, index_split='train')
    ice = []

    predictions = []
    predict_predictions = []
    unpredict_predictions = []
    all_samples = []  # 存储所有样本信息
    predic_labels = []
    unpredic_labels = []
    all_output= []
    unpredict_idx = 0
    for idx in tqdm(range(len(test_dataset)), desc="Processing test samples", unit="sample"):
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.empty_cache()
        text = test_dataset[idx]["text"]
        # text1 = clean_text(test_dataset[idx]["premise"]).split(" ")[:64]
        # text1 = " ".join(text1)
        # text2 = clean_text(test_dataset[idx]["hypothesis"]).split(" ")[:64]
        # text2 = " ".join(text2)
        true_label = test_dataset[idx]["label"]
        ice = test_dataset[idx]["ice"]
        predictable = test_dataset[idx]["predictable"]
        if predictable == 1:
            ice_item = ice
        else:

            ice_idx_list = bert_retriever.retrieve_single(text)
            # ice_idx_list = bert_retriever.retrieve_single_nli(text1,text2)
            ice_item = format_predictions(ice_idx_list, candidate_pool)

            # print(ice_item)
        prompt = f"Solve the sentiment analysis task. Options for sentiment: negative, positive, neutral.\n{ice_item}Text: {text} Prediction:"
        # prompt = f"Solve the toxic detection task. Options for toxicity: benign, toxic.\n{ice_item}Text: {text} Prediction:"
        # prompt = f"Solve the NLI task. Options for entailment relationship: entailment, neutral, contradiction.\n{ice_item}Premise: {text1}  Hypothesis: {text2} Prediction:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True , padding=True,max_length=1024)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        # print("prompt:",prompt)
        with torch.no_grad():
            # 使用 model(**inputs) 进行推理
            outputs = model(**inputs)  # outputs 包含 logits
        # 提取 logits (模型输出的最后一层)
        logits = outputs.logits  # logits 的形状是 [batch_size, seq_len, vocab_size]
        # 假设是分类任务，我们可以选择最后一个 token 的 logits 来计算置信度
        last_token_logits = logits[:, -1, :]  # 获取最后一个 token 的 logits
        probs = torch.softmax(last_token_logits, dim=-1)  # 计算最后一个 token 的概率分布
        max_prob = torch.max(probs, dim=-1).values  # 获取最大概率
        # 直接使用最大概率作为置信度
        confidence = max_prob.item()  # 转换为 Python 数值（标量）
        predicted_ids = torch.argmax(last_token_logits, dim=-1)  # 获取最大概率的 id
        predicted_label = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)  # 解码预测的 ID
        # 提取 sentiment 信息（假设预测结果包含 "Prediction:" 字符串）
        sentiment = predicted_label
        # print(sentiment)

        if 'positive' in sentiment:
            predictions.append(1)
            pred_label = 1
            # print(1)
        elif 'negative' in sentiment:
            predictions.append(0)
            pred_label = 0
            # print(0)
        elif 'neutral' in sentiment:
            predictions.append(2)
            pred_label = 2
            # print(2)
        else:
            predictions.append(2)
            pred_label = 2
        # if 'benign' in sentiment:
        #     predictions.append(0)
        #     pred_label = 0
        #     # print(0)
        # elif 'toxic' in sentiment:
        #     predictions.append(1)
        #     pred_label = 1
        #     # print(1)
        # else:
        #     predictions.append(-1)
        #     pred_label = -1
        # if 'entail' in sentiment:
        #     predictions.append(0)
        #     pred_label = 0
        #     # print(0)
        # elif 'contradiction' in sentiment:
        #     predictions.append(1)
        #     pred_label = 1
        #     # print(1)
        # elif 'neutral' in sentiment:
        #     predictions.append(2)
        #     pred_label = 2
        #     # print(2)
        # else:
        #     predictions.append(-1)
        #     pred_label = -1

    # Save predictions to a JSON file
        if predictable == 1:
            if confidence >= 0.7:
                if pred_label == -1:
                    pred_label = 2
                new_item_online = {"Text": text, "Label": pred_label}
                # new_item_online = {"Premise": text1,"Hypothesis":text1, "Label": pred_label}
                bert_retriever.add_to_index(new_item_online)
                new_item = {"Text": [text], "Label": [pred_label]}
                # new_item = {"Premise": [text1],"Hypothesis":[text1], "Label": [pred_label]}
                candidate_pool = concatenate_datasets([
                    candidate_pool,
                    Dataset.from_dict(new_item)
                ])
            predic_labels.append(true_label)
            predict_predictions.append(pred_label)
        else:
            unpredic_labels.append(true_label)
            unpredict_predictions.append(pred_label)

        # 计算准确率
    accuracy = calculate_accuracy(predictions, labels)
    predct_accuracy = calculate_accuracy(predict_predictions, predic_labels)
    unpredct_accuracy = calculate_accuracy(unpredict_predictions, unpredic_labels)
    print(f"Accuracy: {accuracy}")
    print(f"predct_accuracy: {predct_accuracy}")
    print(f"unpredct_accuracy: {unpredct_accuracy}")



if __name__ == '__main__':
    import nltk
    amazon_tp_dict = {
        0: "</E>Text: </text> Prediction: negative",
        1: "</E>Text: </text> Prediction: positive",
        2: "</E>Text: </text> Prediction: neutral"
    }
    amazon_template = PromptTemplate(amazon_tp_dict, {'Text': '</text>'}, ice_token='</E>')

    td_tp_dict = {
        0: "</E>Text: </text> Prediction: benign",
        1: "</E>Text: </text> Prediction: toxic"
    }
    td_template = PromptTemplate(td_tp_dict, {'Text': '</text>'}, ice_token='</E>')

    nli_tp_dict = {
        0: "</E>Premise: </text1> Hypothesis:</text> Prediction: entailment",
        1: "</E>Premise: </text1> Hypothesis:</text> Prediction: contradiction",
        2: "</E>Premise: </text1> Hypothesis:</text> Prediction: neutral"
    }
    nli_template = PromptTemplate(nli_tp_dict, {'Premise': '</text1>', 'Hypothesis': '</text>'}, ice_token='</E>')
    templates = {
                 "amazon": amazon_template,
                 'td': td_template,
                 "nli": nli_template
                 }

    input_columns = {
                     "amazon": ["Text"],
                     'td': ["Text"],
                     "nli": ['Premise', 'Hypothesis']
                     }

    output_columns = {
                      "amazon": 'Label',
                      'td': 'Label',
                      "nli": 'Label'
                      }

    test_split = {
        "amazon": 'test',
        'td': 'test',
        "nli": 'validation'
    }
    task_names = ['amazon']
    model_names = ['gpt2']
    task = ['dynasent','semeval','sst5']
    # task = ['implicit_hate', 'toxigen', 'hsaol']
    # task = ['wanli', 'anli', 'contract_nli']
    seeds = [42]



    for task_item in task:
        for seed1 in seeds:
            for task_name in task_names:
                file = f""
                train_path = f"train.jsonl"
                test_path = f"{file}ppl_data_knn3_0.1_classify_qwen.jsonl"
                ice_num = 3
                main(task_item, train_path,test_path,input_columns[task_name], output_columns[task_name],ice_num,file,seed1)
