import random
from collections import defaultdict
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, concatenate_datasets
import torch
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast
from sentence_transformers import SentenceTransformer
from torch import bfloat16
import json
from openicl.icl_dataset_reader import DatasetReader
from openicl.icl_prompt_template import PromptTemplate
from retriever.bert_retriever import ColBERTRetriever
import numpy as np

from retriever.bert_retriever_unpreict import ColBERTRetriever_unpre

# model_name = "/root/autodl-fs/google/gemma-2-2b"
# model_name = "/root/autodl-tmp/meta-llama/Llama-3.2-3B"
# model_name = "/root/autodl-fs/transformer/gpt2-xl"
# # model_name = "/root/autodl-fs/transformer/Qwen3-4B"
model_name = "/root/autodl-fs/transformer/Qwen3-1.7B"
# model_name = "/root/autodl-fs/transformer/pythia-2.8b"
# model_name = "/root/autodl-tmp/meta-llama/Llama-3.1-8B"
# model_name = "/root/autodl-tmp/meta-llama/Qwen3-8B"
# tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name,padding_side='left')
tokenizer = AutoTokenizer.from_pretrained(model_name,padding_side='left')
# model = LlamaForCausalLM.from_pretrained(model_name, ignore_mismatched_sizes=True).half()
model = AutoModelForCausalLM.from_pretrained(model_name, ignore_mismatched_sizes=True).half()
# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=bfloat16, trust_remote_code=True)

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

from tqdm import tqdm  # 引入 tqdm
from datasets import Dataset, DatasetDict
def format_predictions(idx_list, train_dataset):
    # 情感分析标签映射
    label_map = {0: 'benign', 1: 'toxic'}
    label_groups = defaultdict(list)
    for idx in idx_list:
        text = train_dataset[idx]["text"]
        label = train_dataset[idx]["predict_label"]
        label_groups[label].append((text, label))

    # 交错不同标签的样本
    formatted_predictions = []
    max_count = max(len(group) for group in label_groups.values())

    for i in range(max_count):
        for label in sorted(label_groups.keys()):  # 按标签顺序排列
            if i < len(label_groups[label]):
                text, label_val = label_groups[label][i]
                label_str = label_map[label_val]
                formatted_predictions.append(f"Text: {text} Prediction: {label_str}\n")

    return ''.join(formatted_predictions)
def main(task_item,test_path, input_columns_name, output_columns_name,ice_num,file):
    # load dataset
    train_samples = []
    test_samples = []
    input_file = f"{file}ppl_data_knn3_0.1_classify_qwen.jsonl"
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item.get("predictable",0) == 1:
                if item["confidence"] >= 0.7:
                    train_samples.append(item)
            else:
                test_samples.append(item)
    print("train size:", len(train_samples))
    combined_dataset = load_dataset("json", data_files={"test": test_path})
    test_dataset = combined_dataset["test"]
    labels = [item["label"] for item in test_dataset]
    # print(labels)
    # Construct the DatasetReader
    data = DatasetReader(combined_dataset, input_columns=input_columns_name, output_column=output_columns_name)
    ice = []
    train_dataset_unpredict = Dataset.from_list(train_samples)
    test_dataset_ubpredict = Dataset.from_list(test_samples)

    # 构建 combined_dataset
    combined_dataset_unpredict = DatasetDict({
        "train": train_dataset_unpredict,
        "test": test_dataset_ubpredict
    })

    data = DatasetReader(combined_dataset_unpredict, input_columns=input_columns_name, output_column=output_columns_name)
    bert_retriever = ColBERTRetriever_unpre(data, ice_num=ice_num, index_split='train', test_split='test')
    ice_idx_list = bert_retriever.retrieve()
    print(ice_idx_list)
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
        true_label = test_dataset[idx]["label"]
        ice = test_dataset[idx]["ice"]
        predictable = test_dataset[idx]["predictable"]
        if predictable == 1:
            ice_item = ice
        else:

            ice_item = format_predictions(ice_idx_list[unpredict_idx],train_dataset_unpredict)
            unpredict_idx += 1
        prompt = f"Solve the toxic detection task. Options for toxicity: benign, toxic.\n{ice_item}Text: {text} Prediction:"
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
        predicted_ids = torch.argmax(last_token_logits, dim=-1)  # 获取最大概率的 id
        predicted_label = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)  # 解码预测的 ID
        # 提取 sentiment 信息（假设预测结果包含 "Prediction:" 字符串）
        sentiment = predicted_label
        # print(sentiment)

        if 'benign' in sentiment:
            predictions.append(0)
            pred_label = 0
            # print(0)
        elif 'toxic' in sentiment:
            predictions.append(1)
            pred_label = 1
            # print(1)
        else:
            predictions.append(-1)
            pred_label = -1


        sample_dict = dict(test_dataset[idx])
        sample_dict["predict_label"] = pred_label
        all_output.append(sample_dict)
    # Save predictions to a JSON file
        if predictable == 1:
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
                     "amazon": ["text","ice","predictable"],
                     'td': ["text","ice","predictable"],
                     "nli": ['premise', 'hypothesis',"ice","predictable"]
                     }

    output_columns = {
                      "amazon": 'label',
                      'td': 'label',
                      "nli": 'label'
                      }

    test_split = {
        "amazon": 'test',
        'td': 'test',
        "nli": 'validation'
    }
    task_names = ['td']
    model_names = ['gpt2']
    task = ['implicit_hate', 'toxigen', 'hsaol']



    for task_item in task:
        for task_name in task_names:

                file = f"/"
                test_path = f"{file}ppl_data_knn3_0.1_classify_qwen.jsonl"
                ice_num = 3

                main(task_item, test_path,input_columns[task_name], output_columns[task_name],ice_num,file)
