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
from torch.nn.functional import softmax

# model_name = "/root/autodl-tmp/meta-llama/Llama-3.1-8B"
# model_name = "/root/autodl-tmp/meta-llama/Qwen3-8B"
# model_name = "/root/autodl-tmp/meta-llama/Llama-3.2-3B"
# model_name = "/root/autodl-fs/transformer/gpt2-xl"
# model_name = "/root/autodl-fs/transformer/Qwen3-4B"
model_name = "/root/autodl-fs/transformer/Qwen3-1.7B"
# model_name = "/root/autodl-fs/transformer/pythia-2.8b"
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

def calculate_accuracy(predictions, references):
    correct = sum(p == r for p, r in zip(predictions, references))
    total = len(references)
    return correct / total if total > 0 else 0.0

from tqdm import tqdm  # 引入 tqdm
def clean_text(text: str) -> str:
    # 替换换行符为空格
    text = text.replace("\n", " ")
    # 去掉下划线、长横线、破折号等无效符号
    text = re.sub(r"[_\-–—….]+", " ", text)
    # 合并多个空格为一个空格
    text = re.sub(r"\s+", " ", text)
    # 去掉首尾空格
    return text.strip()

def calculate_confidence(logits):
    # 对logits应用softmax得到概率分布
    probs = softmax(logits, dim=-1)
    # 对每个token的概率取对数并求和
    log_probs = torch.log(probs)
    avg_log_prob = log_probs.mean()  # 平均对数概率
    confidence = torch.exp(avg_log_prob)  # 取指数得到置信度
    return confidence.item()  # 返回纯数值
def main(template, test_path, input_columns_name, output_columns_name,file):
    # load dataset
    combined_dataset = load_dataset("json", data_files={"test": test_path})
    test_dataset = combined_dataset["test"]
    labels = [item["label"] for item in test_dataset]
    # print(labels)
    # Construct the DatasetReader
    data = DatasetReader(combined_dataset, input_columns=input_columns_name, output_column=output_columns_name)
    ice = []

    predictions = []
    predict_predictions = []
    unpredict_predictions = []
    all_samples = []  # 存储所有样本信息
    predic_labels = []
    unpredic_labels = []
    all_output= []

    for idx in tqdm(range(len(test_dataset)), desc="Processing test samples", unit="sample"):
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.empty_cache()
        text = test_dataset[idx]["text"]
        true_label = test_dataset[idx]["label"]
        ice = test_dataset[idx]["ice"]
        predictable = test_dataset[idx]["predictable"]
        prompt = f"Solve the toxic detection task. Options for toxicity: benign, toxic.\n{ice}Text: {text} Prediction:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True , padding=True,max_length=1024)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            # 使用 model(**inputs) 进行推理
            outputs = model(**inputs)  # outputs 包含 logits

        # 提取 logits (模型输出的最后一层)
        logits = outputs.logits  # logits 的形状是 [batch_size, seq_len, vocab_size]
        last_token_logits = logits[:, -1, :]  # 获取最后一个 token 的 logits
        # 计算概率（通过 softmax）
        probs = torch.softmax(last_token_logits, dim=-1)  # 计算最后一个 token 的概率分布
        max_prob = torch.max(probs, dim=-1).values  # 获取最大概率
        # 直接使用最大概率作为置信度
        confidence = max_prob.item()  # 转换为 Python 数值（标量）
        # 获取预测的 label（通过 argmax 选择最大概率的类别）
        predicted_ids = torch.argmax(last_token_logits, dim=-1)  # 获取最大概率的 id
        predicted_label = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)  # 解码预测的 ID
        sentiment = predicted_label
        # print(f"Sentiment: {sentiment}")
        # print(f"Confidence: {confidence}")
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
        sample_dict["confidence"] = confidence  # 保存置信度
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
    with open(f"{file}ppl_data_knn3_0.1_classify_qwen.jsonl", "w",encoding="utf-8") as f:
        for item in all_output:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")



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
    task = ['implicit_hate', 'toxigen', 'hsaol']

    for task_item in task:

            for task_name in task_names:

                file = f"/"
                test_path = f"{file}ppl_data_knn3_0.1_classify_qwen.jsonl"
                main(templates[task_name], test_path,input_columns[task_name], output_columns[task_name],file)
