import random
from collections import defaultdict
import re
import math
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
import torch.nn.functional as F

# model_name = "/root/autodl-fs/google/gemma-2-2b"
model_name = "/root/autodl-tmp/meta-llama/Llama-3.2-3B"
# model_name = "/root/autodl-tmp/meta-llama/Llama-3.1-8B"
# model_name = "/root/autodl-tmp/meta-llama/Qwen3-8B"
# model_name = "/root/autodl-fs/transformer/gpt2-xl"
# model_name = "/root/autodl-fs/transformer/Qwen3-4B"
# model_name = "/root/autodl-fs/transformer/Qwen3-1.7B"
# model_name = "/root/autodl-fs/transformer/pythia-2.8b"
tokenizer = AutoTokenizer.from_pretrained(model_name,padding_side='left')
model = AutoModelForCausalLM.from_pretrained(model_name, ignore_mismatched_sizes=True).half()
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def calculate_accuracy(predictions, references):
    correct = sum(p == r for p, r in zip(predictions, references))
    total = len(references)
    return correct / total if total > 0 else 0.0

def format_predictions(idx_list, train_dataset):
    label_map = {0: 'negative', 1: 'positive', 2: 'neutral'}
    # label_map = {0: 'benign', 1: 'toxic'}
    # label_map = {0: 'entailment', 1: 'contradiction', 2: 'neutral'}
    label_groups = defaultdict(list)
    for idx in idx_list:
        text = train_dataset[idx]["Text"].split(" ")[:80]
        text = " ".join(text)
        # Premise = clean_text(train_dataset[idx]["Premise"]).split(" ")[:64]
        # Premise = " ".join(Premise)
        # Hypothesis = clean_text(train_dataset[idx]["Hypothesis"]).split(" ")[:64]
        # Hypothesis = " ".join(Hypothesis)
        label = train_dataset[idx]["Label"]
        # text = f"Premise: {Premise} Hypothesis: {Hypothesis}"
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
                # formatted_predictions.append(f"{text} Prediction: {label_str}\n")

    return ''.join(formatted_predictions)
from tqdm import tqdm  # 引入 tqdm

def main(template, train_path, test_path, model_path, sentence_model_path, input_columns_name, output_columns_name,
         ice_num, candidate_num, select_time, batch_size, seed, task):

    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # load dataset
    combined_dataset = load_dataset("json", data_files={"train": train_path, "test": test_path})
    train_dataset = combined_dataset["train"]
    test_dataset = combined_dataset["test"]
    # train_dataset = train_dataset.map(lambda x: {"Text": clean_text(x["Text"])})
    # test_dataset = test_dataset.map(lambda x: {"Text": clean_text(x["Text"])})

    # Construct the DatasetReader
    data = DatasetReader(combined_dataset, input_columns=input_columns_name, output_column=output_columns_name)

    print("Start inference....")
    bert_retriever = ColBERTRetriever(data, ice_num=ice_num, index_split='train', test_split='test')
    # ice_idx_list = bert_retriever.retrieve()
    ice_idx_list = bert_retriever.retrieve_mmr()

    # 保存每个样本的 (ice,text,label,ppl)
    sample_infos = []
    predictions = []

    for idx in tqdm(range(len(test_dataset)), desc="Processing test samples", unit="sample"):
        # === 构造样例 + prompt
        ice_item = format_predictions(ice_idx_list[idx], train_dataset)
        torch.cuda.empty_cache()
        text = test_dataset[idx]["Text"].split(" ")[:80]
        text = " ".join(text)
        # premise = clean_text(test_dataset[idx]["Premise"]).split(" ")[:64]
        # premise = " ".join(premise)
        # hypothesis = clean_text(test_dataset[idx]["Hypothesis"]).split(" ")[:64]
        # hypothesis = " ".join(hypothesis)
        true_label = test_dataset[idx]["Label"]

        prompt = f"Solve the sentiment analysis task. Options for sentiment: negative, positive, neutral.\n{ice_item}Text: {text} Prediction:"
        # prompt = f"Solve the toxic detection task. Options for toxicity: benign, toxic.\n{ice_item}Text: {text} Prediction:"
        # prompt = f"Solve the NLI task. Options for entailment relationship: entailment, neutral, contradiction.\n{ice_item}Premise: {premise}  Hypothesis: {hypothesis} Prediction:"

        # print(f"Prompt: {prompt}")

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=1024)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # ====== 生成预测 ======
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=3, pad_token_id=tokenizer.eos_token_id)

        # Extract the explanation (predicted label)
        batch_explanations = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        explanation = batch_explanations[0]
        sentiment = explanation.split("Prediction:")[-1].strip().lower()
        print(prompt)
        print(sentiment)
        # pred_label = -1
        if 'positive' in sentiment:
            predictions.append(1)
            pred_label = 1
            # print(1)
        elif 'negative' in sentiment:
            predictions.append(0)
            pred_label = 0
            # print(0)
        elif 'neutral' in sentiment:
        # else:
            predictions.append(2)
            pred_label = 2
            # print(2)
        else:
            predictions.append(-1)
            pred_label = -1
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

        # if 'entailment' in sentiment:
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
        #     print(2)
        # else:
        #     predictions.append(-1)
        #     pred_label = -1
        if pred_label == true_label:
            predictable = 1
        else:
            predictable = 0
        # ====== 计算第一个 token PPL ======
        with torch.no_grad():
            logits = model(**inputs).logits
            print(logits)
            gen_ids = outputs[:, inputs["input_ids"].shape[1]:]
            if gen_ids.shape[1] > 0:
                first_token_id = gen_ids[:, 0]
                # first_token_logits = logits[:, -1, :]
                first_token_logits = logits[:, inputs["input_ids"].shape[1] - 1, :]
                first_token_loss = F.cross_entropy(first_token_logits, first_token_id)
                first_token_ppl = math.exp(first_token_loss.item())
            else:
                first_token_ppl = float("nan")
        #
        sample_infos.append({
            "ice": ice_item,
            "text": text,
            "label": true_label,
            "ppl": first_token_ppl,
            "predictable": predictable
        })
        # sample_infos.append({
        #     "ice": ice_item,
        #     "premise": premise,
        #     "hypothesis": hypothesis,
        #     "label": true_label,
        #     "ppl": first_token_ppl,
        #     "predictable": predictable
        # })
        print(f"PPL: {first_token_ppl:.2f}")
        print(f"Predictable: {predictable} True Label: {true_label} Predicted Label: {pred_label}")

    # === 根据 PPL 由小到大排序 ===
    sample_infos = sorted(sample_infos, key=lambda x: x["ppl"])

    with open(f"/root/autodl-tmp/llm_code/ood_agent/sentiment/{task}/mmr_ice_data.jsonl", "w",encoding="utf-8") as f:
        import json
        json.dump(sample_infos, f, ensure_ascii=False, indent=2)

    return sample_infos


if __name__ == '__main__':
    import nltk

    print(nltk.data.path)
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
                     'td': ['Text'],
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
    tasks = ["dynasent","semeval","sst5"]
    # tasks = ["implicit_hate","toxigen","hsaol"]
    # tasks = ["wanli", "anli", "contract_nli"]
    model_names = ['gpt2']
    seeds = [43]
    
    sentence_transformer_path = 'all-mpnet-base-v2'
    data_dir = ''

    for model_name in model_names:
        model_path = model_dir + model_name
        sentence_model_path = sentence_transformer_path
        seed = 1

        for t in tasks:
            for task_name in task_names:
                train_path = 'train.jsonl'

                test_name = test_split[task_name]
                test_path = f'test_filtered.jsonl'
                ice_num = 3

                batch_size = 10

                candidate_num = 30
                select_time = 10

                main(templates[task_name], train_path, test_path, model_path, sentence_model_path,
                     input_columns[task_name], output_columns[task_name], ice_num, candidate_num, select_time,
                     batch_size, seed, t)

