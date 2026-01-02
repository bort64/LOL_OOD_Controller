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
import requests
import time
from datasets import Dataset, DatasetDict
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from retriever.bert_retriever_unpreict import ColBERTRetriever_unpre

def clean_text(text: str) -> str:
    # 替换换行符为空格
    text = text.replace("\n", " ")
    # 去掉下划线、长横线、破折号等无效符号
    text = re.sub(r"[_\-–—….]+", " ", text)
    # 合并多个空格为一个空格
    text = re.sub(r"\s+", " ", text)
    # 去掉首尾空格
    return text.strip()



def format_predictions(idx_list, train_dataset):
    # 情感分析标签映射
    label_map = {0: 'negative', 1: 'positive', 2: 'neutral',-1: "neutral"}
    # label_map = {0: 'benign', 1: 'toxic', -1: "benign"}
    # label_map = {0: 'entailment', 1: 'contradiction', 2: 'neutral',-1: "neutral"}
    # 按标签分组样本
    label_groups = defaultdict(list)
    for idx in idx_list:
        text = train_dataset[idx]["text"]
        url_pattern = re.compile(
            r'\b(?:https?://|http?://|www\.|[a-zA-Z0-9\-]+\.(?:com|co|org|io|net|gov|edu|t\.co)/)\S*\b',
            flags=re.IGNORECASE
        )
        mention_pattern = re.compile(r'@\w(?:[\w\.\-_]*\w)?', flags=re.UNICODE)
        # 3) 去掉首尾空格
        cleaned = url_pattern.sub('', text)  # 删除 URL
        cleaned = mention_pattern.sub('', cleaned)  # 删除 mention
        # 删除空的括号或方括号（可能是 URL 删除后残留）
        cleaned = re.sub(r'\(\s*\)', '', cleaned)
        cleaned = re.sub(r'\[\s*\]', '', cleaned)
        text = re.sub(r'\s+', ' ', cleaned).strip()
        # Premise = clean_text(train_dataset[idx]["premise"]).split(" ")[:64]
        # Premise = " ".join(Premise)
        # Hypothesis = clean_text(train_dataset[idx]["hypothesis"]).split(" ")[:64]
        # Hypothesis = " ".join(Hypothesis)
        label = train_dataset[idx]["predict_label"]
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



def calculate_accuracy(predictions, references):
    correct = sum(p == r for p, r in zip(predictions, references))
    total = len(references)
    return correct / total if total > 0 else 0.0

from tqdm import tqdm  # 引入 tqdm

def main(template, test_path, input_columns_name, output_columns_name,file):
    combined_dataset = load_dataset("json", data_files={"test": test_path})
    test_dataset = combined_dataset["test"]
    labels = [item["label"] for item in test_dataset]
    predictions = []
    predict_predictions = []
    unpredict_predictions = []
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
        # text1 = clean_text(test_dataset[idx]["premise"]).split(" ")[:64]
        # text1 = " ".join(text1)
        # text2 = clean_text(test_dataset[idx]["hypothesis"]).split(" ")[:64]
        # text2 = " ".join(text2)
        # prompt = f"Solve the NLI task. Options for entailment relationship: entailment, neutral, contradiction.\n{ice}Premise: {text1}  Hypothesis: {text2} Prediction:"
        prompt = f"Solve the sentiment analysis task. Options for sentiment: negative, positive, neutral.\n{ice}Text: {text} Prediction:"
        # prompt = f"Solve the toxic detection task. Options for toxicity: benign, toxic.\n{ice}Text: {text} Prediction:"
        # print(f"Prompt: {prompt}")
        sentiment = chat(prompt)
        confidence = 0.8
        # print(f"Sentiment: {sentiment}")
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
        # else:
        #     predictions.append(-1)
        #     pred_label = -1
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

    #     # 计算准确率
    accuracy = calculate_accuracy(predictions, labels)
    predct_accuracy = calculate_accuracy(predict_predictions, predic_labels)
    unpredct_accuracy = calculate_accuracy(unpredict_predictions, unpredic_labels)
    print(f"Accuracy: {accuracy}")
    print(f"predct_accuracy: {predct_accuracy}")
    print(f"unpredct_accuracy: {unpredct_accuracy}")
    with open(f"{file}random_llama_classify0.1_cut500_gpt4.jsonl", "w",encoding="utf-8") as f:
        for item in all_output:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    train_samples = []
    test_samples = []
    input_file = f"{file}random_llama_classify0.1_cut500_gpt4.jsonl"
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item.get("predictable", 0) == 1:
                if item["confidence"] >= 0.7:
                    train_samples.append(item)
            else:
                test_samples.append(item)
            predictions.append(item["predict_label"])
    print("train size:", len(train_samples))
    train_dataset_unpredict = Dataset.from_list(train_samples)
    test_dataset_ubpredict = Dataset.from_list(test_samples)
    combined_dataset_unpredict = DatasetDict({
        "train": train_dataset_unpredict,
        "test": test_dataset_ubpredict
    })

    data = DatasetReader(combined_dataset_unpredict, input_columns=input_columns_name,
                         output_column=output_columns_name)
    bert_retriever = ColBERTRetriever_unpre(data, ice_num=3, index_split='train', test_split='test')
    ice_idx_list = bert_retriever.retrieve()
    predictions_u = []
    predict_predictions_u = []
    unpredict_predictions_u = []
    predic_labels_u = []
    unpredic_labels_u = []
    all_output_u = []
    unpredict_idx = 0
    for idx in tqdm(range(len(test_dataset)), desc="Processing test samples", unit="sample"):
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.empty_cache()
        text = test_dataset[idx]["text"]
        true_label = test_dataset[idx]["label"]
        predictable = test_dataset[idx]["predictable"]
        if predictable == 1:
            predictions_u.append(predictions[idx])
            pred_label = predictions[idx]
        else:
            if task_item == "dynasent":
                text1 = "My Wife and I moved to Arizona due to my company sending me down here and we spent MONTHS looking at places online and everywhere looked kinda crappy so we were really worried."
                text2 = "We left the car. The smell was amazing as we got out of our car."
                text3 = "They are deep fried I believe 3 times?"
            elif task_item == "semeval":
                text1 = "HamidMirGEO udassarAmin76 shame on you for calling this """"war on terror"""".. May Allah destroy this nation who have taken arms against Islam"
                text2 = "GLEN HODDLE: Harry Kane's impact Friday was extraordinary. It was great to see Kane, the leading Premier League scorer, come on and score!"
                text3 = "National Hot Dog day is tomorrow! Celebrate w cheffinis DTContainerPark - . #food #vegas #yum"
            else:
                text1 = "There is not a character in the movie with a shred of plausibility , not an event that is believable , not a confrontation that is not staged , not a moment that is not false ."
                text2 = "Nicolas Philibert observes life inside a one-room schoolhouse in northern France in his documentary To Be and to Have , easily one of the best films of the year ."
                text3 = "Both awful and appealing ."
            text1 = " ".join(text1.split(" ")[:80])
            text2 = " ".join(text2.split(" ")[:80])
            text3 = " ".join(text3.split(" ")[:80])
            # ice_item = f"Text: {text1} Prediction: negative\nText: {text2} Prediction: positive\nText: {text3} Prediction: neutral\n"
            ice_item = format_predictions(ice_idx_list[unpredict_idx],train_dataset_unpredict)
            unpredict_idx += 1
            # text1 = clean_text(test_dataset[idx]["premise"]).split(" ")[:64]
            # text1 = " ".join(text1)
            # text2 = clean_text(test_dataset[idx]["hypothesis"]).split(" ")[:64]
            # text2 = " ".join(text2)
            # prompt = f"Solve the NLI task. Options for entailment relationship: entailment, neutral, contradiction.\n{ice_item}Premise: {text1}  Hypothesis: {text2} Prediction:"
            prompt = f"Solve the sentiment analysis task. Options for sentiment: negative, positive, neutral.\n{ice_item}Text: {text} Prediction:"
            # prompt = f"Solve the toxic detection task. Options for toxicity: benign, toxic.\n{ice_item}Text: {text} Prediction:"

            # print(prompt)
            predicted_label = chat(prompt)
            sentiment = predicted_label
            if 'positive' in sentiment:
                predictions_u.append(1)
                pred_label = 1
                # print(1)
            elif 'negative' in sentiment:
                predictions_u.append(0)
                pred_label = 0
                # print(0)
            elif 'neutral' in sentiment:
                predictions_u.append(2)
                pred_label = 2
                # print(2)
            else:
                predictions_u.append(-1)
                pred_label = -1
            # if 'benign' in sentiment:
            #     predictions_u.append(0)
            #     pred_label = 0
            #     # print(0)
            # elif 'toxic' in sentiment:
            #     predictions_u.append(1)
            #     pred_label = 1
            #     # print(1)
            # else:
            #     predictions_u.append(-1)
            #     pred_label = -1
            # if 'entail' in sentiment:
            #     predictions_u.append(0)
            #     pred_label = 0
            #     # print(0)
            # elif 'contradiction' in sentiment:
            #     predictions_u.append(1)
            #     pred_label = 1
            #     # print(1)
            # elif 'neutral' in sentiment:
            #     predictions_u.append(2)
            #     pred_label = 2
            # else:
            #     predictions_u.append(-1)
            #     pred_label = -1
        if predictable == 1:
            predic_labels_u.append(true_label)
            predict_predictions_u.append(pred_label)
        else:
            unpredic_labels_u.append(true_label)
            unpredict_predictions_u.append(pred_label)

    accuracy_u = calculate_accuracy(predictions_u, labels)
    predct_accuracy_u = calculate_accuracy(predict_predictions_u, predic_labels_u)
    unpredct_accuracy_u = calculate_accuracy(unpredict_predictions_u, unpredic_labels_u)
    print(f"Accuracy: {accuracy_u}")
    print(f"predct_accuracy: {predct_accuracy_u}")
    print(f"unpredct_accuracy: {unpredct_accuracy_u}")

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
    # task = ['dynasent','semeval','sst5']
    # task = ['implicit_hate', 'toxigen', 'hsaol']
    # task = ['implicit_hate', 'hsaol']
    # task = ['wanli', 'anli', 'contract_nli']
    task = ['semeval','sst5']

    for task_item in task:

            for task_name in task_names:
                # test_path = '/root/autodl-tmp/llm_code/ood_agent/sentiment/dynasent/qwen_ppl_sort_data_classify.jsonl'
                file = f"/root/autodl-tmp/llm_code/ood_agent/sentiment/random/{task_item}/"
                test_path = f"{file}random_llama_classify0.1_cut500.jsonl"
                main(templates[task_name], test_path,input_columns[task_name], output_columns[task_name],file)
