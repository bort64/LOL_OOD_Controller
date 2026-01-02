import json
import logging
import random

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModel
from transformers import BertTokenizer, BertModel

from openicl.icl_dataset_reader import DatasetReader
from openicl.utils.logging import get_logger
from typing import List, Optional
from tqdm import trange, tqdm
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5EncoderModel

from retriever.base_retriever import BaseRetriever

logger = get_logger(__name__)


class ColBERTRetriever(BaseRetriever):
    """基于BERT的In-context Learning Retriever"""

    def __init__(self,
                 dataset_reader: DatasetReader,
                 ice_separator: Optional[str] = '\n',
                 ice_eos_token: Optional[str] = '\n',
                 prompt_eos_token: Optional[str] = '',
                 ice_num: Optional[int] = 1,
                 index_split: Optional[str] = 'train',
                 test_split: Optional[str] = 'test',
                 model_name: Optional[str] = '/root/autodl-tmp/huggingface/transformers/bert-base-uncased') -> None:
                 # model_name: Optional[str] = '/root/autodl-fs/sentence-transformer/all-mpnet-base-v2') -> None:
                 # model_name: Optional[str] = '/root/autodl-tmp/meta-llama/bge-m3') -> None:
                 # model_name: Optional[str] = '/root/autodl-tmp/meta-llama/DrICL/TD/') -> None:
        super().__init__(dataset_reader, ice_separator, ice_eos_token, prompt_eos_token, ice_num, index_split,
                         test_split)

        # # 加载 BERT 模型和 tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.model = AutoModel.from_pretrained(model_name).to(self.device)
        # # self.tokenizer.pad_token = self.tokenizer.eos_token
        # self.model = SentenceTransformer("/root/autodl-fs/sentence-transformer/all-mpnet-base-v2")
        # self.model = SentenceTransformer("/root/autodl-tmp/meta-llama/all-MiniLM-L6-v2")

        # self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        # self.model = T5EncoderModel.from_pretrained(model_name).to(self.device)
        self.model.eval()


        # 初始化索引和测试语料
        self.index_corpus = self._embed_corpus(self.index_ds)
        # self.index_corpus = self._embed_corpus_unpredict(self.index_ds)
        self.test_corpus = [self.tokenizer.tokenize(data) for data in
                            self.dataset_reader.generate_input_field_corpus(self.test_ds)]
        # print("index_corpus shape:", len(self.index_corpus),type(self.index_corpus),self.index_corpus)
        # print("test_corpus shape:", len(self.test_corpus),type(self.index_corpus))
#在用的
    def _get_embedding(self, text: str) -> torch.Tensor:
        """生成单条文本的嵌入"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(
            self.device)
        with torch.no_grad():
            output = self.model(**inputs)
        return output.last_hidden_state.mean(dim=1)  # Mean pooling to get a single vector



    # def _get_embedding(self, text: str) -> torch.Tensor:
    #     # 直接调用 SentenceTransformer 的 encode 方法
    #     embedding = self.model.encode(text, convert_to_tensor=True)
    #     return embedding.unsqueeze(0)

    #t5模型的embedding
    # def _get_embedding(self, text: str) -> torch.Tensor:
    #     inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(
    #         self.device)
    #     with torch.no_grad():
    #         output = self.model(**inputs)
    #     # 用 encoder 最后一层输出的均值作为嵌入
    #     return output.last_hidden_state.mean(dim=1)  # shape: [1, hidden_size]


    def _compute_cosine_similarity(self, query_embedding: torch.Tensor, index_embeddings: torch.Tensor) -> torch.Tensor:
        """计算查询与索引之间的余弦相似度"""
        # 归一化查询嵌入和索引嵌入
        query_embedding = query_embedding / query_embedding.norm(p=2, dim=-1, keepdim=True)  # 归一化查询嵌入
        index_embeddings = index_embeddings / index_embeddings.norm(p=2, dim=-1, keepdim=True)  # 归一化索引嵌
        query_embedding = query_embedding.squeeze(0)  # Shape: [768]
        index_embeddings = index_embeddings.squeeze(1)  # Shape: [n, 768]
        # 计算余弦相似度
        similarity_scores = torch.matmul(query_embedding, index_embeddings.T)  # [1, N]
        return similarity_scores

    def _embed_corpus(self, dataset) -> dict:
        """生成按标签分组的嵌入字典"""
        # print(dataset)
        label_embeddings = {}
        label_indices = {}
        # print(type(label_indices),len(label_indices))
        # print(type(label_embeddings),len(label_embeddings))
        for idx, item in enumerate(dataset):
            # text = " ".join([str(item[col]) for col in self.dataset_reader.input_columns])
            # Premise = item['Premise'].split(" ")[:64]
            # Premise = " ".join(Premise)
            # Hypothesis = item['Hypothesis'].split(" ")[:64]
            # Hypothesis = " ".join(Hypothesis)
            # text = Premise + Hypothesis
            # print(text)
            text = item['Text'].split(" ")[:80]
            text = " ".join(text)
            label = item['Label']
            embedding = self._get_embedding(text)
            if label not in label_embeddings:
                label_embeddings[label] = []
                label_indices[label] = []
            label_embeddings[label].append(embedding)
            label_indices[label].append(idx)

        # 将每个标签的嵌入堆叠为张量
        for label in label_embeddings:
            label_embeddings[label] = torch.stack(label_embeddings[label])
        # print(len(label_embeddings), len(label_indices))
        return {'embeddings': label_embeddings, 'indices': label_indices}

# 使用相似度检索
    def retrieve(self) -> List[List]:
        rtr_idx_list = []
        logger.info("Retrieving data for test set...")
        random.seed(42)
        for idx in tqdm(range(len(self.test_corpus)), disable=not self.is_main_process):
            test_item = self.test_ds[idx]
            # test_premise = test_item['Premise'][:64]
            # test_hypothesis = test_item['Hypothesis'][:64]

            text = test_item['Text'].split(" ")[:80]
            text = " ".join(text)
            query = text
            # query = test_premise + test_hypothesis
            # query = " ".join(self.test_corpus[idx])
            query_embedding = self._get_embedding(query)

            # 按标签分组检索
            all_scores, all_indices = [], []
            for label in self.index_corpus['embeddings']:
                label_emb = self.index_corpus['embeddings'][label]
                similarity = self._compute_cosine_similarity(query_embedding, label_emb)
                local_indices = similarity.topk(min(int(self.ice_num/3), len(label_emb))).indices
                global_indices = [self.index_corpus['indices'][label][i] for i in local_indices]
                all_scores.extend(similarity[local_indices].tolist())
                all_indices.extend(global_indices)

            sorted_pairs = sorted(zip(all_scores, all_indices), reverse=True, key=lambda x: x[0])
            final_indices = [idx for _, idx in sorted_pairs[:self.ice_num]]
            rtr_idx_list.append(final_indices)
        return rtr_idx_list

    def retrieve_single(self, text: str) -> List[int]:
        # for label in self.index_corpus['embeddings']:
        #     print(len(self.index_corpus['indices'][label]))
        query_embedding = self._get_embedding(" ".join(text.split(" ")[:80]))
        all_scores, all_indices = [], []
        for label, label_emb in self.index_corpus['embeddings'].items():
            similarity = self._compute_cosine_similarity(query_embedding, label_emb)
            local_indices = similarity.topk(min(int(self.ice_num / 3), len(label_emb))).indices
            global_indices = [self.index_corpus['indices'][label][i] for i in local_indices]
            all_scores.extend(similarity[local_indices].tolist())
            all_indices.extend(global_indices)
        sorted_pairs = sorted(zip(all_scores, all_indices), reverse=True, key=lambda x: x[0])
        return [idx for _, idx in sorted_pairs[:self.ice_num]]

#随机+采样
    # def retrieve(self) -> List[List]:
    #     rtr_idx_list = []
    #     logger.info("Randomly selecting data for test set...")
    #     random.seed(42)
    #
    #     # 遍历测试集
    #     for idx in trange(len(self.test_corpus), disable=not self.is_main_process):
    #         # 按标签分组随机选择
    #         all_indices = []
    #         for label in self.index_corpus['embeddings']:
    #             label_indices = self.index_corpus['indices'][label]
    #
    #             # 随机选择 ice_num 个索引（如果该标签下的索引数足够）
    #             random_indices = random.sample(label_indices, min(int(self.ice_num/3), len(label_indices)))
    #             all_indices.extend(random_indices)
    #
    #         # 打乱选择的索引，保证返回结果的随机性
    #         random.shuffle(all_indices)
    #
    #         # 只选择最相关的 ice_num 个
    #         final_indices = all_indices[:self.ice_num]
    #         rtr_idx_list.append(final_indices)
    #
    #     return rtr_idx_list
    def retrieve_single_random(self, text: str) -> List[int]:
        """
        Randomly retrieve samples for a single text input.
        This method does not use embeddings — it randomly selects self.ice_num examples
        from the indexed corpus, evenly distributed across labels.

        Args:
            text (str): The input text to retrieve in-context examples for.

        Returns:
            List[int]: List of randomly selected example indices.
        """
        random.seed(42)
        all_indices = []

        # 按标签均衡随机选取
        for label, label_indices in self.index_corpus['indices'].items():
            # 从该标签下随机选取若干条
            sampled = random.sample(label_indices, min(int(self.ice_num / 3), len(label_indices)))
            all_indices.extend(sampled)

        # 打乱，增加样本多样性
        # random.shuffle(all_indices)

        # 最终只取前 self.ice_num 个
        final_indices = all_indices[:self.ice_num]

        return final_indices



    @torch.no_grad()
    def retrieve_mmr_low(self) -> List[List[int]]:
        """
        Low-memory MMR: compute sims in CPU chunks, select top-K then run MMR on those candidates.
        """
        logger.info("MMR-Low-Memory started ...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # flatten index_corpus into CPU tensor
        label_embs = []
        label_ids = []
        for label, emb in self.index_corpus["embeddings"].items():
            label_embs.append(emb.cpu())  # keep on CPU
            label_ids.extend(self.index_corpus["indices"][label])

        if len(label_embs) == 0:
            return [[] for _ in range(len(self.test_ds))]

        all_embs = torch.cat(label_embs, dim=0)  # [N, D] on CPU
        all_ids = list(label_ids)
        num_candidates = all_embs.size(0)

        top_k_candidate = getattr(self, "k_candidate", 15)
        ice_num = getattr(self, "ice_num", 3)
        lambda_param = getattr(self, "lambda_param", 0.7)
        batch_size = getattr(self, "sim_batch_size", 512)  # CPU batch for sims
        chunk_size = getattr(self, "div_chunk_size", 512)

        rtr_idx_list = []

        for idx in trange(len(self.test_ds), desc="MMR-Low-Memory", disable=not self.is_main_process):
            test_item = self.test_ds[idx]
            Premise = test_item['Premise'].split(" ")[:64]
            Premise = " ".join(Premise)
            Hypothesis = test_item['Hypothesis'].split(" ")[:64]
            Hypothesis = " ".join(Hypothesis)
            query_text = Premise + Hypothesis
            # query_text = " ".join(test_item['Text'].split(" ")[:80])
            query_emb = self._get_embedding(query_text).to(device)  # [1, D]

            # compute sims in CPU batches (ensure sims is 1D)
            sims_parts = []
            for start in range(0, all_embs.size(0), batch_size):
                end = min(start + batch_size, all_embs.size(0))
                batch = all_embs[start:end].to(device)  # move small batch to device
                # normalize
                q = query_emb.squeeze(0)
                q = q / (q.norm(p=2) + 1e-12)
                batch_norm = batch / (batch.norm(p=2, dim=1, keepdim=True) + 1e-12)
                sim = torch.matmul(batch_norm, q)  # [batch_size]
                sims_parts.append(sim.cpu())
                del batch, batch_norm, sim

            sims = torch.cat(sims_parts, dim=0).view(-1)  # [N] on CPU

            if sims.numel() == 0:
                rtr_idx_list.append([])
                continue

            # Top-k candidates (on CPU)
            topk = min(top_k_candidate, sims.size(0))
            topk_values, topk_idx = torch.topk(sims, k=topk)
            topk_idx = topk_idx.view(-1)  # tensor on CPU
            topk_values = topk_values.view(-1)

            # cand embeddings moved to device for MMR computations
            cand_embs = all_embs[topk_idx].to(device)  # [K, D]
            cand_ids = [all_ids[int(i)] for i in topk_idx.cpu().tolist()]

            # MMR selection (operate on device)
            selected = []
            remaining = list(range(cand_embs.size(0)))

            for _ in range(min(ice_num, len(remaining))):
                mmr_scores = []
                for i in remaining:
                    # relevance: use topk_values (on CPU)
                    rel = float(topk_values[i].item())

                    # diversity: compute max similarity between cand i and selected
                    if len(selected) == 0:
                        div = 0.0
                    else:
                        div_vals = []
                        # compute in selected chunks to avoid big memory
                        for j in range(0, len(selected), chunk_size):
                            sel_chunk = cand_embs[selected[j:j + chunk_size]]  # [s_chunk, D]
                            sim_chunk = torch.nn.functional.cosine_similarity(
                                cand_embs[i].unsqueeze(0), sel_chunk, dim=1
                            )  # [s_chunk]
                            div_vals.append(sim_chunk.max().item())
                        div = max(div_vals)

                    mmr_score = lambda_param * rel - (1 - lambda_param) * div
                    mmr_scores.append(mmr_score)

                best_idx_in_remaining = int(np.argmax(mmr_scores))
                best_cand_idx = remaining[best_idx_in_remaining]
                selected.append(best_cand_idx)
                remaining.remove(best_cand_idx)

            final_ids = [cand_ids[i] for i in selected]
            rtr_idx_list.append(final_ids)

            # cleanup
            try:
                del query_emb, sims, cand_embs
            except Exception:
                pass

        logger.info("MMR-Low-Memory finished.")
        return rtr_idx_list