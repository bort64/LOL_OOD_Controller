import json
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, T5Tokenizer, T5EncoderModel
from transformers import BertTokenizer, BertModel
from openicl.icl_dataset_reader import DatasetReader

from openicl.utils.logging import get_logger
from typing import List, Optional
from accelerate import Accelerator
from tqdm import trange, tqdm
from sentence_transformers import SentenceTransformer

from retriever.base_retriever import BaseRetriever

logger = get_logger(__name__)


class NerBERTRetriever_unpre_online(BaseRetriever):
    """基于BERT的In-context Learning Retriever"""

    def __init__(self,
                 dataset_reader: DatasetReader,
                 ice_separator: Optional[str] = '\n',
                 ice_eos_token: Optional[str] = '\n',
                 prompt_eos_token: Optional[str] = '',
                 ice_num: Optional[int] = 2,
                 index_split: Optional[str] = 'train',
                 model_name: Optional[str] = '/root/autodl-tmp/huggingface/transformers/bert-base-uncased') -> None:
        super().__init__(dataset_reader, ice_separator, ice_eos_token, prompt_eos_token, ice_num, index_split)

        # # 加载 BERT 模型和 tokenizer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval()


        # 初始化索引和测试语料
        self.index_corpus = self._embed_corpus(self.index_ds)

    def _get_embedding(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=1024).to(
            self.device)
        with torch.no_grad():
            output = self.model(**inputs)
        # 用 encoder 最后一层输出的均值作为嵌入
        return output.last_hidden_state.mean(dim=1)  # shape: [1, hidden_size]

    def _compute_cosine_similarity(self, query_embedding: torch.Tensor, index_embeddings: torch.Tensor) -> torch.Tensor:
        """计算查询与索引之间的余弦相似度"""
        # 归一化查询嵌入和索引嵌入
        query_embedding = query_embedding / query_embedding.norm(p=2, dim=-1, keepdim=True)  # 归一化查询嵌入
        index_embeddings = index_embeddings / index_embeddings.norm(p=2, dim=-1, keepdim=True)  # 归一化索引嵌入

        query_embedding = query_embedding.squeeze(0)  # Shape: [768]
        index_embeddings = index_embeddings.squeeze(1)  # Shape: [n, 768]

        # 计算余弦相似度
        similarity_scores = torch.matmul(query_embedding, index_embeddings.T)  # [1, N]
        return similarity_scores

    def _embed_corpus(self, dataset) -> dict:
        """生成整个数据集的嵌入向量"""
        all_embeddings = []
        all_indices = []
        all_samples = []  # 存完整样本，方便后面用 Predicted Entity

        for idx, item in enumerate(tqdm(dataset, desc="Generating embeddings")):
            text = item['Text']
            embedding = self._get_embedding(text)
            all_embeddings.append(embedding)
            all_indices.append(idx)
            all_samples.append(item)  # 把原始 dict 存进去

        embeddings_tensor = torch.stack(all_embeddings)
        print(f"Generated embeddings tensor with shape: {embeddings_tensor.shape}")

        return {
            'embeddings': embeddings_tensor,  # [N, hidden]
            'indices': all_indices,  # [N]
            'samples': all_samples  # [N] 原始样本
        }

    def add_to_index(self, new_item):
        # ----- 1. 生成文本 embedding -----
        text = new_item['Text']
        embedding = self._get_embedding(text).unsqueeze(0)  # shape [1, dim]

        # ----- 2. 更新 embedding -----
        if self.index_corpus['embeddings'] is None:
            self.index_corpus['embeddings'] = embedding
        else:
            self.index_corpus['embeddings'] = torch.cat(
                [self.index_corpus['embeddings'], embedding], dim=0
            )

        # ----- 3. 更新 indices -----
        new_index = len(self.index_corpus['indices'])
        self.index_corpus['indices'].append(new_index)

        # ----- 4. ⚠ 必须更新 samples -----
        self.index_corpus['samples'].append(new_item)
        print(self.index_corpus['samples'][-1])


    def retrieve_single_all(self, text: str) -> List[int]:
        """
        与 retrieve_multi_all 完全一致，但只检索单条 text。
        其他逻辑（相似度、多样性、排序、补全）全部保持不变。
        """

        alpha = 0.4# 多样性权重与原逻辑保持一致

        # === 1. 获取 query embedding ===
        query_embedding = self._get_embedding(text)

        # === 2. 计算相似度 ===
        candidate_embeddings = self.index_corpus['embeddings']

        similarities = self._compute_cosine_similarity(query_embedding, candidate_embeddings)

        # === 3. 构造样本信息（保持原逻辑不变） ===
        sample_infos = []
        for i in range(len(candidate_embeddings)):
            sample = self.index_corpus['samples'][i]
            pred_entities = sample.get("Entity", [])


            ent_types = set()
            for ent in pred_entities:
                if ent.startswith("O-"):
                    ent_types.add("O")
                elif ent.startswith("P-"):
                    ent_types.add("P")
                elif ent.startswith("L-"):
                    ent_types.add("L")

            num_types = len(ent_types)
            diversity_bonus = alpha * (num_types / 3.0)
            final_score = similarities[i].item() + diversity_bonus

            sample_infos.append({
                "idx": i,
                "score": final_score,
                "sim": similarities[i].item(),
                "num_types": num_types,
                "types": ent_types
            })

        # === 排序 ===
        sample_infos.sort(key=lambda x: x["score"], reverse=True)

        selected = []
        covered_types = set()

        # === Step 1: 保证 O / P / L 至少各一个 ===
        for target_type in ["O", "P", "L"]:
            for info in sample_infos:
                if target_type in info["types"] and info["idx"] not in selected:
                    selected.append(info["idx"])
                    covered_types |= info["types"]
                    break

        # === Step 2: 补足剩余样本 ===
        for info in sample_infos:
            if info["idx"] not in selected:
                selected.append(info["idx"])
            if len(selected) >= self.ice_num:
                break

        # 映射为全局 index
        global_indices = [self.index_corpus['indices'][i] for i in selected[:self.ice_num]]

        return global_indices

    def retrieve_multi_all_single(self, text: str) -> List[List[int]]:
        query_embedding = self._get_embedding(text)

        alpha = 0.3  # 多样性权重

        candidate_embeddings = self.index_corpus['embeddings']
        similarities = self._compute_cosine_similarity(query_embedding, candidate_embeddings)

        sample_infos = []

        for i in range(len(candidate_embeddings)):
            sample = self.index_corpus['samples'][i]
            entity_dict = sample.get("Entity", {})

            # === 实体类型解析：严格对齐 format_predictions_train ===
            ent_types = set()
            if entity_dict.get("organization", []):
                ent_types.add("O")
            if entity_dict.get("person", []):
                ent_types.add("P")
            if entity_dict.get("location", []):
                ent_types.add("L")
            # ===============================================

            num_types = len(ent_types)
            diversity_bonus = alpha * (num_types / 3.0)

            sim = similarities[i].item()
            final_score = sim + diversity_bonus

            sample_infos.append({
                "idx": i,
                "score": final_score,
                "sim": sim,
                "num_types": num_types,
                "types": ent_types
            })

        # === 排序 ===
        sample_infos.sort(key=lambda x: x["score"], reverse=True)

        selected = []

        # === Step 1: 优先选一个 O/P/L 各自一个（如果存在） ===
        for target_type in ["O", "P", "L"]:
            for info in sample_infos:
                if target_type in info["types"] and info["idx"] not in selected:
                    selected.append(info["idx"])
                    break

        # === Step 2: 按得分补足 ===
        for info in sample_infos:
            if info["idx"] not in selected:
                selected.append(info["idx"])
            if len(selected) >= self.ice_num:
                break

        global_indices = [self.index_corpus['indices'][i] for i in selected[:self.ice_num]]

        return global_indices



