import json
import random

import numpy as np
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


class NerBERTRetriever(BaseRetriever):
    """基于BERT的In-context Learning Retriever"""

    def __init__(self,
                 dataset_reader: DatasetReader,
                 ice_separator: Optional[str] = '\n',
                 ice_eos_token: Optional[str] = '\n',
                 prompt_eos_token: Optional[str] = '',
                 ice_num: Optional[int] = 1,
                 index_split: Optional[str] = 'train',
                 test_split: Optional[str] = 'test',
                 accelerator: Optional[Accelerator] = None,
                 model_name: Optional[str] = '/root/autodl-tmp/huggingface/transformers/bert-base-uncased') -> None:

        super().__init__(dataset_reader, ice_separator, ice_eos_token, prompt_eos_token, ice_num, index_split,
                         test_split, accelerator)

        # # 加载 BERT 模型和 tokenizer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval()


        # 初始化索引和测试语料
        self.index_corpus = self._embed_corpus(self.index_ds)
        self.test_corpus = [self.tokenizer.tokenize(data) for data in
                            self.dataset_reader.generate_input_field_corpus(self.test_ds)]
        # print("index_corpus shape:", len(self.index_corpus),type(self.index_corpus),self.index_corpus)
        # print("test_corpus shape:", len(self.test_corpus),type(self.index_corpus))

    def _get_embedding(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(
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
        # print(f"Processing dataset with {len(dataset)} items")
        # print(f"Sample item: {dataset[0]}")  # 打印第一条数据示例

        # 初始化存储结构
        all_embeddings = []
        all_indices = []

        # 使用tqdm显示进度条
        for idx, item in enumerate(tqdm(dataset, desc="Generating embeddings")):
            text = item['Text']
            embedding = self._get_embedding(text)
            all_embeddings.append(embedding)
            all_indices.append(idx)

        # 将嵌入列表转换为张量
        embeddings_tensor = torch.stack(all_embeddings)


        return {
            'embeddings': embeddings_tensor,  # 整个数据集的嵌入张量
            'indices': all_indices  # 对应的原始索引
        }
    #
    def retrieve_random1(self) -> List[List]:
        rtr_idx_list = []
        logger.info("Randomly selecting data for test set...")

        for idx in trange(len(self.test_corpus), disable=not self.is_main_process):
            # 获取所有候选样本的全局索引
            all_indices = self.index_corpus['indices']  # 假设这是所有候选样本的索引列表
            num_samples = min(self.ice_num, len(all_indices))
            selected_indices = random.sample(all_indices, num_samples) if len(
                all_indices) >= num_samples else all_indices

            rtr_idx_list.append(selected_indices)

        return rtr_idx_list
    def retrieve_random(self) -> List[List]:
        rtr_idx_list = []
        logger.info("Randomly selecting data for test set...")

        # 用原始 index_ds 取样（包含 Entity 字段）
        all_indices = list(range(len(self.index_ds)))

        for _ in trange(len(self.test_ds), disable=not self.is_main_process):
            valid_selection = False

            while not valid_selection:
                num_samples = min(self.ice_num, len(all_indices))
                selected_indices = random.sample(all_indices, num_samples)

                # 检查这几个样本是否覆盖三类实体
                has_location = any(
                    self.index_ds[i]["Entity"].get("location")
                    for i in selected_indices
                )
                has_person = any(
                    self.index_ds[i]["Entity"].get("person")
                    for i in selected_indices
                )
                has_organization = any(
                    self.index_ds[i]["Entity"].get("organization")
                    for i in selected_indices
                )

                # 三类实体都至少出现一次才算通过
                if has_location and has_person and has_organization:
                    valid_selection = True

            rtr_idx_list.append(selected_indices)

        return rtr_idx_list
# 使用相似度检索
    def retrieve(self) -> List[List]:
        rtr_idx_list = []
        logger.info("Retrieving data for test set...")

        for idx in trange(len(self.test_corpus), disable=not self.is_main_process):
            # 获取测试文本和嵌入
            test_item = self.test_ds[idx]
            query = test_item['Text']
            query_embedding = self._get_embedding(query)

            # 计算与所有候选样本的相似度
            candidate_embeddings = self.index_corpus['embeddings']  # shape: [num_samples, embed_dim]
            similarities = self._compute_cosine_similarity(query_embedding, candidate_embeddings)

            # 直接选择相似度最高的9个样本
            topk_values, topk_indices = similarities.topk(min(self.ice_num, len(candidate_embeddings)))

            # 转换为原始索引
            global_indices = [self.index_corpus['indices'][i] for i in topk_indices]
            random.shuffle(global_indices)
            rtr_idx_list.append(global_indices)

        return rtr_idx_list

    @torch.no_grad()
    def retrieve_mmr(self) -> List[List[int]]:
        """
        使用 Maximal Marginal Relevance (MMR) 进行多样性增强的样本检索。
        在显存受限场景下支持分块计算，自动切换 GPU/CPU。
        """
        logger.info("MMR retrieval started ...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rtr_idx_list = []

        # 全部候选样本的 embedding 与索引
        all_embs = self.index_corpus["embeddings"].to(device)  # may be [N, D] or [N,1,D]
        # 防御性 squeeze：确保 all_embs 为 [N, D]
        if all_embs.dim() == 3 and all_embs.size(1) == 1:
            all_embs = all_embs.squeeze(1)

        all_ids = self.index_corpus["indices"]
        num_candidates = all_embs.size(0)

        lambda_param = 0.5  # 控制相似度 vs 多样性
        chunk_size = 2048  # 分块避免爆显存

        for idx in trange(len(self.test_corpus), disable=not self.is_main_process):
            test_item = self.test_ds[idx]
            query = test_item["Text"]

            query_emb = self._get_embedding(query).to(device)  # [1, D] or [1, D]
            # squeeze query_emb -> [D]
            if query_emb.dim() == 2 and query_emb.size(0) == 1:
                query_emb_squeezed = query_emb.squeeze(0)
            else:
                query_emb_squeezed = query_emb

            # 计算所有候选与 query 的相似度（返回 [N]）
            # 归一化再 matmul
            q_norm = query_emb_squeezed / (query_emb_squeezed.norm(p=2) + 1e-12)
            emb_norm = all_embs / (all_embs.norm(p=2, dim=1, keepdim=True) + 1e-12)
            sims = torch.matmul(emb_norm, q_norm)  # shape: [N]

            # Step 2: 先选 top-K 候选（减少后续计算量）
            topk_cand = min(15, num_candidates)
            topk_values, topk_idx = torch.topk(sims, k=topk_cand)
            cand_embs = all_embs[topk_idx]  # -> [topk_cand, D]
            cand_ids = [all_ids[i] for i in topk_idx.view(-1).cpu().tolist()]

            # Step 3: 初始化 MMR
            selected = []
            remaining = list(range(len(cand_embs)))

            for _ in range(min(self.ice_num, len(remaining))):
                mmr_scores = []
                for i in remaining:
                    # sims[topk_idx[i]] ->  scalar
                    rel = sims[topk_idx[i].item()].item() if isinstance(topk_idx[i], torch.Tensor) else sims[
                        topk_idx[i]].item()

                    # 多样性惩罚项
                    if len(selected) == 0:
                        div = 0.0
                    else:
                        max_div = -1.0
                        # 分块计算以防显存
                        for j in range(0, len(selected), chunk_size):
                            sel_chunk = cand_embs[selected[j:j + chunk_size]]
                            sim_chunk = torch.nn.functional.cosine_similarity(
                                cand_embs[i].unsqueeze(0), sel_chunk, dim=1
                            )
                            max_div = max(max_div, float(sim_chunk.max().item()))
                        div = max_div

                    # MMR分数
                    mmr_score = lambda_param * rel - (1 - lambda_param) * div
                    mmr_scores.append(mmr_score)

                best_idx_in_remaining = int(np.argmax(mmr_scores))
                best_cand_idx = remaining[best_idx_in_remaining]
                selected.append(best_cand_idx)
                remaining.remove(best_cand_idx)

            # Step 4: 取回全局索引
            final_ids = [cand_ids[i] for i in selected]
            random.shuffle(final_ids)
            rtr_idx_list.append(final_ids)

        logger.info("MMR retrieval finished ")
        return rtr_idx_list

