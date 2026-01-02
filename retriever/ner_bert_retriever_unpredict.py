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


class NerBERTRetriever_unpre(BaseRetriever):
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
        print("index_corpus shape:", len(self.index_corpus),type(self.index_corpus),self.index_corpus)
        print("test_corpus shape:", len(self.test_corpus),type(self.index_corpus))

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
            text = item['text']
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

    #
    def retrieve_random(self) -> List[List]:
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
    def retrieve_random_label(self) -> List[List]:
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
# # 使用相似度检索
    def retrieve_no_label(self) -> List[List]:
        rtr_idx_list = []
        logger.info("Retrieving data for test set...")

        for idx in trange(len(self.test_corpus), disable=not self.is_main_process):
            # 获取测试文本和嵌入
            test_item = self.test_ds[idx]
            query = test_item['text']
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

    def retrieve(self) -> List[List]:
        rtr_idx_list = []
        logger.info("Retrieving data for test set...")

        for idx in trange(len(self.test_corpus), disable=not self.is_main_process):
            test_item = self.test_ds[idx]
            query = test_item['text']
            query_embedding = self._get_embedding(query)

            # 计算相似度
            candidate_embeddings = self.index_corpus['embeddings']
            similarities = self._compute_cosine_similarity(query_embedding, candidate_embeddings)

            # 先取前 50 个候选（不够就取全部）
            topk_values, topk_indices = similarities.topk(min(50, len(candidate_embeddings)))

            selected = []
            covered_types = set()

            # === 先保证 O/P/L 三种必须覆盖 ===
            for i in topk_indices:
                i = i.item()
                sample = self.index_corpus['samples'][i]
                pred_entities = sample.get("Predicted Entity", [])

                ent_types = set()
                for ent in pred_entities:
                    if ent.startswith("O-"):
                        ent_types.add("O")
                    elif ent.startswith("P-"):
                        ent_types.add("P")
                    elif ent.startswith("L-"):
                        ent_types.add("L")

                if not ent_types.issubset(covered_types):
                    selected.append(i)
                    covered_types |= ent_types

                if covered_types == {"O", "P", "L"}:
                    break

            # === 如果没覆盖全，继续往下补齐 ===
            if covered_types != {"O", "P", "L"}:
                for i in topk_indices:
                    i = i.item()
                    if i in selected:
                        continue
                    sample = self.index_corpus['samples'][i]
                    pred_entities = sample.get("Predicted Entity", [])
                    ent_types = set()
                    for ent in pred_entities:
                        if ent.startswith("O-"):
                            ent_types.add("O")
                        elif ent.startswith("P-"):
                            ent_types.add("P")
                        elif ent.startswith("L-"):
                            ent_types.add("L")
                    if not ent_types.issubset(covered_types):
                        selected.append(i)
                        covered_types |= ent_types
                    if covered_types == {"O", "P", "L"}:
                        break

            # === 如果还没到 ice_num，继续补齐 ===
            if len(selected) < self.ice_num:
                for i in topk_indices:
                    i = i.item()
                    if i not in selected:
                        selected.append(i)
                    if len(selected) >= self.ice_num:
                        break

            # 转换为全局索引并随机打乱
            global_indices = [self.index_corpus['indices'][i] for i in selected]

            rtr_idx_list.append(global_indices)

        return rtr_idx_list

    def retrieve_sim(self) -> List[List[int]]:
        rtr_idx_list = []
        logger.info("Retrieving data for test set (similarity-prioritized)...")

        for idx in trange(len(self.test_corpus), disable=not self.is_main_process):
            test_item = self.test_ds[idx]
            query = test_item['text']
            query_embedding = self._get_embedding(query)

            # 计算相似度并按降序排序
            candidate_embeddings = self.index_corpus['embeddings']
            similarities = self._compute_cosine_similarity(query_embedding, candidate_embeddings)
            sorted_indices = torch.argsort(similarities, descending=True).tolist()

            selected = []
            covered_types = set()

            # === 从最高相似度开始选，直到覆盖 O/P/L ===
            for i in sorted_indices:
                sample = self.index_corpus['samples'][i]
                pred_entities = sample.get("Predicted Entity", [])

                ent_types = set()
                for ent in pred_entities:
                    if ent.startswith("O-"):
                        ent_types.add("O")
                    elif ent.startswith("P-"):
                        ent_types.add("P")
                    elif ent.startswith("L-"):
                        ent_types.add("L")

                # 如果这个样本能补充新的类型，则保留
                if not ent_types.issubset(covered_types):
                    selected.append(i)
                    covered_types |= ent_types

                # 如果已经齐全，提前退出
                if covered_types == {"O", "P", "L"}:
                    break

            # === 类型不齐或数量不够，再按相似度补齐 ===
            if len(selected) < self.ice_num:
                for i in sorted_indices:
                    if i not in selected:
                        selected.append(i)
                    if len(selected) >= self.ice_num:
                        break

            # === 转全局索引（不再打乱顺序） ===
            global_indices = [self.index_corpus['indices'][i] for i in selected[:self.ice_num]]
            rtr_idx_list.append(global_indices)

        return rtr_idx_list

    def retrieve_multi(self) -> List[List[int]]:
        rtr_idx_list = []
        logger.info("Retrieving data for test set (similarity + diversity balanced)...")

        alpha = 0.2  # 多样性权重，可调。0.1~0.3 较稳定，越大越偏向多类型样本。

        for idx in trange(len(self.test_corpus), disable=not self.is_main_process):
            test_item = self.test_ds[idx]
            query = test_item['text']
            query_embedding = self._get_embedding(query)

            candidate_embeddings = self.index_corpus['embeddings']
            similarities = self._compute_cosine_similarity(query_embedding, candidate_embeddings)

            sample_infos = []
            for i in range(len(candidate_embeddings)):
                sample = self.index_corpus['samples'][i]
                pred_entities = sample.get("Predicted Entity", [])

                ent_types = set()
                for ent in pred_entities:
                    if ent.startswith("O-"):
                        ent_types.add("O")
                    elif ent.startswith("P-"):
                        ent_types.add("P")
                    elif ent.startswith("L-"):
                        ent_types.add("L")

                # 用多样性权重调整最终得分
                num_types = len(ent_types)
                diversity_bonus = alpha * (num_types / 3.0)  # 最多加 alpha 分
                final_score = similarities[i].item() + diversity_bonus

                sample_infos.append({
                    "idx": i,
                    "score": final_score,
                    "sim": similarities[i].item(),
                    "num_types": num_types,
                    "types": ent_types
                })

            # 按综合得分排序
            sample_infos.sort(key=lambda x: x["score"], reverse=True)

            selected = []
            covered_types = set()

            # === 先按综合分选样本，确保 O/P/L 尽量齐全 ===
            for info in sample_infos:
                selected.append(info["idx"])
                covered_types |= info["types"]
                if len(selected) >= self.ice_num:
                    break

            global_indices = [self.index_corpus['indices'][i] for i in selected[:self.ice_num]]

            rtr_idx_list.append(global_indices)

        return rtr_idx_list







