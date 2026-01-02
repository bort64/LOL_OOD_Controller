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


class NerKNNRetriever(BaseRetriever):
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
                 # model_name: Optional[str] = "/root/autodl-fs/google/gemma-2-2b") -> None:
                 # model_name: Optional[str] = '/root/autodl-fs/sentence-transformer/all-mpnet-base-v2') -> None:
                 # model_name: Optional[str] = '/root/DrICL/DrICL/NER/') -> None:
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
        print(f"Generated embeddings tensor with shape: {embeddings_tensor.shape}")

        return {
            'embeddings': embeddings_tensor,  # 整个数据集的嵌入张量
            'indices': all_indices  # 对应的原始索引
        }

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

