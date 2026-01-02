import random

import torch
from tqdm import trange
from transformers import BertTokenizer, BertModel
from openicl.icl_dataset_reader import DatasetReader
from openicl.utils.logging import get_logger
from typing import List, Optional
from retriever.base_retriever import BaseRetriever

logger = get_logger(__name__)


class ColBERTRetriever_online(BaseRetriever):
    """支持增量更新的 ColBERT Retriever"""

    def __init__(self,
                 dataset_reader: DatasetReader,
                 ice_separator: Optional[str] = '\n',
                 ice_eos_token: Optional[str] = '\n',
                 prompt_eos_token: Optional[str] = '',
                 ice_num: Optional[int] = 3,
                 index_split: Optional[str] = 'train',
                 model_name: Optional[str] = '/root/autodl-tmp/huggingface/transformers/bert-base-uncased') -> None:
        super().__init__(dataset_reader, ice_separator, ice_eos_token, prompt_eos_token, ice_num, index_split)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.eval()

        # 初始化索引
        self.index_corpus = self._embed_corpus(self.index_ds)

    def _get_embedding(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(
            self.device)
        with torch.no_grad():
            output = self.model(**inputs)
        return output.last_hidden_state.mean(dim=1)  # Mean pooling to get a single vector

    def _compute_cosine_similarity(self, query_embedding: torch.Tensor, index_embeddings: torch.Tensor) -> torch.Tensor:
        """计算查询与索引之间的余弦相似度"""
        query_embedding = query_embedding / query_embedding.norm(p=2, dim=-1, keepdim=True)  # 归一化查询嵌入
        index_embeddings = index_embeddings / index_embeddings.norm(p=2, dim=-1, keepdim=True)  # 归一化索引嵌
        query_embedding = query_embedding.squeeze(0)  # Shape: [768]
        index_embeddings = index_embeddings.squeeze(1)  # Shape: [n, 768]
        # 计算余弦相似度
        similarity_scores = torch.matmul(query_embedding, index_embeddings.T)  # [1, N]
        return similarity_scores

    def _embed_corpus(self, dataset):
        label_embeddings, label_indices = {}, {}
        for idx, item in enumerate(dataset):
            # text = " ".join(item['Text'].split(" ")[:80])
            Premise = item['Premise'].split(" ")[:64]
            Premise = " ".join(Premise)
            Hypothesis = item['Hypothesis'].split(" ")[:64]
            Hypothesis = " ".join(Hypothesis)
            text = Premise + Hypothesis
            label = item['Label']
            embedding = self._get_embedding(text)
            label_embeddings.setdefault(label, []).append(embedding)
            label_indices.setdefault(label, []).append(idx)
        for label in label_embeddings:
            label_embeddings[label] = torch.stack(label_embeddings[label])
        return {'embeddings': label_embeddings, 'indices': label_indices}

    def add_to_index(self, new_item):
        """动态向索引中添加一个样本"""
        # text = " ".join(new_item['Text'].split(" ")[:80])
        Premise = " ".join(new_item['Premise'].split(" ")[:64])
        Hypothesis = " ".join(new_item['Hypothesis'].split(" ")[:64])
        text = Premise + Hypothesis
        label = new_item['Label']
        embedding = self._get_embedding(text)

        if label not in self.index_corpus['embeddings']:
            self.index_corpus['embeddings'][label] = embedding.unsqueeze(0)
            self.index_corpus['indices'][label] = [len(self.index_corpus['indices'].get(label, []))]
        else:
            self.index_corpus['embeddings'][label] = torch.cat(
                [self.index_corpus['embeddings'][label], embedding.unsqueeze(0)], dim=0
            )
            self.index_corpus['indices'][label].append(len(self.index_corpus['indices'][label]))


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

    def retrieve_single_nli(self, text1, text2: str) -> List[int]:
        # for label in self.index_corpus['embeddings']:
        #     print(len(self.index_corpus['indices'][label]))
        text1 = " ".join(text1.split(" ")[:64])
        text2 = " ".join(text2.split(" ")[:64])
        text = text1 + text2
        query_embedding = self._get_embedding(text)
        all_scores, all_indices = [], []
        for label, label_emb in self.index_corpus['embeddings'].items():
            similarity = self._compute_cosine_similarity(query_embedding, label_emb)
            local_indices = similarity.topk(min(int(self.ice_num / 3), len(label_emb))).indices
            global_indices = [self.index_corpus['indices'][label][i] for i in local_indices]
            all_scores.extend(similarity[local_indices].tolist())
            all_indices.extend(global_indices)
        sorted_pairs = sorted(zip(all_scores, all_indices), reverse=True, key=lambda x: x[0])
        return [idx for _, idx in sorted_pairs[:self.ice_num]]



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

        final_indices = all_indices[:self.ice_num]

        return final_indices




