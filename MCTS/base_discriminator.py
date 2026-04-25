"""
三元组判别器抽象基类
所有判别器（LLM、KG-BERT等）均需实现此接口
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseDiscriminator(ABC):
    """三元组判别器抽象基类"""

    @abstractmethod
    def judge_batch(self, triples_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        批量判断三元组的正确性

        Args:
            triples_list: 每个元素为字典，至少包含:
                - "input": 文本描述（用于需要文本输入的判别器）
                - "embedding_ids": [head_id, relation_id, tail_id]（用于需要嵌入的判别器）
                具体字段由各判别器实现决定是否使用

        Returns:
            列表，每个元素为字典:
                - "triple_str": 三元组的文本描述
                - "is_correct": bool, 判别器认为该三元组是否正确
                - "confidence": float, 判别器的置信度，范围 [0, 1]
        """
        pass
