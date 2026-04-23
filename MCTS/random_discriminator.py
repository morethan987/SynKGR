"""
随机三元组判别器
以固定概率随机判定三元组正确性，用于消融实验
"""

import random
from typing import List, Dict, Any

from MCTS.base_discriminator import BaseDiscriminator


class RandomDiscriminator(BaseDiscriminator):
    def __init__(self, positive_rate: float = 0.5):
        """
        Args:
            positive_rate: 随机判定为正确的概率，默认 0.5
        """
        self.positive_rate = positive_rate

    def judge_batch(self, triples_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {
                "triple_str": item.get("input", ""),
                "is_correct": random.random() < self.positive_rate,
            }
            for item in triples_list
        ]
