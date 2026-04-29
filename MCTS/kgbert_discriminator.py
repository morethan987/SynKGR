"""
KG-BERT 三元组判别器
使用训练好的 KG-BERT 模型判断三元组正确性
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
from typing import List, Dict, Any

from MCTS.base_discriminator import BaseDiscriminator


class KGBERTDiscriminator(BaseDiscriminator):
    def __init__(
        self,
        model_dir: str,
        data_dir: str,
        max_seq_length: int = 128,
        batch_size: int = 32,
        do_lower_case: bool = True,
        device: str = None,
    ):
        """
        Args:
            model_dir: 训练好的 KG-BERT 模型目录（包含 pytorch_model.bin, config.json, vocab.txt）
            data_dir: KG-BERT 格式的数据目录（包含 entity2text.txt, relation2text.txt 等）
            max_seq_length: 最大序列长度
            batch_size: 推理批大小
            device: 推理设备
        """
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.do_lower_case = do_lower_case

        self._load_model_and_resources()

    def _load_model_and_resources(self):
        """加载模型、tokenizer 和实体/关系文本映射"""
        from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig  # type: ignore[import-untyped]
        from pytorch_pretrained_bert.tokenization import BertTokenizer  # type: ignore[import-untyped]

        print(f"Loading KG-BERT model from {self.model_dir} to device: {self.device}")

        num_labels = 2
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_dir, num_labels=num_labels
        )
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = BertTokenizer.from_pretrained(
            self.model_dir, do_lower_case=self.do_lower_case
        )

        self.ent2text = self._load_tsv_dict(
            os.path.join(self.data_dir, "entity2text.txt")
        )
        self.rel2text = self._load_tsv_dict(
            os.path.join(self.data_dir, "relation2text.txt")
        )

        print("KG-BERT model and resources loaded successfully.")

    @staticmethod
    def _load_tsv_dict(path: str) -> Dict[str, str]:
        result = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t", 1)
                if len(parts) == 2:
                    result[parts[0]] = parts[1]
        return result

    def _triple_to_features(self, head: str, relation: str, tail: str) -> Dict[str, Any]:
        """
        将三元组 (entity_id 形式) 转换为 BERT 输入特征

        格式: [CLS] head_text [SEP] relation_text [SEP] tail_text [SEP]
        """
        head_text = self.ent2text.get(head, head.replace("_", " "))
        rel_text = self.rel2text.get(relation, relation.replace("/", " ").replace("_", " "))
        tail_text = self.ent2text.get(tail, tail.replace("_", " "))

        tokens_a = self.tokenizer.tokenize(head_text)
        tokens_b = self.tokenizer.tokenize(rel_text)
        tokens_c = self.tokenizer.tokenize(tail_text)

        self._truncate_seq_triple(tokens_a, tokens_b, tokens_c, self.max_seq_length - 4)

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        tokens += tokens_b + ["[SEP]"]
        segment_ids += [1] * (len(tokens_b) + 1)

        tokens += tokens_c + ["[SEP]"]
        segment_ids += [0] * (len(tokens_c) + 1)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        padding = [0] * (self.max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        return {
            "input_ids": input_ids,
            "input_mask": input_mask,
            "segment_ids": segment_ids,
        }

    @staticmethod
    def _truncate_seq_triple(tokens_a, tokens_b, tokens_c, max_length):
        while True:
            total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b) and len(tokens_a) > len(tokens_c):
                tokens_a.pop()
            elif len(tokens_b) > len(tokens_a) and len(tokens_b) > len(tokens_c):
                tokens_b.pop()
            elif len(tokens_c) > len(tokens_a) and len(tokens_c) > len(tokens_b):
                tokens_c.pop()
            else:
                tokens_c.pop()

    def judge_batch(self, triples_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        使用 KG-BERT 批量判断三元组正确性

        Args:
            triples_list: 每个元素包含:
                - "input": 文本描述（用于记录）
                - "embedding_ids": [head_entity_idx, relation_idx, tail_entity_idx]
                  注意: 这里使用 data_loader 中的 id 来查找对应的 KG-BERT 实体/关系文本

        Returns:
            列表，每个元素包含 "triple_str" 和 "is_correct"
        """
        results = []

        with torch.no_grad():
            for i in range(0, len(triples_list), self.batch_size):
                batch_data = triples_list[i:i + self.batch_size]

                batch_input_ids = []
                batch_input_mask = []
                batch_segment_ids = []
                batch_triple_strs = []

                for item in batch_data:
                    head_idx, rel_idx, tail_idx = item["embedding_ids"]
                    head_ent = self._idx_to_entity.get(head_idx, str(head_idx))
                    rel_ent = self._idx_to_relation.get(rel_idx, str(rel_idx))
                    tail_ent = self._idx_to_entity.get(tail_idx, str(tail_idx))

                    features = self._triple_to_features(head_ent, rel_ent, tail_ent)
                    batch_input_ids.append(features["input_ids"])
                    batch_input_mask.append(features["input_mask"])
                    batch_segment_ids.append(features["segment_ids"])
                    batch_triple_strs.append(item.get("input", ""))

                input_ids_t = torch.LongTensor(batch_input_ids).to(self.device)
                input_mask_t = torch.LongTensor(batch_input_mask).to(self.device)
                segment_ids_t = torch.LongTensor(batch_segment_ids).to(self.device)

                logits = self.model(input_ids_t, segment_ids_t, input_mask_t)
                probs = torch.softmax(logits, dim=-1)

                for j in range(len(batch_data)):
                    confidence = probs[j, 1].item()
                    results.append({
                        "triple_str": batch_triple_strs[j],
                        "is_correct": confidence >= 0.7,
                    })

        return results

    def set_id_mappings(self, id2entity: Dict[int, str], id2relation: Dict[int, str]):
        """
        设置从 embedding index 到 KG-BERT 实体/关系 ID 的映射

        Args:
            id2entity: {entity_idx: entity_id_str} （来自 data_loader.id2entity）
            id2relation: {relation_idx: relation_id_str} （来自 data_loader）
        """
        self._idx_to_entity = id2entity
        self._idx_to_relation = id2relation


if __name__ == "__main__":
    import json

    DATASET = "CoDEx-S"
    DATA_DIR = f"data/{DATASET}"
    KGBERT_MODEL_DIR = f"kg-bert/output/triple_classifier_{DATASET}"
    KGBERT_DATA_DIR = f"kg-bert/data/{DATASET}"
    TEST_TSV = f"{KGBERT_DATA_DIR}/test.tsv"

    if not os.path.isfile(f"{KGBERT_MODEL_DIR}/pytorch_model.bin"):
        print(f"[SKIP] KG-BERT model not found: {KGBERT_MODEL_DIR}/pytorch_model.bin")
        print("  Run: bash scripts/train_kgbert_triple_classifier.sh")
        sys.exit(1)

    if not os.path.isfile(TEST_TSV):
        print(f"[SKIP] Test data not found: {TEST_TSV}")
        sys.exit(1)

    # Load id mappings: entity_id_str -> int, relation_id_str -> int
    entity2id = {}
    with open(f"{DATA_DIR}/entity2id.txt") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                entity2id[parts[0]] = int(parts[1])

    relation2id = {}
    with open(f"{DATA_DIR}/relation2id.txt") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                relation2id[parts[0]] = int(parts[1])

    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}

    # Load test.tsv (head_id, rel_id, tail_id, label) — includes positive AND negative triples
    test_inputs = []
    ground_truth = []

    with open(TEST_TSV, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            head_ent, rel, tail_ent = parts[0], parts[1], parts[2]
            label = parts[3] if len(parts) > 3 else "1"

            h_id = entity2id.get(head_ent)
            r_id = relation2id.get(rel)
            t_id = entity2id.get(tail_ent)

            if h_id is None or r_id is None or t_id is None:
                continue

            test_inputs.append({
                "input": f"( {head_ent}, {rel}, {tail_ent} )",
                "embedding_ids": [h_id, r_id, t_id],
            })
            ground_truth.append(label == "1")

    num_pos = sum(ground_truth)
    num_neg = len(ground_truth) - num_pos

    print(f"=== KG-BERT Discriminator Test ===")
    print(f"Dataset:  {DATASET}")
    print(f"Model:    {KGBERT_MODEL_DIR}")
    print(f"Samples:  {len(test_inputs)} (positive: {num_pos}, negative: {num_neg})\n")

    # Create discriminator
    disc = KGBERTDiscriminator(
        model_dir=KGBERT_MODEL_DIR,
        data_dir=KGBERT_DATA_DIR,
        max_seq_length=128,
        batch_size=256,
        do_lower_case=True,
        device="cpu",
    )
    disc.set_id_mappings(id2entity=id2entity, id2relation=id2relation)

    # Run inference
    results = disc.judge_batch(test_inputs)

    # Print results
    correct = 0
    tp, tn, fp, fn = 0, 0, 0, 0
    for i, (inp, res, gt) in enumerate(zip(test_inputs, results, ground_truth)):
        pred = res["is_correct"]
        match_str = "✓" if pred == gt else "✗"
        correct += 1 if pred == gt else 0
        if gt and pred:
            tp += 1
        elif not gt and not pred:
            tn += 1
        elif not gt and pred:
            fp += 1
        elif gt and not pred:
            fn += 1

    acc = correct / len(test_inputs) * 100
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Accuracy:   {correct}/{len(test_inputs)} ({acc:.2f}%)")
    print(f"TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print(f"Precision:  {precision:.2f}%")
    print(f"Recall:     {recall:.2f}%")
    print(f"F1:         {f1:.2f}%")
    print("=== Test Complete ===")
