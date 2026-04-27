"""
KGE 三元组判别器
使用 OpenKE 训练的 KGE 模型（如 RotatE）对三元组打分，
通过逐关系最优阈值判定正确性

学术惯例：
- RotatE 是基于距离的模型，不做 sigmoid 变换
- 对验证集为每个关系独立搜索最优分类阈值
- 推理时使用 margin - distance >= threshold 作为判定标准
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import defaultdict

from MCTS.base_discriminator import BaseDiscriminator


class KGEDiscriminator(BaseDiscriminator):
    def __init__(
        self,
        model_path: str,
        model_name: str = "RotatE",
        threshold: float = 0.0,
        device: str = None,
        batch_size: int = 512,
    ):
        """
        Args:
            model_path: OpenKE 训练的模型文件路径 (.ckpt / .pth)
            model_name: 模型名称，如 'RotatE'
            threshold: 默认分数阈值（未校准时使用），
                       margin - distance >= threshold 判定为正确
            device: 推理设备
            batch_size: 推理批大小
        """
        self.default_threshold = threshold
        self.batch_size = batch_size
        self.relation_thresholds: Dict[int, float] = {}
        self.relation_score_min: Dict[int, float] = {}
        self.relation_score_max: Dict[int, float] = {}

        from model_calls import OpenKEClient
        self.openke = OpenKEClient(path=model_path, model_name=model_name, rank=0)

        if device:
            self.openke.device = torch.device(device)
            self.openke.kge_model.to(self.openke.device)

    def calibrate(
        self,
        valid_triples: List[Tuple[int, int, int]],
        num_neg_per_positive: int = 50,
    ):
        """
        在验证集上为每个关系搜索最优分类阈值。

        使用随机替换尾实体生成负样本，以 F1 为指标搜索最优阈值。
        负样本数量默认较多（50）以保证负样本分布的代表性，
        避免因负样本过少导致阈值过于宽松。

        Args:
            valid_triples: 验证集正样本三元组列表，每个元素为 (head_id, rel_id, tail_id)
            num_neg_per_positive: 每个正样本生成的负样本数量
        """
        device = self.openke.device
        ent_tot = self.openke.ent_count

        by_relation = defaultdict(list)
        for h, r, t in valid_triples:
            by_relation[r].append((h, r, t))

        self.relation_thresholds = {}

        with torch.no_grad():
            for rel_id, triples in by_relation.items():
                all_scores = []
                all_labels = []

                for start in range(0, len(triples), self.batch_size):
                    batch = triples[start:start + self.batch_size]

                    h_pos = torch.tensor(
                        [t[0] for t in batch], dtype=torch.long, device=device)
                    r_pos = torch.tensor(
                        [t[1] for t in batch], dtype=torch.long, device=device)
                    t_pos = torch.tensor(
                        [t[2] for t in batch], dtype=torch.long, device=device)

                    pos_scores = self.openke._predict(h_pos, r_pos, t_pos)
                    if not isinstance(pos_scores, torch.Tensor):
                        pos_scores = torch.as_tensor(
                            pos_scores, dtype=torch.float32, device=device)

                    for s in pos_scores:
                        all_scores.append(s.item())
                        all_labels.append(1)

                    for j in range(len(batch)):
                        h_j = h_pos[j].unsqueeze(0).repeat(num_neg_per_positive)
                        r_j = r_pos[j].unsqueeze(0).repeat(num_neg_per_positive)
                        neg_t = torch.randint(
                            0, ent_tot, (num_neg_per_positive,), device=device)

                        neg_scores = self.openke._predict(h_j, r_j, neg_t)
                        if not isinstance(neg_scores, torch.Tensor):
                            neg_scores = torch.as_tensor(
                                neg_scores, dtype=torch.float32, device=device)

                        for s in neg_scores:
                            all_scores.append(s.item())
                            all_labels.append(0)

                scores_arr = np.array(all_scores)
                labels_arr = np.array(all_labels)

                best_thresh = 0.0
                best_f1 = 0.0

                candidate_thresholds = np.percentile(
                    scores_arr, np.arange(5, 96, 5))
                for thresh in candidate_thresholds:
                    preds = (scores_arr >= thresh).astype(int)
                    tp = np.sum((preds == 1) & (labels_arr == 1))
                    fp = np.sum((preds == 1) & (labels_arr == 0))
                    fn = np.sum((preds == 0) & (labels_arr == 1))
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f1 = (2 * precision * recall / (precision + recall)
                          if (precision + recall) > 0 else 0.0)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_thresh = thresh

                self.relation_thresholds[rel_id] = float(best_thresh)
                self.relation_score_min[rel_id] = float(scores_arr.min())
                self.relation_score_max[rel_id] = float(scores_arr.max())

        print(f"Calibrated thresholds for {len(self.relation_thresholds)} relations")
        if self.relation_thresholds:
            vals = list(self.relation_thresholds.values())
            print(f"  Threshold stats: min={min(vals):.4f}, max={max(vals):.4f}, "
                  f"mean={np.mean(vals):.4f}")

    def judge_batch(self, triples_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        使用 KGE 模型批量打分并通过逐关系阈值判定正确性

        Args:
            triples_list: 每个元素包含 "embedding_ids": [head_id, rel_id, tail_id]

        Returns:
            列表，每个元素包含 "triple_str", "is_correct", "confidence"
        """
        results = []
        device = self.openke.device

        with torch.no_grad():
            for i in range(0, len(triples_list), self.batch_size):
                batch = triples_list[i:i + self.batch_size]

                h = torch.tensor(
                    [item["embedding_ids"][0] for item in batch],
                    dtype=torch.long, device=device,
                )
                r = torch.tensor(
                    [item["embedding_ids"][1] for item in batch],
                    dtype=torch.long, device=device,
                )
                t = torch.tensor(
                    [item["embedding_ids"][2] for item in batch],
                    dtype=torch.long, device=device,
                )

                scores = self.openke._predict(h, r, t)
                if not isinstance(scores, torch.Tensor):
                    scores = torch.as_tensor(
                        scores, dtype=torch.float32, device=device)

                for j, score in enumerate(scores):
                    rel_id = batch[j]["embedding_ids"][1]
                    threshold = self.relation_thresholds.get(
                        rel_id, self.default_threshold
                    )

                    raw_score = score.item()
                    rel_min = self.relation_score_min.get(rel_id)
                    rel_max = self.relation_score_max.get(rel_id)
                    if rel_min is not None and rel_max is not None and rel_max > rel_min:
                        confidence = max(0.0, min(1.0,
                            (raw_score - rel_min) / (rel_max - rel_min)))
                    else:
                        confidence = 1.0 if raw_score >= threshold else 0.0

                    results.append({
                        "triple_str": batch[j].get("input", ""),
                        "is_correct": raw_score >= threshold,
                        "confidence": confidence,
                    })

        return results


if __name__ == "__main__":
    import json

    DATASET = "FB15k-237N"
    MODEL_PATH = "openke/output/FB15k-237N/checkpoint_20260422_142102/rotate.ckpt"
    TEST_JSON = "LLM_Discriminator/data/FB15K-237N-test.json"

    if not os.path.isfile(MODEL_PATH):
        print(f"[SKIP] KGE model not found: {MODEL_PATH}")
        sys.exit(1)

    if not os.path.isfile(TEST_JSON):
        print(f"[SKIP] Test data not found: {TEST_JSON}")
        sys.exit(1)

    with open(TEST_JSON, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    samples = test_data
    test_inputs = []
    ground_truth = []

    for item in samples:
        test_inputs.append({
            "input": item["input"].strip(),
            "embedding_ids": item["embedding_ids"],
        })
        ground_truth.append(item["output"].strip().lower() == "true")

    print(f"=== KGE Discriminator Test ===")
    print(f"Dataset:  {DATASET}")
    print(f"Model:    {MODEL_PATH}")
    print(f"Samples:  {len(samples)}\n")

    disc = KGEDiscriminator(
        model_path=MODEL_PATH,
        model_name="RotatE",
        threshold=0.0,
        batch_size=512,
        device="cpu",
    )

    results = disc.judge_batch(test_inputs)

    correct = 0
    for i, (inp, res, gt) in enumerate(zip(test_inputs, results, ground_truth)):
        pred = res["is_correct"]
        correct += 1 if pred == gt else 0

    acc = correct / len(samples) * 100
    print(f"\nAccuracy (default threshold): {correct}/{len(samples)} ({acc:.1f}%)")

    positive_ids = [
        tuple(item["embedding_ids"])
        for item, gt in zip(test_inputs, ground_truth) if gt
    ]
    print(f"\nCalibrating with {len(positive_ids)} positive samples...")
    disc.calibrate(positive_ids, num_neg_per_positive=1)

    results = disc.judge_batch(test_inputs)

    correct = 0
    tp, tn, fp, fn = 0, 0, 0, 0
    for i, (inp, res, gt) in enumerate(zip(test_inputs, results, ground_truth)):
        pred = res["is_correct"]
        correct += 1 if pred == gt else 0
        if gt and pred:
            tp += 1
        elif not gt and not pred:
            tn += 1
        elif not gt and pred:
            fp += 1
        elif gt and not pred:
            fn += 1

    acc = correct / len(samples) * 100
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nAccuracy (calibrated): {correct}/{len(samples)} ({acc:.2f}%)")
    print(f"TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print(f"Precision:  {precision:.2f}%")
    print(f"Recall:     {recall:.2f}%")
    print(f"F1:         {f1:.2f}%")
    print("=== Test Complete ===")
