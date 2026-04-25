"""
KGE 三元组判别器
使用 OpenKE 训练的 KGE 模型（如 RotatE）对三元组打分，
通过 sigmoid + 阈值判定正确性
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from typing import List, Dict, Any

from MCTS.base_discriminator import BaseDiscriminator


class KGEDiscriminator(BaseDiscriminator):
    def __init__(
        self,
        model_path: str,
        model_name: str = "RotatE",
        threshold: float = 0.5,
        device: str = None,
        batch_size: int = 512,
    ):
        """
        Args:
            model_path: OpenKE 训练的模型文件路径 (.ckpt / .pth)
            model_name: 模型名称，如 'RotatE'
            threshold: sigmoid 概率阈值，>= threshold 判定为正确
            device: 推理设备
            batch_size: 推理批大小
        """
        self.threshold = threshold
        self.batch_size = batch_size

        from model_calls import OpenKEClient
        self.openke = OpenKEClient(path=model_path, model_name=model_name, rank=0)

        if device:
            self.openke.device = torch.device(device)
            self.openke.kge_model.to(self.openke.device)

    def judge_batch(self, triples_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        使用 KGE 模型批量打分并通过 sigmoid + 阈值判定正确性

        Args:
            triples_list: 每个元素包含 "embedding_ids": [head_id, rel_id, tail_id]

        Returns:
            列表，每个元素包含 "triple_str" 和 "is_correct"
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
                    scores = torch.as_tensor(scores, dtype=torch.float32, device=device)
                probs = torch.sigmoid(scores)

                for j, prob in enumerate(probs):
                    results.append({
                        "triple_str": batch[j].get("input", ""),
                        "is_correct": prob.item() >= self.threshold,
                        "confidence": prob.item(),
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

    # Load test samples
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

    # Create discriminator
    disc = KGEDiscriminator(
        model_path=MODEL_PATH,
        model_name="RotatE",
        threshold=0.5,
        batch_size=512,
        device="cpu",
    )

    # Run inference
    results = disc.judge_batch(test_inputs)

    # Print results
    correct = 0
    for i, (inp, res, gt) in enumerate(zip(test_inputs, results, ground_truth)):
        pred = res["is_correct"]
        match = "✓" if pred == gt else "✗"
        correct += 1 if pred == gt else 0
        # print(f"  [{i+1:>2}] pred={'CORRECT':>8} gt={'CORRECT' if gt else 'INCORRECT':>8} {match}  {inp['input'][:80]}")

    acc = correct / len(samples) * 100
    print(f"\nAccuracy: {correct}/{len(samples)} ({acc:.1f}%)")
    print("=== Test Complete ===")
