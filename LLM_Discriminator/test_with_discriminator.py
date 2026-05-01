import os
import argparse
import json
import numpy as np
import torch
import torch.distributed as dist
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


def load_test_dataset(path):
    with open(path, "r") as f:
        test_dataset = json.load(f)
    return test_dataset


def _compute_metrics(answer, predict):
    acc = accuracy_score(y_true=answer, y_pred=predict)
    p = precision_score(y_true=answer, y_pred=predict, zero_division=0)
    r = recall_score(y_true=answer, y_pred=predict, zero_division=0)
    f1 = f1_score(y_true=answer, y_pred=predict, zero_division=0)
    tp = sum(1 for a, pr in zip(answer, predict) if a == 1 and pr == 1)
    fp = sum(1 for a, pr in zip(answer, predict) if a == 0 and pr == 1)
    fn = sum(1 for a, pr in zip(answer, predict) if a == 1 and pr == 0)
    tn = sum(1 for a, pr in zip(answer, predict) if a == 0 and pr == 0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return acc, p, r, f1, fpr


def main(args):
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(device)

    embedding_path = "{}/embeddings.pth".format(args.lora_weights)
    discriminator = TriplesDiscriminator(
        llm_path=args.base_model,
        lora_path=args.lora_weights,
        embedding_path=embedding_path,
        device=device,
        dtype="fp16",
        batch_size=args.batch_size
    )

    if local_rank == 0:
        print(f"Loading test data from {args.test_data}...")
    test_dataset = load_test_dataset(args.test_data)
    local_data_slice = test_dataset[local_rank::world_size]

    # === Step 1: 获取所有样本的 P(True) ===
    if local_rank == 0:
        print(f"Running inference on {world_size} devices...")
    predictions = discriminator.judge_batch(local_data_slice)

    local_scores = []
    local_labels = []
    for i, pred_result in enumerate(predictions):
        local_scores.append(pred_result["confidence"])
        local_labels.append(1 if "True" in local_data_slice[i]["output"] else 0)

    local_results = list(zip(local_scores, local_labels))
    gathered_results = [None] * world_size
    dist.barrier()
    dist.all_gather_object(gathered_results, local_results)

    if local_rank != 0:
        return

    all_scores_labels = [item for sublist in gathered_results for item in sublist]
    all_scores = np.array([s for s, _ in all_scores_labels])
    all_labels = np.array([l for _, l in all_scores_labels])

    print(f"\nTotal samples: {len(all_scores)}")
    print(f"  Positive: {int(all_labels.sum())}, Negative: {int(len(all_labels) - all_labels.sum())}")
    print(f"  Score stats: min={all_scores.min():.4f}, max={all_scores.max():.4f}, "
          f"mean={all_scores.mean():.4f}, median={np.median(all_scores):.4f}")

    # === Step 2: 默认阈值 0.5 的结果 ===
    preds_default = (all_scores >= 0.5).astype(int)
    acc, p, r, f1, fpr = _compute_metrics(all_labels.tolist(), preds_default.tolist())
    print(f"\n{'='*60}")
    print(f"  Default threshold = 0.5")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {p:.4f}")
    print(f"  Recall:    {r:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  FPR:       {fpr:.4f} ({fpr*100:.1f}%)")
    print(f"{'='*60}")

    # === Step 3: 校准 - 搜索最优阈值 ===
    best_f1 = 0.0
    best_thresh = 0.5
    candidate_thresholds = np.percentile(all_scores, np.arange(5, 96, 2))
    for thresh in candidate_thresholds:
        preds = (all_scores >= thresh).astype(int)
        tp = np.sum((preds == 1) & (all_labels == 1))
        fp = np.sum((preds == 1) & (all_labels == 0))
        fn = np.sum((preds == 0) & (all_labels == 1))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_t = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        if f1_t > best_f1:
            best_f1 = f1_t
            best_thresh = float(thresh)

    preds_calibrated = (all_scores >= best_thresh).astype(int)
    acc, p, r, f1, fpr = _compute_metrics(all_labels.tolist(), preds_calibrated.tolist())
    print(f"\n{'='*60}")
    print(f"  Calibrated threshold = {best_thresh:.4f} (best F1 search)")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {p:.4f}")
    print(f"  Recall:    {r:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  FPR:       {fpr:.4f} ({fpr*100:.1f}%)")
    print(f"{'='*60}")

    # === Step 4: 模拟 MCTS 场景 — 极度不平衡数据 ===
    # 假设 MCTS 叶节点中只有 5% 是正确的（更接近实际场景）
    for imbalance_ratio in [0.10, 0.05, 0.02]:
        positive_mask = all_labels == 1
        negative_mask = all_labels == 0
        n_pos = int(positive_mask.sum())
        n_neg = int(n_pos / imbalance_ratio - n_pos)
        if n_neg > int(negative_mask.sum()):
            n_neg = int(negative_mask.sum())
        neg_indices = np.where(negative_mask)[0]
        np.random.seed(42)
        sampled_neg = np.random.choice(neg_indices, size=min(n_neg, len(neg_indices)), replace=False)
        selected = np.concatenate([np.where(positive_mask)[0], sampled_neg])
        imb_scores = all_scores[selected]
        imb_labels = all_labels[selected]

        for thresh_label, thresh in [("0.5", 0.5), ("calibrated", best_thresh)]:
            preds = (imb_scores >= thresh).astype(int)
            tp = np.sum((preds == 1) & (imb_labels == 1))
            fp = np.sum((preds == 1) & (imb_labels == 0))
            fn = np.sum((preds == 0) & (imb_labels == 1))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_t = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            fpr_val = fp / (fp + np.sum(imb_labels == 0)) if np.sum(imb_labels == 0) > 0 else 0.0
            print(f"  Imbalance=1:{int(1/imbalance_ratio)-1:3d} | thresh={thresh_label:>10s} | "
                  f"TP={tp:4d} FP={fp:4d} FN={fn:3d} | "
                  f"P={prec:.3f} R={rec:.3f} F1={f1_t:.3f} FPR={fpr_val:.3f}")

    # === Step 5: 不同阈值下的完整扫描表 ===
    print(f"\n{'='*60}")
    print(f"  Threshold scan (full test set)")
    print(f"{'='*60}")
    print(f"  {'Thresh':>8s} | {'Prec':>6s} {'Recall':>6s} {'F1':>6s} {'FPR':>6s} | {'FP':>5s} {'FN':>5s}")
    print(f"  {'-'*8}-+-{'-'*22}-+-{'-'*11}")
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        preds = (all_scores >= thresh).astype(int)
        tp = np.sum((preds == 1) & (all_labels == 1))
        fp = np.sum((preds == 1) & (all_labels == 0))
        fn = np.sum((preds == 0) & (all_labels == 1))
        tn = np.sum((preds == 0) & (all_labels == 0))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_t = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        marker = " <-- calib" if abs(thresh - round(best_thresh, 1)) < 0.05 else ""
        print(f"  {thresh:>8.2f} | {prec:.4f} {rec:.4f} {f1_t:.4f} {fpr_val:.4f} | {fp:5d} {fn:5d}{marker}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True, help="Path to the base model")
    parser.add_argument("--test_data", type=str, required=True, help="Path to the test dataset")
    parser.add_argument("--lora_weights", type=str, required=True, help="Path to the LoRA weights")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for the discriminator inference")
    parser.add_argument("--root_dir", type=str, default=".", help="Root directory for saving outputs")
    args = parser.parse_args()

    import sys
    sys.path.append(args.root_dir)
    from discriminator import TriplesDiscriminator

    main(args)
