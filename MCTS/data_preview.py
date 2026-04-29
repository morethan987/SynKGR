import json

import torch
import numpy as np


def load_triples(file_path):
    triples = set()
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                triples.add((parts[0], parts[1], parts[2]))
    return triples


def load_auxiliary_triples(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    triples_with_conf = {}
    for key, conf in data.items():
        parts = key.split("\t")
        if len(parts) == 3:
            triples_with_conf[(parts[0], parts[1], parts[2])] = conf
    return triples_with_conf


def print_confidence_distribution(confidences, label):
    arr = np.array(confidences)
    if len(arr) == 0:
        print(f"\n[{label}] 无重叠三元组")
        return

    print(f"\n{'='*60}")
    print(f"[{label}] 置信度分布 (共 {len(arr)} 个重叠三元组)")
    print(f"{'='*60}")
    print(f"  最小值:  {arr.min():.6f}")
    print(f"  最大值:  {arr.max():.6f}")
    print(f"  均值:    {arr.mean():.6f}")
    print(f"  中位数:  {np.median(arr):.6f}")
    print(f"  标准差:  {arr.std():.6f}")

    percentiles = [10, 25, 50, 75, 90]
    p_values = np.percentile(arr, percentiles)
    for p, v in zip(percentiles, p_values):
        print(f"  P{p}:     {v:.6f}")

    num_bins = 10
    hist, bins = np.histogram(arr, bins=num_bins)
    max_hist = max(hist) if hist.size > 0 else 1
    print(f"\n  直方图:")
    for i in range(len(hist)):
        bar_length = int((hist[i] / max_hist) * 40) if max_hist > 0 else 0
        print(f"    {bins[i]:.3f} - {bins[i+1]:.3f}: {'█' * bar_length} ({hist[i]})")


def analyze_auxiliary_triples_overlap(
    auxiliary_json_path,
    test_path,
    valid_path,
    train_path=None,
):
    """
    分析辅助三元组与已知数据集的重叠情况

    auxiliary_json_path: KG-BERT 打分输出的 JSON 文件路径
        格式: {"head\\trel\\ttail": confidence, ...}
    """
    aux_triples = load_auxiliary_triples(auxiliary_json_path)
    test_triples = load_triples(test_path)
    valid_triples = load_triples(valid_path)

    aux_set = set(aux_triples.keys())
    print(f"辅助三元组总数: {len(aux_set)}")
    print(f"测试集三元组总数: {len(test_triples)}")
    print(f"验证集三元组总数: {len(valid_triples)}")

    if train_path:
        train_triples = load_triples(train_path)
        print(f"训练集三元组总数: {len(train_triples)}")

    # 与测试集重叠
    test_overlap = aux_set & test_triples
    test_overlap_conf = [aux_triples[t] for t in test_overlap]
    print(f"\n与测试集重叠: {len(test_overlap)} / {len(test_triples)} "
          f"({len(test_overlap)/len(test_triples)*100:.2f}%)")
    print_confidence_distribution(test_overlap_conf, "测试集重叠")

    # 与验证集重叠
    valid_overlap = aux_set & valid_triples
    valid_overlap_conf = [aux_triples[t] for t in valid_overlap]
    print(f"\n与验证集重叠: {len(valid_overlap)} / {len(valid_triples)} "
          f"({len(valid_overlap)/len(valid_triples)*100:.2f}%)")
    print_confidence_distribution(valid_overlap_conf, "验证集重叠")

    # 与测试集+验证集重叠
    test_valid = test_triples | valid_triples
    tv_overlap = aux_set & test_valid
    tv_overlap_conf = [aux_triples[t] for t in tv_overlap]
    print(f"\n与测试集+验证集重叠: {len(tv_overlap)} / {len(test_valid)} "
          f"({len(tv_overlap)/len(test_valid)*100:.2f}%)")
    print_confidence_distribution(tv_overlap_conf, "测试集+验证集重叠")

    # 与训练集重叠
    if train_path:
        train_overlap = aux_set & train_triples
        train_overlap_conf = [aux_triples[t] for t in train_overlap]
        print(f"\n与训练集重叠: {len(train_overlap)} / {len(train_triples)} "
              f"({len(train_overlap)/len(train_triples)*100:.2f}%)")
        print_confidence_distribution(train_overlap_conf, "训练集重叠")

    # 未出现在任何已知集合中的辅助三元组
    if train_path:
        known = test_triples | valid_triples | train_triples
        unknown = aux_set - known
        unknown_conf = [aux_triples[t] for t in unknown]
        print(f"\n未出现在训练/验证/测试集中的辅助三元组: {len(unknown)} "
              f"({len(unknown)/len(aux_set)*100:.2f}%)")
        print_confidence_distribution(unknown_conf, "未知三元组")

    # 打印部分重叠样本
    print(f"\n{'='*60}")
    print("测试集重叠样本 (前10个):")
    print(f"{'='*60}")
    for t in sorted(test_overlap, key=lambda x: aux_triples[x], reverse=True)[:10]:
        print(f"  {t[0]}\t{t[1]}\t{t[2]}\t置信度: {aux_triples[t]:.6f}")

    print(f"\n验证集重叠样本 (前10个):")
    print(f"{'='*60}")
    for t in sorted(valid_overlap, key=lambda x: aux_triples[x], reverse=True)[:10]:
        print(f"  {t[0]}\t{t[1]}\t{t[2]}\t置信度: {aux_triples[t]:.6f}")


def relation_cnt_distribution(file_path):
    # 加载数据
    datas = torch.load(file_path)

    # 用来存储每个实体的关系数量
    entity_relation_counts = []

    # 遍历数据，计算每个实体对应的关系数量
    for entity_id, relations_list in datas.items():
        entity_relation_counts.append(len(relations_list))

    # 将关系数量转换为numpy数组
    entity_relation_counts_array = np.array(entity_relation_counts)

    # 打印总关系数量
    print(f"Total number of entities: {len(entity_relation_counts_array)}")
    print(f"Total number of relations: {entity_relation_counts_array.sum()}")

    # 计算每个关系数量出现的频率
    hist, bins = np.histogram(entity_relation_counts_array, bins=5)

    # 打印分布
    max_hist = max(hist) if hist.size > 0 else 1
    for i in range(len(hist)):
        # 根据频次大小调整字符数量，最大不超过50个字符
        bar_length = int((hist[i] / max_hist) * 50) if max_hist > 0 else 0
        print(f"{bins[i]:.2f} - {bins[i+1]:.2f}: {'█' * bar_length} ({hist[i]})")

    # 打印样本
    print("Sample entity relation counts:\n")
    for data in list(datas.items())[:3]:
        print(data)


if __name__ == "__main__":
    # cdko && ackopa && python MCTS/data_preview.py

    # relation_cnt_distribution("MCTS/output/fb15k-237n/processed_data.pth")

    analyze_auxiliary_triples_overlap(
        auxiliary_json_path="MCTS/output/codex-s-kge/20260424_223000/auxiliary_triples_confidence_kgbert.json",
        test_path="data/CoDEx-S/test.txt",
        valid_path="data/CoDEx-S/valid.txt",
        train_path="data/CoDEx-S/train.txt",
    )
