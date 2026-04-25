"""
对已有的 discovered_triplets.txt 批量推理获取判别器置信度。
输出 auxiliary_triples_confidence.json 供 KGE 训练时采集指标使用。

用法:
python MCTS/score_auxiliary_confidence.py \
    --triplets MCTS/output/codex-s-kge/20260101_120000/discovered_triplets.txt \
    --entity2id data/CoDEx-S/entity2id.txt \
    --relation2id data/CoDEx-S/relation2id.txt \
    --entity2name data/CoDEx-S/entity2name.txt \
    --model_path openke/output/CoDEx-S/rotate.ckpt \
    --model_name RotatE \
    --output MCTS/output/codex-s-kge/20260101_120000/auxiliary_triples_confidence.json
"""

import argparse
import json
import os
import sys

import torch
from tqdm import tqdm


def load_id_map(path):
    d = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                d[parts[0]] = int(parts[1])
    return d


def load_name_map(path):
    d = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                d[parts[0]] = parts[1]
    return d


def main():
    parser = argparse.ArgumentParser(description="Batch-score auxiliary triplets with KGE discriminator")
    parser.add_argument("--triplets", required=True, help="Path to discovered_triplets.txt")
    parser.add_argument("--entity2id", required=True, help="Path to entity2id.txt")
    parser.add_argument("--relation2id", required=True, help="Path to relation2id.txt")
    parser.add_argument("--entity2name", default=None, help="Path to entity2name.txt")
    parser.add_argument("--model_path", required=True, help="Path to KGE model checkpoint")
    parser.add_argument("--model_name", default="RotatE", help="KGE model name")
    parser.add_argument("--output", required=True, help="Output JSON path for confidence map")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    entity2id = load_id_map(args.entity2id)
    relation2id = load_id_map(args.relation2id)
    entity2name = load_name_map(args.entity2name) if args.entity2name else {}

    triples = []
    with open(args.triplets, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                triples.append((parts[0], parts[1], parts[2]))

    print(f"Loaded {len(triples)} triplets from {args.triplets}")

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from model_calls import OpenKEClient

    client = OpenKEClient(path=args.model_path, model_name=args.model_name, rank=0)
    device = client.device
    if args.device:
        device = torch.device(args.device)
        client.kge_model.to(device)

    confidence_map = {}
    batch_size = args.batch_size
    num_skipped = 0

    with torch.no_grad():
        for start in tqdm(range(0, len(triples), batch_size), desc="Scoring"):
            batch_triples = triples[start:start + batch_size]
            h_list, r_list, t_list = [], [], []

            for head, rel, tail in batch_triples:
                h_id = entity2id.get(head)
                r_id = relation2id.get(rel)
                t_id = entity2id.get(tail)

                if h_id is None or r_id is None or t_id is None:
                    num_skipped += 1
                    continue
                h_list.append(h_id)
                r_list.append(r_id)
                t_list.append(t_id)

            if not h_list:
                continue

            h_t = torch.tensor(h_list, dtype=torch.long, device=device)
            r_t = torch.tensor(r_list, dtype=torch.long, device=device)
            t_t = torch.tensor(t_list, dtype=torch.long, device=device)

            scores = client._predict(h_t, r_t, t_t)
            if not isinstance(scores, torch.Tensor):
                scores = torch.as_tensor(scores, dtype=torch.float32, device=device)
            probs = torch.sigmoid(scores)

            idx = 0
            for head, rel, tail in batch_triples:
                h_id = entity2id.get(head)
                r_id = relation2id.get(rel)
                t_id = entity2id.get(tail)

                if h_id is None or r_id is None or t_id is None:
                    continue

                confidence = probs[idx].item()
                key = f"{head}\t{rel}\t{tail}"
                confidence_map[key] = round(confidence, 6)
                idx += 1

    if num_skipped > 0:
        print(f"Warning: skipped {num_skipped} triplets due to missing IDs")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(confidence_map, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(confidence_map)} confidence entries to {args.output}")

    if confidence_map:
        vals = list(confidence_map.values())
        print(f"Confidence stats: min={min(vals):.4f}, max={max(vals):.4f}, "
              f"mean={sum(vals)/len(vals):.4f}")


if __name__ == "__main__":
    main()
