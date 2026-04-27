"""
使用 KG-BERT 作为第三方判别器对 discovered_triplets.txt 评分。
输出 auxiliary_triples_confidence_kgbert.json 供下游使用。

置信度 = P(correct | triple) — KG-BERT 的 softmax 正类概率
理论上值域 [0, 1]，具有良好的区分度。

用法:
python MCTS/score_auxiliary_confidence_kgbert.py \
    --triplets MCTS/output/codex-s-kge/20260101_120000/discovered_triplets.txt \
    --kgbert_model_dir kg-bert/output/triple_classifier_CoDEx-S \
    --kgbert_data_dir kg-bert/data/CoDEx-S \
    --output MCTS/output/codex-s-kge/20260101_120000/auxiliary_triples_confidence_kgbert.json
"""

import argparse
import json
import os
import sys

import torch
import numpy as np
from tqdm import tqdm


def load_id_map(path):
    d = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                d[parts[0]] = int(parts[1])
    return d


def load_tsv_dict(path):
    result = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                result[parts[0]] = parts[1]
    return result


class KGBERTScorer:
    def __init__(self, model_dir, data_dir, max_seq_length=128,
                 batch_size=256, do_lower_case=True, device=None):
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        from pytorch_pretrained_bert.modeling import BertForSequenceClassification
        from pytorch_pretrained_bert.tokenization import BertTokenizer

        print(f"Loading KG-BERT model from {model_dir} to device: {self.device}")
        self.model = BertForSequenceClassification.from_pretrained(
            model_dir, num_labels=2
        )
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = BertTokenizer.from_pretrained(
            model_dir, do_lower_case=do_lower_case
        )

        self.ent2text = load_tsv_dict(os.path.join(data_dir, "entity2text.txt"))
        self.rel2text = load_tsv_dict(os.path.join(data_dir, "relation2text.txt"))
        print(f"KG-BERT loaded: {len(self.ent2text)} entities, "
              f"{len(self.rel2text)} relations")

    def _truncate_seq_triple(self, tokens_a, tokens_b, tokens_c, max_length):
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

    def _triple_to_features(self, head_ent, rel_ent, tail_ent):
        head_text = self.ent2text.get(head_ent, head_ent.replace("_", " "))
        rel_text = self.rel2text.get(rel_ent, rel_ent.replace("/", " ").replace("_", " "))
        tail_text = self.ent2text.get(tail_ent, tail_ent.replace("_", " "))

        tokens_a = self.tokenizer.tokenize(head_text)
        tokens_b = self.tokenizer.tokenize(rel_text)
        tokens_c = self.tokenizer.tokenize(tail_text)

        self._truncate_seq_triple(tokens_a, tokens_b, tokens_c,
                                  self.max_seq_length - 4)

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

        return input_ids, input_mask, segment_ids

    def score_batch(self, triples):
        """
        Args:
            triples: list of (head_ent, rel, tail_ent) string tuples
        Returns:
            dict: {"h\tr\tt": confidence} for each triple
        """
        results = {}

        with torch.no_grad():
            for i in tqdm(range(0, len(triples), self.batch_size),
                          desc="KG-BERT scoring"):
                batch = triples[i:i + self.batch_size]

                batch_input_ids = []
                batch_input_mask = []
                batch_segment_ids = []
                batch_keys = []

                for head_ent, rel_ent, tail_ent in batch:
                    input_ids, input_mask, segment_ids = \
                        self._triple_to_features(head_ent, rel_ent, tail_ent)
                    batch_input_ids.append(input_ids)
                    batch_input_mask.append(input_mask)
                    batch_segment_ids.append(segment_ids)
                    batch_keys.append(f"{head_ent}\t{rel_ent}\t{tail_ent}")

                input_ids_t = torch.LongTensor(batch_input_ids).to(self.device)
                input_mask_t = torch.LongTensor(batch_input_mask).to(self.device)
                segment_ids_t = torch.LongTensor(batch_segment_ids).to(self.device)

                logits = self.model(input_ids_t, segment_ids_t, input_mask_t)
                probs = torch.softmax(logits, dim=-1)

                for j, key in enumerate(batch_keys):
                    confidence = probs[j, 1].item()
                    results[key] = round(confidence, 6)

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Score auxiliary triplets using KG-BERT discriminator")
    parser.add_argument("--triplets", required=True,
                        help="Path to discovered_triplets.txt")
    parser.add_argument("--kgbert_model_dir", required=True,
                        help="KG-BERT model directory (with pytorch_model.bin)")
    parser.add_argument("--kgbert_data_dir", required=True,
                        help="KG-BERT data directory (with entity2text.txt, etc.)")
    parser.add_argument("--output", required=True,
                        help="Output JSON path for confidence map")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for KG-BERT inference")
    parser.add_argument("--max_seq_length", type=int, default=128,
                        help="Max sequence length for BERT input")
    parser.add_argument("--device", default=None,
                        help="Device (default: auto-detect)")
    args = parser.parse_args()

    triples = []
    with open(args.triplets, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                triples.append((parts[0], parts[1], parts[2]))

    print(f"Loaded {len(triples)} triplets from {args.triplets}")

    scorer = KGBERTScorer(
        model_dir=args.kgbert_model_dir,
        data_dir=args.kgbert_data_dir,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        do_lower_case=True,
        device=args.device,
    )

    confidence_map = scorer.score_batch(triples)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(confidence_map, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(confidence_map)} confidence entries to {args.output}")

    if confidence_map:
        vals = list(confidence_map.values())
        vals_arr = np.array(vals)
        print(f"Confidence stats: min={vals_arr.min():.4f}, "
              f"max={vals_arr.max():.4f}, "
              f"mean={vals_arr.mean():.4f}, "
              f"std={vals_arr.std():.4f}, "
              f"median={np.median(vals_arr):.4f}")

        bins = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
                (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
        print("Distribution:")
        for lo, hi in bins:
            cnt = sum(1 for v in vals if lo <= v < hi)
            print(f"  [{lo:.1f}, {hi:.1f}): {cnt:>6} ({cnt/len(vals)*100:.1f}%)")
        cnt = sum(1 for v in vals if v == 1.0)
        print(f"  [1.0]:       {cnt:>6}")


if __name__ == "__main__":
    main()
