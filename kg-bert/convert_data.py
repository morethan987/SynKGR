"""
Convert datasets from data/ to kg-bert expected format in kg-bert/data/.

Supports: FB15k-237N, CoDEx-S

Source layout (data/<DATASET>/):
  train.txt, test.txt, valid.txt : head\\trelation\\ttail
  entity2name.txt                : entity\\tname
  entity2des.txt                 : entity\\tdescription
  entity2id.txt                  : entity\\tindex
  relation2id.txt                : relation\\tindex
  relation2name.txt              : relation\\tname (CoDEx-S only)

Additional source (LLM_Discriminator/data/):
  <DATASET>-train.json           : positive + negative triples with labels
  <DATASET>-valid.json           : positive + negative triples with labels
  <DATASET>-test.json            : positive + negative triples with labels
  Format: [{"embedding_ids": [h, r, t], "output": "True|False"}, ...]

Target layout (kg-bert/data/<DATASET>/):
  train.tsv                      : head\\trelation\\ttail  (positive only, train-time neg sampling)
  dev.tsv                        : head\\trelation\\ttail\\tlabel  (from LLM_Discriminator)
  test.tsv                       : head\\trelation\\ttail\\tlabel  (from LLM_Discriminator)
  entities.txt                   : one entity per line
  relations.txt                  : one relation per line
  entity2text.txt                : entity\\tname
  entity2textlong.txt            : entity\\tdescription
  relation2text.txt              : relation\\tnatural_text
"""

import json
import os

DATASETS = ["FB15k-237N", "CoDEx-S"]
BASE_SRC = "data"
BASE_LLM = "LLM_Discriminator/data"
BASE_DST = "kg-bert/data"

DATASET_NAME_MAP = {
    "FB15k-237N": "FB15K-237N",
    "CoDEx-S": "CoDeX-S",
}


def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def write_lines(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")


def read_tsv_dict(path):
    result = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                result[parts[0]] = parts[1]
    return result


def read_id_map(path):
    idx_to_id = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                idx_to_id[int(parts[1])] = parts[0]
    return idx_to_id


def relation_to_text(rel):
    text = rel.replace("/", " ").replace("_", " ").strip()
    return " ".join(text.split())


def convert_llm_split(llm_path, ent_idx2id, rel_idx2id, kg_ents):
    triples = []
    skipped = 0
    with open(llm_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        h_idx, r_idx, t_idx = item["embedding_ids"]
        h_eid = ent_idx2id.get(h_idx)
        r_eid = rel_idx2id.get(r_idx)
        t_eid = ent_idx2id.get(t_idx)
        if h_eid is None or r_eid is None or t_eid is None:
            skipped += 1
            continue
        if h_eid not in kg_ents or t_eid not in kg_ents:
            skipped += 1
            continue
        label = "1" if item["output"] == "True" else "0"
        triples.append(f"{h_eid}\t{r_eid}\t{t_eid}\t{label}")
    return triples, skipped


def convert_dataset(name):
    src_dir = os.path.join(BASE_SRC, name)
    dst_dir = os.path.join(BASE_DST, name)
    os.makedirs(dst_dir, exist_ok=True)
    llm_name = DATASET_NAME_MAP[name]

    print(f"\n{'='*50}")
    print(f"Converting: {name}")
    print(f"{'='*50}")

    # --- train.tsv (positive triples only, from original data) ---
    train_lines = read_lines(os.path.join(src_dir, "train.txt"))
    write_lines(os.path.join(dst_dir, "train.tsv"), train_lines)
    print(f"  train.tsv: {len(train_lines)} triples (positive only)")

    # --- entities ---
    kg_ents = set()
    for fname in ("train.txt", "test.txt", "valid.txt"):
        for line in read_lines(os.path.join(src_dir, fname)):
            parts = line.split("\t")
            if len(parts) >= 3:
                kg_ents.add(parts[0])
                kg_ents.add(parts[2])
    sorted_ents = sorted(kg_ents)
    write_lines(os.path.join(dst_dir, "entities.txt"), sorted_ents)
    print(f"  entities.txt: {len(kg_ents)} entities")

    # --- relations ---
    relations = set()
    for fname in ("train.txt", "test.txt", "valid.txt"):
        for line in read_lines(os.path.join(src_dir, fname)):
            parts = line.split("\t")
            if len(parts) >= 3:
                relations.add(parts[1])
    sorted_rels = sorted(relations)
    write_lines(os.path.join(dst_dir, "relations.txt"), sorted_rels)
    print(f"  relations.txt: {len(relations)} relations")

    # --- entity2text.txt (short: entity name) ---
    ent2name = read_tsv_dict(os.path.join(src_dir, "entity2name.txt"))
    entity2text_lines = []
    missing_name = 0
    for ent in sorted_ents:
        name = ent2name.get(ent, ent).replace("_", " ")
        entity2text_lines.append(f"{ent}\t{name}")
        if ent not in ent2name:
            missing_name += 1
    write_lines(os.path.join(dst_dir, "entity2text.txt"), entity2text_lines)
    print(f"  entity2text.txt: {len(entity2text_lines)} entries ({missing_name} missing name)")

    # --- entity2textlong.txt (long: entity description) ---
    ent2des = read_tsv_dict(os.path.join(src_dir, "entity2des.txt"))
    entity2textlong_lines = []
    missing_des = 0
    for ent in sorted_ents:
        des = ent2des.get(ent, ent2name.get(ent, ent)).replace("_", " ")
        entity2textlong_lines.append(f"{ent}\t{des}")
        if ent not in ent2des:
            missing_des += 1
    write_lines(os.path.join(dst_dir, "entity2textlong.txt"), entity2textlong_lines)
    print(f"  entity2textlong.txt: {len(entity2textlong_lines)} entries ({missing_des} missing description)")

    # --- relation2text.txt ---
    rel2name_path = os.path.join(src_dir, "relation2name.txt")
    rel2name = read_tsv_dict(rel2name_path) if os.path.exists(rel2name_path) else {}
    relation2text_lines = []
    missing_rel_name = 0
    for rel in sorted_rels:
        text = rel2name.get(rel, relation_to_text(rel))
        relation2text_lines.append(f"{rel}\t{text}")
        if rel not in rel2name:
            missing_rel_name += 1
    write_lines(os.path.join(dst_dir, "relation2text.txt"), relation2text_lines)
    print(f"  relation2text.txt: {len(relation2text_lines)} entries ({missing_rel_name} using heuristic)")

    # --- dev.tsv & test.tsv (from LLM_Discriminator, with positive/negative labels) ---
    ent_idx2id = read_id_map(os.path.join(src_dir, "entity2id.txt"))
    rel_idx2id = read_id_map(os.path.join(src_dir, "relation2id.txt"))

    for llm_split, dst_name in [("valid", "dev.tsv"), ("test", "test.tsv")]:
        llm_path = os.path.join(BASE_LLM, f"{llm_name}-{llm_split}.json")
        if not os.path.exists(llm_path):
            print(f"  {dst_name}: SKIP (not found: {llm_path})")
            continue
        triples, skipped = convert_llm_split(llm_path, ent_idx2id, rel_idx2id, kg_ents)
        write_lines(os.path.join(dst_dir, dst_name), triples)

        n_pos = sum(1 for t in triples if t.endswith("\t1"))
        n_neg = sum(1 for t in triples if t.endswith("\t0"))
        print(f"  {dst_name}: {len(triples)} triples ({n_pos} positive, {n_neg} negative, {skipped} skipped)")


def main():
    for name in DATASETS:
        convert_dataset(name)
    print("\nAll done.")


if __name__ == "__main__":
    main()
