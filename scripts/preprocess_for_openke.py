#!/usr/bin/env python3
"""
将 data/FB15k-237N/ 中的原始数据转换为 OpenKE 所需格式，输出到 openke/FB15k-237N/。

原始数据格式:
  entity2id.txt  : entity_name<TAB>id (每行，无总数行)
  relation2id.txt: relation_name<TAB>id (每行，无总数行)
  train.txt      : head<TAB>relation<TAB>tail (文本名称)
  valid.txt      : head<TAB>relation<TAB>tail
  test.txt       : head<TAB>relation<TAB>tail

OpenKE 期望格式:
  entity2id.txt  : 第一行为实体总数，之后每行 entity_name<TAB>id
  relation2id.txt: 第一行为关系总数，之后每行 relation_name<TAB>id
  train2id.txt   : 第一行为三元组总数，之后每行 head_id<TAB>tail_id<TAB>rel_id
  valid2id.txt   : 同上
  test2id.txt    : 同上
"""

import os
import sys
import argparse


def build_name2id_map(filepath):
    name2id = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                name2id[parts[0]] = int(parts[1])
    return name2id


def convert_entity2id(src_path, dst_path):
    entries = []
    with open(src_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(line)
    with open(dst_path, "w", encoding="utf-8") as f:
        f.write(f"{len(entries)}\n")
        for entry in entries:
            f.write(f"{entry}\n")
    return len(entries)


def convert_relation2id(src_path, dst_path):
    return convert_entity2id(src_path, dst_path)


def convert_triples(src_path, dst_path, ent_map, rel_map):
    triples = []
    missing_ent = set()
    missing_rel = set()
    with open(src_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                print(f"  WARNING: {src_path}:{line_num} - 跳过格式异常行: {line[:80]}")
                continue
            h_name, r_name, t_name = parts
            if h_name not in ent_map:
                missing_ent.add(h_name)
                continue
            if t_name not in ent_map:
                missing_ent.add(t_name)
                continue
            if r_name not in rel_map:
                missing_rel.add(r_name)
                continue
            triples.append((ent_map[h_name], ent_map[t_name], rel_map[r_name]))

    if missing_ent:
        print(f"  WARNING: {len(missing_ent)} 个实体在 entity2id.txt 中未找到")
    if missing_rel:
        print(f"  WARNING: {len(missing_rel)} 个关系在 relation2id.txt 中未找到")

    with open(dst_path, "w", encoding="utf-8") as f:
        f.write(f"{len(triples)}\n")
        for h, t, r in triples:
            f.write(f"{h}\t{t}\t{r}\n")
    return len(triples)


def main():
    parser = argparse.ArgumentParser(description="将原始数据预处理为 OpenKE 格式")
    parser.add_argument("--src_dir", type=str, default="data/FB15k-237N",
                        help="原始数据目录")
    parser.add_argument("--dst_dir", type=str, default="openke/FB15k-237N",
                        help="OpenKE 格式输出目录")
    args = parser.parse_args()

    src_dir = args.src_dir.rstrip("/") + "/"
    dst_dir = args.dst_dir.rstrip("/") + "/"

    required_src_files = [
        "entity2id.txt", "relation2id.txt", "train.txt", "valid.txt", "test.txt"
    ]
    for fname in required_src_files:
        fpath = src_dir + fname
        if not os.path.isfile(fpath):
            print(f"错误: 找不到源文件 {fpath}")
            sys.exit(1)

    os.makedirs(dst_dir, exist_ok=True)

    print("=" * 50)
    print("OpenKE 数据预处理")
    print(f"  源目录: {src_dir}")
    print(f"  目标目录: {dst_dir}")
    print("=" * 50)

    print("\n[1/5] 读取 entity2id.txt 和 relation2id.txt 构建映射...")
    ent_map = build_name2id_map(src_dir + "entity2id.txt")
    rel_map = build_name2id_map(src_dir + "relation2id.txt")
    print(f"  实体数: {len(ent_map)}, 关系数: {len(rel_map)}")

    print("\n[2/5] 转换 entity2id.txt...")
    ent_total = convert_entity2id(src_dir + "entity2id.txt", dst_dir + "entity2id.txt")
    print(f"  完成, 共 {ent_total} 个实体")

    print("\n[3/5] 转换 relation2id.txt...")
    rel_total = convert_relation2id(src_dir + "relation2id.txt", dst_dir + "relation2id.txt")
    print(f"  完成, 共 {rel_total} 个关系")

    print("\n[4/5] 转换 train.txt -> train2id.txt...")
    train_total = convert_triples(src_dir + "train.txt", dst_dir + "train2id.txt", ent_map, rel_map)
    print(f"  完成, 共 {train_total} 个训练三元组")

    print("\n[5/5] 转换 valid.txt / test.txt...")
    valid_total = convert_triples(src_dir + "valid.txt", dst_dir + "valid2id.txt", ent_map, rel_map)
    print(f"  valid2id.txt: {valid_total} 个三元组")
    test_total = convert_triples(src_dir + "test.txt", dst_dir + "test2id.txt", ent_map, rel_map)
    print(f"  test2id.txt: {test_total} 个三元组")

    print("\n" + "=" * 50)
    print("预处理完成!")
    print(f"  输出目录: {dst_dir}")
    print(f"  文件: entity2id.txt, relation2id.txt, train2id.txt, valid2id.txt, test2id.txt")
    print("=" * 50)


if __name__ == "__main__":
    main()
