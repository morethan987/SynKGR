# -*- coding: utf-8 -*-
# 关系类型分类脚本
# 输入: train.txt, valid.txt, test.txt
# 输出: type_constrain.txt, 1-1.txt, 1-n.txt, n-1.txt, n-n.txt, test_all.txt
# 使用:
# cdko && acko && python data/n-n.py --data FB15k-237N
# cdko && acko && python data/n-n.py --data CoDEx-S

import argparse

def read_triples(file_path):
    """读取三元组文件，返回列表[(h, t, r), ...]"""
    triples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                print(f"warning: split {line} get wrong parts count")
                continue
            h, r, t = parts
            triples.append((h, t, r))
    return triples

def main(args):
    # 读取数据
    train_triples = read_triples(f"data/{args.dataset}/train.txt")
    valid_triples = read_triples(f"data/{args.dataset}/valid.txt")
    test_triples = read_triples(f"data/{args.dataset}/test.txt")

    all_triples = train_triples + valid_triples + test_triples

    lef = {}      # (h, r) -> [t1, t2, ...]
    rig = {}      # (r, t) -> [h1, h2, ...]
    rellef = {}   # r -> {h:1}
    relrig = {}   # r -> {t:1}

    # 统计结构
    for h, t, r in all_triples:
        lef.setdefault((h, r), []).append(t)
        rig.setdefault((r, t), []).append(h)
        rellef.setdefault(r, {})[h] = 1
        relrig.setdefault(r, {})[t] = 1

    # 输出 type_constrain.txt
    with open(f"data/{args.dataset}/type_constrain.txt", "w", encoding="utf-8") as f:
        for r in rellef:
            f.write(f"{r}\t{len(rellef[r])}")
            for h in rellef[r]:
                f.write(f"\t{h}")
            f.write("\n")
            f.write(f"{r}\t{len(relrig[r])}")
            for t in relrig[r]:
                f.write(f"\t{t}")
            f.write("\n")

    # 计算每个关系的平均连接度
    rellef_count = {}
    totlef = {}
    relrig_count = {}
    totrig = {}

    for (h, r), t_list in lef.items():
        rellef_count[r] = rellef_count.get(r, 0) + len(t_list)
        totlef[r] = totlef.get(r, 0) + 1.0

    for (r, t), h_list in rig.items():
        relrig_count[r] = relrig_count.get(r, 0) + len(h_list)
        totrig[r] = totrig.get(r, 0) + 1.0

    # 分类统计
    s11 = s1n = sn1 = snn = 0
    for h, t, r in test_triples:
        rign = rellef_count[r] / totlef[r]
        lefn = relrig_count[r] / totrig[r]
        if rign < 1.5 and lefn < 1.5:
            s11 += 1
        elif rign >= 1.5 and lefn < 1.5:
            s1n += 1
        elif rign < 1.5 and lefn >= 1.5:
            sn1 += 1
        else:
            snn += 1

    # 分类输出
    with open(f"data/{args.dataset}/1-1.txt", "w", encoding="utf-8") as f11, \
        open(f"data/{args.dataset}/1-n.txt", "w", encoding="utf-8") as f1n, \
        open(f"data/{args.dataset}/n-1.txt", "w", encoding="utf-8") as fn1, \
        open(f"data/{args.dataset}/n-n.txt", "w", encoding="utf-8") as fnn, \
        open(f"data/{args.dataset}/test_all.txt", "w", encoding="utf-8") as fall:

        for h, t, r in test_triples:
            rign = rellef_count[r] / totlef[r]
            lefn = relrig_count[r] / totrig[r]
            triple_str = f"{h}\t{r}\t{t}\n"
            if rign < 1.5 and lefn < 1.5:
                f11.write(triple_str)
                fall.write(f"0\t{h}\t{r}\t{t}\n")
            elif rign >= 1.5 and lefn < 1.5:
                f1n.write(triple_str)
                fall.write(f"1\t{h}\t{r}\t{t}\n")
            elif rign < 1.5 and lefn >= 1.5:
                fn1.write(triple_str)
                fall.write(f"2\t{h}\t{r}\t{t}\n")
            else:
                fnn.write(triple_str)
                fall.write(f"3\t{h}\t{r}\t{t}\n")

    print("✅ 分类完成：")
    print(f"1-1: {s11}, 1-N: {s1n}, N-1: {sn1}, N-N: {snn}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser For Arguments', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', dest='dataset', default='FB15k-237', help='Dataset to use, default: FB15k-237')
    args = parser.parse_args()

    main(args)
