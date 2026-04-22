#!/usr/bin/env python3
"""
KGE模型测试脚本 - 从保存的checkpoint加载模型并运行 Link Prediction 评估。

使用方法:
    python -u LLM_Discriminator/test_kge.py --config LLM_Discriminator/train_kge_output/config_<timestamp>.json
    # 或手动指定参数:
    python -u LLM_Discriminator/test_kge.py \
        --in_path openke/FB15k-237N/ \
        --checkpoint LLM_Discriminator/train_kge_output/checkpoint_<timestamp>/rotate.ckpt \
        --dimension 512 \
        --margin 6.0 \
        --epsilon 2.0
"""

import os
import sys
import argparse
import json
import time
import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="KGE模型测试 (OpenKE Link Prediction)")

    parser.add_argument("--config", type=str, default=None,
                        help="训练时保存的 config JSON 文件路径; 若指定则自动加载所有超参数")
    parser.add_argument("--in_path", type=str, default=None,
                        help="OpenKE格式数据目录路径 (需以 / 结尾)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="模型 checkpoint 文件路径 (.ckpt)")
    parser.add_argument("--out_path", type=str, default=None,
                        help="结果输出目录 (默认取 checkpoint 所在目录的上级)")

    parser.add_argument("--dimension", type=int, default=None,
                        help="嵌入维度")
    parser.add_argument("--margin", type=float, default=None,
                        help="RotatE margin γ")
    parser.add_argument("--epsilon", type=float, default=2.0,
                        help="RotatE epsilon")

    parser.add_argument("--use_gpu", action="store_true", default=True,
                        help="是否使用 GPU")
    parser.add_argument("--no_gpu", action="store_true",
                        help="禁用 GPU")
    parser.add_argument("--type_constrain", action="store_true", default=False,
                        help="测试时是否使用类型约束")

    args = parser.parse_args()

    if args.no_gpu:
        args.use_gpu = False

    # 从 config 文件加载参数
    if args.config is not None:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        if args.in_path is None:
            args.in_path = cfg.get("in_path")
        if args.dimension is None:
            args.dimension = cfg.get("dimension", 512)
        if args.margin is None:
            args.margin = cfg.get("margin", 6.0)
        if args.out_path is None:
            args.out_path = cfg.get("out_path")

        args.time_stamp = cfg.get("time_stamp", "")
        args.epsilon = cfg.get("epsilon", 2.0)
        args.ent_tot = cfg.get("ent_tot")
        args.rel_tot = cfg.get("rel_tot")

        # 自动推断 checkpoint 路径
        if args.checkpoint is None:
            ts = args.time_stamp
            ckpt_dir = os.path.join(args.out_path, f"checkpoint_{ts}" if ts else "checkpoint")
            args.checkpoint = os.path.join(ckpt_dir, "rotate.ckpt")
    else:
        args.time_stamp = ""

    # 参数校验
    if args.in_path is None:
        parser.error("必须指定 --in_path 或通过 --config 加载")
    if args.checkpoint is None:
        parser.error("必须指定 --checkpoint 或通过 --config 加载")

    if not args.in_path.endswith("/"):
        args.in_path += "/"

    return args


def main():
    args = parse_args()

    print("=" * 60)
    print("KGE模型测试 - Link Prediction")
    print("=" * 60)
    print(f"数据路径:      {args.in_path}")
    print(f"Checkpoint:    {args.checkpoint}")
    print(f"嵌入维度:      {args.dimension}")
    print(f"Margin γ:      {args.margin}")
    print(f"Epsilon:       {args.epsilon}")
    print(f"使用GPU:       {args.use_gpu}")
    print(f"类型约束:      {args.type_constrain}")
    print("=" * 60)

    if not os.path.isfile(args.checkpoint):
        print(f"\n错误: checkpoint 文件不存在: {args.checkpoint}")
        sys.exit(1)

    required_files = ["entity2id.txt", "relation2id.txt"]
    for fname in required_files:
        fpath = os.path.join(args.in_path, fname)
        if not os.path.isfile(fpath):
            print(f"错误: 数据文件不存在: {fpath}")
            sys.exit(1)

    # 构建 RotatE 模型
    print("\n[1/3] 构建 RotatE 模型...")

    ent_tot = args.ent_tot
    rel_tot = args.rel_tot

    if ent_tot is None or rel_tot is None:
        print("  未从 config 获取 ent_tot/rel_tot，从数据文件推断...")
        with open(os.path.join(args.in_path, "entity2id.txt"), "r") as f:
            ent_tot = int(f.readline().strip())
        with open(os.path.join(args.in_path, "relation2id.txt"), "r") as f:
            rel_tot = int(f.readline().strip())

    print(f"  实体总数: {ent_tot}")
    print(f"  关系总数: {rel_tot}")

    rotate = RotatE(
        ent_tot=ent_tot,
        rel_tot=rel_tot,
        dim=args.dimension,
        margin=args.margin,
        epsilon=args.epsilon,
    )

    # 加载 checkpoint
    print("\n[2/3] 加载 checkpoint...")
    rotate.load_checkpoint(args.checkpoint)
    print(f"  已加载: {args.checkpoint}")

    # 运行测试
    print("\n[3/3] 运行 Link Prediction 测试...")
    start_time = time.time()

    test_dataloader = TestDataLoader(args.in_path, "link", type_constrain=args.type_constrain)
    tester = Tester(
        model=rotate,
        data_loader=test_dataloader,
        use_gpu=args.use_gpu,
    )
    mrr, mr, hit10, hit3, hit1 = tester.run_link_prediction(type_constrain=args.type_constrain)

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("Link Prediction 结果:")
    print(f"  MRR:    {mrr:.4f}")
    print(f"  MR:     {mr:.4f}")
    print(f"  Hit@1:  {hit1:.4f}")
    print(f"  Hit@3:  {hit3:.4f}")
    print(f"  Hit@10: {hit10:.4f}")
    print(f"  耗时:   {elapsed:.1f}s ({datetime.timedelta(seconds=int(elapsed))})")
    print("=" * 60)

    # 保存结果
    out_path = args.out_path
    if out_path is None:
        out_path = os.path.dirname(os.path.dirname(args.checkpoint))

    results = {
        "MRR": float(mrr),
        "MR": float(mr),
        "Hit@1": float(hit1),
        "Hit@3": float(hit3),
        "Hit@10": float(hit10),
        "checkpoint": args.checkpoint,
        "type_constrain": args.type_constrain,
        "elapsed_seconds": elapsed,
    }

    results_path = os.path.join(
        out_path,
        f"results_{args.time_stamp}.json" if args.time_stamp else "results.json",
    )
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n测试结果已保存: {results_path}")


if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    from openke.config import Tester
    from openke.module.model import RotatE
    from openke.data import TestDataLoader

    main()
