#!/usr/bin/env python3
"""
KGE模型训练主脚本 (RotatE + adversarial negative sampling)
基于 OpenKE 官方样例 train_rotate_WN18RR_adv.py 改写，支持 argparse 命令行参数。

使用方法:
    python LLM_Discriminator/train_kge.py \
        --model RotatE \
        --in_path openke/FB15k-237N/ \
        --out_path LLM_Discriminator/train_kge_output/ \
        --dimension 512 \
        --batch_size 2048 \
        --learning_rate 2e-5 \
        --margin 6.0 \
        --optimizer Adam \
        --neg_ent 32 \
        --epochs 3000
"""

import os
import sys
import argparse
import json
import time
import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="KGE模型训练 (OpenKE)")

    parser.add_argument("--model", type=str, default="RotatE",
                        help="模型名称 (目前仅支持 RotatE)")
    parser.add_argument("--in_path", type=str, required=True,
                        help="OpenKE格式数据目录路径 (需以 / 结尾)")
    parser.add_argument("--out_path", type=str, required=True,
                        help="模型输出目录")
    parser.add_argument("--time_stamp", type=str, default="",
                        help="时间戳标识，用于区分不同训练运行")

    parser.add_argument("--dimension", type=int, default=512,
                        help="嵌入维度 (RotatE: dim_e = dim*2, dim_r = dim)")
    parser.add_argument("--batch_size", type=int, default=2048,
                        help="训练批次大小")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="学习率 (alpha)")
    parser.add_argument("--margin", type=float, default=6.0,
                        help="RotatE margin γ")
    parser.add_argument("--epsilon", type=float, default=2.0,
                        help="RotatE epsilon (用于嵌入范围计算)")
    parser.add_argument("--optimizer", type=str, default="Adam",
                        help="优化器 (Adam / SGD / Adagrad / Adadelta)")
    parser.add_argument("--neg_ent", type=int, default=32,
                        help="负采样实体数 K")
    parser.add_argument("--epochs", type=int, default=3000,
                        help="训练轮数")
    parser.add_argument("--threads", type=int, default=8,
                        help="数据加载线程数")

    parser.add_argument("--adv_temperature", type=float, default=2.0,
                        help="对抗负采样温度参数")
    parser.add_argument("--regul_rate", type=float, default=0.0,
                        help="正则化率")

    parser.add_argument("--test_triple", action="store_true",
                        help="训练完成后是否运行 link prediction 测试")
    parser.add_argument("--save_steps", type=int, default=None,
                        help="每隔多少轮保存一次 checkpoint")
    parser.add_argument("--use_gpu", action="store_true", default=True,
                        help="是否使用 GPU")
    parser.add_argument("--no_gpu", action="store_true",
                        help="禁用 GPU")

    args = parser.parse_args()

    if args.no_gpu:
        args.use_gpu = False

    if not args.in_path.endswith("/"):
        args.in_path += "/"

    return args


def validate_data_path(in_path):
    required_files = ["train2id.txt", "entity2id.txt", "relation2id.txt"]
    for fname in required_files:
        fpath = os.path.join(in_path, fname)
        if not os.path.isfile(fpath):
            print(f"错误: 数据文件不存在: {fpath}")
            print("请先运行预处理: python scripts/preprocess_for_openke.py")
            sys.exit(1)


def main():
    args = parse_args()

    print("=" * 60)
    print("KGE模型训练 - OpenKE")
    print("=" * 60)
    print(f"模型:          {args.model}")
    print(f"数据路径:      {args.in_path}")
    print(f"输出路径:      {args.out_path}")
    print(f"时间戳:        {args.time_stamp}")
    print(f"嵌入维度:      {args.dimension}")
    print(f"批次大小:      {args.batch_size}")
    print(f"学习率:        {args.learning_rate}")
    print(f"Margin γ:      {args.margin}")
    print(f"Epsilon:       {args.epsilon}")
    print(f"优化器:        {args.optimizer}")
    print(f"负采样数 K:    {args.neg_ent}")
    print(f"训练轮数:      {args.epochs}")
    print(f"对抗温度:      {args.adv_temperature}")
    print(f"正则化率:      {args.regul_rate}")
    print(f"使用GPU:       {args.use_gpu}")
    print(f"测试:          {args.test_triple}")
    print("=" * 60)

    validate_data_path(args.in_path)

    os.makedirs(args.out_path, exist_ok=True)

    checkpoint_dir = os.path.join(args.out_path, f"checkpoint_{args.time_stamp}" if args.time_stamp else "checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)

    config_path = os.path.join(args.out_path, f"config_{args.time_stamp}.json" if args.time_stamp else "config.json")

    print("\n[1/4] 初始化数据加载器...")
    train_dataloader = TrainDataLoader(
        in_path=args.in_path,
        batch_size=args.batch_size,
        threads=args.threads,
        sampling_mode="cross",
        bern_flag=False,
        filter_flag=True,
        neg_ent=args.neg_ent,
        neg_rel=0,
    )

    ent_tot = train_dataloader.get_ent_tot()
    rel_tot = train_dataloader.get_rel_tot()
    tri_tot = train_dataloader.get_triple_tot()
    print(f"  实体总数: {ent_tot}")
    print(f"  关系总数: {rel_tot}")
    print(f"  训练三元组数: {tri_tot}")

    print("\n[2/4] 构建 RotatE 模型...")
    rotate = RotatE(
        ent_tot=ent_tot,
        rel_tot=rel_tot,
        dim=args.dimension,
        margin=args.margin,
        epsilon=args.epsilon,
    )

    model = NegativeSampling(
        model=rotate,
        loss=SigmoidLoss(adv_temperature=args.adv_temperature),
        batch_size=train_dataloader.get_batch_size(),
        regul_rate=args.regul_rate,
    )

    print(f"  模型参数量: {sum(p.numel() for p in rotate.parameters()):,}")
    print(f"  实体嵌入维度: {args.dimension * 2} (复数: {args.dimension} re + {args.dimension} im)")
    print(f"  关系嵌入维度: {args.dimension}")

    config = {
        "model": args.model,
        "in_path": args.in_path,
        "out_path": args.out_path,
        "time_stamp": args.time_stamp,
        "dimension": args.dimension,
        "dim_e": args.dimension * 2,
        "dim_r": args.dimension,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "margin": args.margin,
        "epsilon": args.epsilon,
        "optimizer": args.optimizer,
        "neg_ent": args.neg_ent,
        "epochs": args.epochs,
        "adv_temperature": args.adv_temperature,
        "regul_rate": args.regul_rate,
        "ent_tot": ent_tot,
        "rel_tot": rel_tot,
        "train_triple_tot": tri_tot,
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"  配置已保存: {config_path}")

    print("\n[3/4] 开始训练...")
    start_time = time.time()

    trainer = Trainer(
        model=model,
        data_loader=train_dataloader,
        train_times=args.epochs,
        alpha=args.learning_rate,
        use_gpu=args.use_gpu,
        opt_method=args.optimizer,
        save_steps=args.save_steps,
        checkpoint_dir=os.path.join(checkpoint_dir, args.model.lower()),
    )
    trainer.run()

    elapsed = time.time() - start_time
    print(f"\n训练完成! 耗时: {elapsed:.1f}s ({datetime.timedelta(seconds=int(elapsed))})")

    ckpt_path = os.path.join(checkpoint_dir, f"{args.model.lower()}.ckpt")
    rotate.save_checkpoint(ckpt_path)
    print(f"模型已保存: {ckpt_path}")

    if args.test_triple:
        print("\n[4/4] 运行 Link Prediction 测试...")
        test_dataloader = TestDataLoader(args.in_path, "link")

        rotate.load_checkpoint(ckpt_path)
        tester = Tester(
            model=rotate,
            data_loader=test_dataloader,
            use_gpu=args.use_gpu,
        )
        mrr, mr, hit10, hit3, hit1 = tester.run_link_prediction(type_constrain=False)

        print("\n" + "=" * 60)
        print("Link Prediction 结果:")
        print(f"  MRR:   {mrr:.4f}")
        print(f"  MR:    {mr:.4f}")
        print(f"  Hit@1: {hit1:.4f}")
        print(f"  Hit@3: {hit3:.4f}")
        print(f"  Hit@10:{hit10:.4f}")
        print("=" * 60)

        results = {
            "MRR": float(mrr),
            "MR": float(mr),
            "Hit@1": float(hit1),
            "Hit@3": float(hit3),
            "Hit@10": float(hit10),
            "elapsed_seconds": elapsed,
        }
        results_path = os.path.join(
            args.out_path,
            f"results_{args.time_stamp}.json" if args.time_stamp else "results.json",
        )
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"测试结果已保存: {results_path}")
    else:
        print("\n[4/4] 跳过测试 (--test_triple 未指定)")

    print("\n全部完成!")


if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    import torch
    from openke.config import Trainer, Tester
    from openke.module.model import RotatE
    from openke.module.loss import SigmoidLoss
    from openke.module.strategy import NegativeSampling
    from openke.data import TrainDataLoader, TestDataLoader

    main()
