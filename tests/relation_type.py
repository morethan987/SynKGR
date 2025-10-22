import argparse
import os
import sys


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Knowledge Graph Model Tester')

    # 基本参数
    parser.add_argument('--name', default='test_run', help='Test run name')
    parser.add_argument('--dataset', default='FB15K-237N', help='Dataset name')
    parser.add_argument('--model', default='compgcn', help='Model name')
    parser.add_argument('--score_func', default='conve', help='Score function')
    parser.add_argument('--model_path', required=True, help='Path to trained model')
    parser.add_argument('--save_dir', required=True, help='Directory with entity/relation mappings')
    parser.add_argument('--root_dir', required=True, help='The root directory of this project.')

    # 设备参数
    parser.add_argument('--gpu', type=int, default=-1, help='GPU id (-1 for CPU)')
    parser.add_argument('--npu', type=int, default=-1, help='NPU id (-1 for CPU)')
    parser.add_argument('--prefer_npu', action='store_true', help='Prefer NPU over GPU')

    # 模型参数
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers')
    parser.add_argument('--opn', default='corr', help='Composition operation')
    parser.add_argument('--adapt_aggr', dest='adapt_aggr', default=-1, type=int, help='use adaptive message aggregator or not')

    # 模型架构参数
    parser.add_argument('--num_bases', type=int, default=-1, help='Number of basis')
    parser.add_argument('--init_dim', type=int, default=100, help='Initial dimension')
    parser.add_argument('--gcn_dim', type=int, default=200, help='GCN dimension')
    parser.add_argument('--embed_dim', type=int, default=None, help='Embedding dimension')
    parser.add_argument('--gcn_layer', type=int, default=1, help='Number of GCN layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='GCN dropout')
    parser.add_argument('--hid_drop', type=float, default=0.3, help='Hidden dropout')
    parser.add_argument('--bias', action='store_true', help='Use bias')

    # ConvE参数
    parser.add_argument('--hid_drop2', type=float, default=0.3, help='ConvE hidden dropout')
    parser.add_argument('--feat_drop', type=float, default=0.3, help='ConvE feature dropout')
    parser.add_argument('--k_w', type=int, default=10, help='ConvE k_w')
    parser.add_argument('--k_h', type=int, default=20, help='ConvE k_h')
    parser.add_argument('--num_filt', type=int, default=200, help='ConvE number of filters')
    parser.add_argument('--ker_sz', type=int, default=7, help='ConvE kernel size')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    sys.path.append(os.path.join(args.root_dir, "loss_restraint_KGE_model"))
    from tester import Tester

    # 创建测试器
    tester = Tester(args)

    # 在测试集上评估
    test_results = tester.evaluate(split='test')
