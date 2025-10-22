import torch
import numpy as np
import argparse
import os
from collections import defaultdict as ddict
from ordered_set import OrderedSet
from torch.utils.data import DataLoader

from helper import *
from data_loader import *
from models import *


class Tester(object):
    def __init__(self, params):
        """
        测试器类的构造函数

        Parameters
        ----------
        params: 包含模型超参数的对象

        Returns
        -------
        初始化测试器,加载模型和数据
        """
        self.p = params
        self.logger = get_logger(self.p.name + '_test')

        self.logger.info("="*50)
        self.logger.info("Initializing Tester")
        self.logger.info("="*50)
        self.logger.info(vars(self.p))

        # 设置设备
        self.device, self.device_type = setup_device(
            gpu_id=self.p.gpu,
            npu_id=self.p.npu,
            prefer_npu=getattr(self.p, 'prefer_npu', False)
        )
        self.p.device_type = self.device_type
        self.logger.info(f"Using device: {self.device}, device_type: {self.device_type}")

        # 加载数据
        self.load_data()

        # 初始化模型
        self.model = self.add_model(self.p.model, self.p.score_func)

        # 加载训练好的模型
        self.load_model(self.p.model_path)

        self.logger.info("Tester initialized successfully!")

    def load_data(self):
        """
        加载数据集并构建必要的映射
        """
        self.logger.info("Loading data...")

        ent_set, rel_set = OrderedSet(), OrderedSet()
        for split in ['train', 'test', 'valid']:
            for line in open(f'data/{self.p.dataset}/{split}.txt'):
                sub, rel, obj = map(str.lower, line.strip().split('\t'))
                ent_set.add(sub)
                rel_set.add(rel)
                ent_set.add(obj)

        # 加载保存的映射(如果存在)
        entity2id_path = f'{self.p.save_dir}/entity2id.txt'
        relation2id_path = f'{self.p.save_dir}/relation2id.txt'

        if os.path.exists(entity2id_path) and os.path.exists(relation2id_path):
            self.logger.info("Loading entity and relation mappings from files...")
            self.ent2id = {}
            with open(entity2id_path, 'r') as f:
                for line in f:
                    ent, idx = line.strip().split('\t')
                    self.ent2id[ent] = int(idx)

            self.rel2id = {}
            with open(relation2id_path, 'r') as f:
                for line in f:
                    rel, idx = line.strip().split('\t')
                    self.rel2id[rel] = int(idx)
        else:
            self.logger.info("Creating new entity and relation mappings...")
            self.ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
            self.rel2id = {rel: idx for idx, rel in enumerate(rel_set)}
            self.rel2id.update({rel+'_reverse': idx+len(self.rel2id)
                               for idx, rel in enumerate(rel_set)})

        self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}
        self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}

        self.p.num_ent = len(self.ent2id)
        self.p.num_rel = len(self.rel2id) // 2
        self.p.embed_dim = self.p.k_w * self.p.k_h if self.p.embed_dim is None else self.p.embed_dim

        # 加载测试数据
        self.data = ddict(list)
        sr2o = ddict(set)

        for split in ['train', 'test', 'valid']:
            for line in open(f'data/{self.p.dataset}/{split}.txt'):
                sub, rel, obj = map(str.lower, line.strip().split('\t'))
                sub, rel, obj = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]
                self.data[split].append((sub, rel, obj))

                if split == 'train':
                    sr2o[(sub, rel)].add(obj)
                    sr2o[(obj, rel+self.p.num_rel)].add(sub)

        for split in ['test', 'valid']:
            for sub, rel, obj in self.data[split]:
                sr2o[(sub, rel)].add(obj)
                sr2o[(obj, rel+self.p.num_rel)].add(sub)

        self.sr2o_all = {k: list(v) for k, v in sr2o.items()}
        self.data = dict(self.data)

        # 构建测试数据
        self.triples = ddict(list)
        for split in ['test', 'valid']:
            for sub, rel, obj in self.data[split]:
                rel_inv = rel + self.p.num_rel
                self.triples[f'{split}_tail'].append({
                    'triple': (sub, rel, obj),
                    'label': self.sr2o_all[(sub, rel)]
                })
                self.triples[f'{split}_head'].append({
                    'triple': (obj, rel_inv, sub),
                    'label': self.sr2o_all[(obj, rel_inv)]
                })

        self.triples = dict(self.triples)

        # 构建数据加载器
        def get_data_loader(dataset_class, split, batch_size, shuffle=False):
            num_workers = max(0, self.p.num_workers)
            return DataLoader(
                dataset_class(self.triples[split], self.p),
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                collate_fn=dataset_class.collate_fn
            )

        self.data_iter = {
            'valid_head': get_data_loader(TestDataset, 'valid_head', self.p.batch_size),
            'valid_tail': get_data_loader(TestDataset, 'valid_tail', self.p.batch_size),
            'test_head': get_data_loader(TestDataset, 'test_head', self.p.batch_size),
            'test_tail': get_data_loader(TestDataset, 'test_tail', self.p.batch_size),
        }

        # 构建邻接矩阵
        self.edge_index, self.edge_type = self.construct_adj()

        self.logger.info(f"Data loaded: {self.p.num_ent} entities, {self.p.num_rel} relations")

    def construct_adj(self):
        """构建图的邻接矩阵"""
        edge_index, edge_type = [], []

        for sub, rel, obj in self.data['train']:
            edge_index.append((sub, obj))
            edge_type.append(rel)

        for sub, rel, obj in self.data['train']:
            edge_index.append((obj, sub))
            edge_type.append(rel + self.p.num_rel)

        edge_index = torch.LongTensor(edge_index).to(self.device).t()
        edge_type = torch.LongTensor(edge_type).to(self.device)

        return edge_index, edge_type

    def add_model(self, model, score_func):
        """创建模型"""
        model_name = f'{model}_{score_func}'

        if model_name.lower() == 'compgcn_transe':
            model = CompGCN_TransE(self.edge_index, self.edge_type, params=self.p)
        elif model_name.lower() == 'compgcn_distmult':
            model = CompGCN_DistMult(self.edge_index, self.edge_type, params=self.p)
        elif model_name.lower() == 'compgcn_conve':
            model = CompGCN_ConvE(self.edge_index, self.edge_type, params=self.p)
        else:
            raise NotImplementedError(f"Model {model_name} not implemented")

        model.to(self.device)
        return model

    def load_model(self, load_path):
        """加载训练好的模型"""
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model file not found: {load_path}")

        self.logger.info(f"Loading model from {load_path}")
        state = torch.load(load_path, map_location=self.device)
        vars(self.p).update(state['args'])
        self.model = self.add_model(self.p.model, self.p.score_func)
        self.model.load_state_dict(state['state_dict'])

        if 'best_val' in state:
            self.logger.info(f"Model's best validation MRR: {state['best_val'].get('mrr', 'N/A')}")

        self.logger.info("Model loaded successfully!")

    def read_batch(self, batch):
        """读取批次数据"""
        triple, label = [_.to(self.device) for _ in batch]
        return triple[:, 0], triple[:, 1], triple[:, 2], label

    def predict(self, split='test', mode='tail_batch'):
        """
        对指定数据集进行预测

        Parameters
        ----------
        split: 'test' 或 'valid'
        mode: 'tail_batch' 或 'head_batch'

        Returns
        -------
        results: 包含各种评估指标的字典
        """
        self.model.eval()

        with torch.no_grad():
            results = {}
            data_iter = iter(self.data_iter[f'{split}_{mode.split("_")[0]}'])

            for step, batch in enumerate(data_iter):
                sub, rel, obj, label = self.read_batch(batch)
                pred = self.model.forward(sub, rel)

                b_range = torch.arange(pred.size()[0], device=self.device)
                target_pred = pred[b_range, obj]

                # 过滤已知的正样本
                pred = torch.where(label.byte(), -torch.ones_like(pred) * 10000000, pred)
                pred[b_range, obj] = target_pred

                # 计算排名
                ranks = 1 + torch.argsort(
                    torch.argsort(pred, dim=1, descending=True),
                    dim=1,
                    descending=False
                )[b_range, obj]
                ranks = ranks.float()

                # 累积统计
                results['count'] = torch.numel(ranks) + results.get('count', 0.0)
                results['mr'] = torch.sum(ranks).item() + results.get('mr', 0.0)
                results['mrr'] = torch.sum(1.0/ranks).item() + results.get('mrr', 0.0)

                for k in range(10):
                    results[f'hits@{k+1}'] = torch.numel(
                        ranks[ranks <= (k+1)]
                    ) + results.get(f'hits@{k+1}', 0.0)

                if step % 100 == 0:
                    self.logger.info(f'[{split.title()}, {mode.title()} Step {step}]')

        return results

    def evaluate(self, split='test'):
        """
        完整评估指定数据集

        Parameters
        ----------
        split: 'test' 或 'valid'

        Returns
        -------
        results: 包含所有评估指标的字典
        """
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"Evaluating on {split} set")
        self.logger.info(f"{'='*50}")

        left_results = self.predict(split=split, mode='tail_batch')
        right_results = self.predict(split=split, mode='head_batch')
        results = get_combined_results(left_results, right_results)

        self.logger.info(f"\n{split.upper()} Results:")
        self.logger.info(f"MRR: Tail={results['left_mrr']:.5f}, Head={results['right_mrr']:.5f}, Avg={results['mrr']:.5f}")
        self.logger.info(f"MR:  Tail={results['left_mr']:.5f}, Head={results['right_mr']:.5f}, Avg={results['mr']:.5f}")
        self.logger.info(f"Hits@1:  Tail={results['left_hits@1']:.5f}, Head={results['right_hits@1']:.5f}, Avg={results['hits@1']:.5f}")
        self.logger.info(f"Hits@3:  Tail={results['left_hits@3']:.5f}, Head={results['right_hits@3']:.5f}, Avg={results['hits@3']:.5f}")
        self.logger.info(f"Hits@10: Tail={results['left_hits@10']:.5f}, Head={results['right_hits@10']:.5f}, Avg={results['hits@10']:.5f}")

        return results

    def predict_triple(self, head, relation, tail=None, top_k=10):
        """
        预测单个三元组

        Parameters
        ----------
        head: 头实体名称
        relation: 关系名称
        tail: 尾实体名称(可选,用于验证)
        top_k: 返回前k个预测结果

        Returns
        -------
        predictions: 预测结果列表
        """
        self.model.eval()

        if head not in self.ent2id:
            self.logger.error(f"Entity '{head}' not found in dataset")
            return None
        if relation not in self.rel2id:
            self.logger.error(f"Relation '{relation}' not found in dataset")
            return None

        head_id = self.ent2id[head]
        rel_id = self.rel2id[relation]

        with torch.no_grad():
            sub = torch.LongTensor([head_id]).to(self.device)
            rel = torch.LongTensor([rel_id]).to(self.device)

            pred = self.model.forward(sub, rel)
            pred = pred.squeeze()

            # 获取top-k预测
            top_k_scores, top_k_indices = torch.topk(pred, k=min(top_k, len(pred)))

            predictions = []
            for i, (score, idx) in enumerate(zip(top_k_scores, top_k_indices)):
                tail_entity = self.id2ent[idx.item()]
                predictions.append({
                    'rank': i + 1,
                    'entity': tail_entity,
                    'score': score.item()
                })

            # 如果提供了尾实体,计算其排名
            if tail is not None and tail in self.ent2id:
                tail_id = self.ent2id[tail]
                tail_score = pred[tail_id].item()
                tail_rank = (pred > tail_score).sum().item() + 1

                self.logger.info(f"\nQuery: ({head}, {relation}, ?)")
                self.logger.info(f"Ground truth '{tail}' rank: {tail_rank}, score: {tail_score:.4f}")

            self.logger.info(f"\nTop-{top_k} predictions:")
            for p in predictions:
                self.logger.info(f"  {p['rank']}. {p['entity']} (score: {p['score']:.4f})")

            return predictions

    def batch_predict_triples(self, triples, top_k=10):
        """
        批量预测多个三元组

        Parameters
        ----------
        triples: 三元组列表 [(head, relation, tail), ...]
        top_k: 返回前k个预测结果

        Returns
        -------
        all_predictions: 所有预测结果
        """
        all_predictions = []

        for head, relation, tail in triples:
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Predicting: ({head}, {relation}, {tail})")
            predictions = self.predict_triple(head, relation, tail, top_k)
            all_predictions.append({
                'query': (head, relation, tail),
                'predictions': predictions
            })

        return all_predictions

    def get_entity_embedding(self, entity_name):
        """获取实体的嵌入向量"""
        if entity_name not in self.ent2id:
            self.logger.error(f"Entity '{entity_name}' not found")
            return None

        entity_id = self.ent2id[entity_name]
        # 需要先通过模型前向传播获取嵌入
        self.model.eval()
        with torch.no_grad():
            # 调用模型内部方法获取实体嵌入
            if hasattr(self.model, 'init_embed'):
                embedding = self.model.init_embed[entity_id]
            else:
                self.logger.warning("Model does not have init_embed attribute")
                return None

        return embedding.cpu().numpy()

    def find_similar_entities(self, entity_name, top_k=10):
        """查找与给定实体最相似的实体"""
        if entity_name not in self.ent2id:
            self.logger.error(f"Entity '{entity_name}' not found")
            return None

        self.model.eval()
        with torch.no_grad():
            if not hasattr(self.model, 'init_embed'):
                self.logger.warning("Model does not have init_embed attribute")
                return None

            entity_id = self.ent2id[entity_name]
            query_embed = self.model.init_embed[entity_id]
            all_embeds = self.model.init_embed

            # 计算余弦相似度
            similarities = torch.nn.functional.cosine_similarity(
                query_embed.unsqueeze(0), all_embeds
            )

            # 获取top-k(排除自身)
            top_k_scores, top_k_indices = torch.topk(similarities, k=min(top_k+1, len(similarities)))

            similar_entities = []
            for score, idx in zip(top_k_scores, top_k_indices):
                if idx.item() != entity_id:
                    similar_entities.append({
                        'entity': self.id2ent[idx.item()],
                        'similarity': score.item()
                    })
                    if len(similar_entities) >= top_k:
                        break

            self.logger.info(f"\nTop-{top_k} entities similar to '{entity_name}':")
            for i, item in enumerate(similar_entities, 1):
                self.logger.info(f"  {i}. {item['entity']} (similarity: {item['similarity']:.4f})")

            return similar_entities
