"""
训练过程中采集抑制效果指标
用于可视化验证大损失约束和自适应消息聚合对低置信度辅助三元组的抑制效果
"""

import os
import json
import numpy as np
from collections import defaultdict


class MetricsCollector:
    """训练过程中采集抑制效果指标"""

    def __init__(self, aux_confidence_map, aux_triple_set, output_dir):
        """
        Args:
            aux_confidence_map: {(h_id, r_id, t_id): confidence} — 辅助三元组置信度映射（ID 级别）
            aux_triple_set: set of (h_id, r_id, t_id) — 辅助三元组 ID 集合
            output_dir: 数据输出目录
        """
        self.aux_confidence_map = aux_confidence_map
        self.aux_triple_set = aux_triple_set
        self.output_dir = output_dir

        # Loss Restraint: 全局累积统计
        self.accumulated_drop_count = {}
        self.accumulated_loss_sum = {}
        self.accumulated_count = {}
        self.epoch_loss_summary = []

        # Attention Alpha: epoch 级全量记录
        self.epoch_alpha_summary = []

        # 最后一轮 epoch 的原始 alpha 分布（供 violin/box plot 使用）
        self.last_epoch_alpha_raw = {}

    # ---- Loss Restraint 采集 ----

    def record_batch_loss(self, epoch, sub_ids, rel_ids, newadd_label,
                          loss_matrix, final_loss_matrix):
        """
        每个 batch 调用，累积统计 loss drop 情况。

        Args:
            epoch: 当前 epoch
            sub_ids: [batch_size] 头实体 ID
            rel_ids: [batch_size] 关系 ID
            newadd_label: [batch_size, num_ent] 辅助三元组掩码 (0=辅助, 1=其他)
            loss_matrix: [batch_size, num_ent] 原始 BCE loss
            final_loss_matrix: [batch_size, num_ent] 修改后 loss (被 drop 的为 0)
        """
        newadd_mask = (newadd_label == 0).bool()
        indices = newadd_mask.nonzero(as_tuple=False)

        if indices.size(0) == 0:
            return

        batch_indices = indices[:, 0]
        ent_indices = indices[:, 1]

        subs = sub_ids[batch_indices].tolist()
        rels = rel_ids[batch_indices].tolist()
        ents = ent_indices.tolist()

        losses = loss_matrix[batch_indices, ent_indices].tolist()
        final_losses = final_loss_matrix[batch_indices, ent_indices].tolist()

        for i in range(len(subs)):
            key = (subs[i], rels[i], ents[i])
            self.accumulated_count[key] = self.accumulated_count.get(key, 0) + 1
            self.accumulated_loss_sum[key] = self.accumulated_loss_sum.get(key, 0.0) + losses[i]
            if final_losses[i] == 0:
                self.accumulated_drop_count[key] = self.accumulated_drop_count.get(key, 0) + 1

    def finalize_epoch_loss(self, epoch):
        """Epoch 结束时调用，按置信度区间汇总 loss restraint 指标"""
        bins = {f"{i/5:.1f}-{(i+1)/5:.1f}": {'total': 0, 'dropped': 0, 'loss_sum': 0.0}
                for i in range(5)}
        for key, count in self.accumulated_count.items():
            conf = self.aux_confidence_map.get(key, 0.5)
            bin_idx = min(int(conf * 5), 4)
            bin_key = f"{bin_idx/5:.1f}-{(bin_idx+1)/5:.1f}"
            bins[bin_key]['total'] += count
            bins[bin_key]['dropped'] += self.accumulated_drop_count.get(key, 0)
            bins[bin_key]['loss_sum'] += self.accumulated_loss_sum.get(key, 0.0)
        for bin_key, data in bins.items():
            drop_rate = data['dropped'] / max(data['total'], 1)
            avg_loss = data['loss_sum'] / max(data['total'], 1)
            self.epoch_loss_summary.append({
                'epoch': epoch,
                'confidence_bin': bin_key,
                'drop_rate': drop_rate,
                'avg_loss': avg_loss,
                'total_count': data['total'],
            })
        self.accumulated_drop_count.clear()
        self.accumulated_loss_sum.clear()
        self.accumulated_count.clear()

    # ---- Attention Alpha 采集 ----

    def record_epoch_alpha(self, epoch, alpha, edge_index, edge_type):
        """
        Epoch 结束时全量记录 alpha。

        Args:
            epoch: 当前 epoch
            alpha: [num_edges] 注意力权重
            edge_index: [2, num_edges] 边索引
            edge_type: [num_edges] 边类型
        """
        alpha_cpu = alpha.detach().cpu()
        src_cpu = edge_index[0].cpu()
        dst_cpu = edge_index[1].cpu()
        rel_cpu = edge_type.cpu()

        num_edges = alpha_cpu.size(0)

        aux_alphas = {
            'original': [],
            **{f"{i/5:.1f}-{(i+1)/5:.1f}": [] for i in range(5)}
        }

        for i in range(num_edges):
            src = src_cpu[i].item()
            dst = dst_cpu[i].item()
            rel = rel_cpu[i].item()
            key = (src, rel, dst)
            alpha_val = alpha_cpu[i].item()

            if key in self.aux_triple_set:
                conf = self.aux_confidence_map.get(key, 0.5)
                bin_idx = min(int(conf * 5), 4)
                bin_key = f"{bin_idx/5:.1f}-{(bin_idx+1)/5:.1f}"
                aux_alphas[bin_key].append(alpha_val)
            else:
                aux_alphas['original'].append(alpha_val)

        # 汇总统计量
        for bin_key, alphas in aux_alphas.items():
            if alphas:
                self.epoch_alpha_summary.append({
                    'epoch': epoch,
                    'group': bin_key,
                    'alpha_mean': float(np.mean(alphas)),
                    'alpha_std': float(np.std(alphas)),
                    'alpha_median': float(np.median(alphas)),
                    'alpha_q25': float(np.percentile(alphas, 25)),
                    'alpha_q75': float(np.percentile(alphas, 75)),
                    'count': len(alphas),
                })

        # 保存原始分布（仅最后一轮，供 violin/box plot 使用）
        self.last_epoch_alpha_raw = {
            'epoch': epoch,
            'distributions': {},
        }
        for bin_key, alphas in aux_alphas.items():
            if alphas:
                self.last_epoch_alpha_raw['distributions'][bin_key] = alphas

    # ---- 持久化 ----

    def save(self):
        """保存采集数据到磁盘"""
        os.makedirs(self.output_dir, exist_ok=True)
        data = {
            'epoch_loss_summary': self.epoch_loss_summary,
            'epoch_alpha_summary': self.epoch_alpha_summary,
            'last_epoch_alpha_raw': {
                'epoch': self.last_epoch_alpha_raw.get('epoch', -1),
                'distributions': {
                    k: v for k, v in self.last_epoch_alpha_raw.get('distributions', {}).items()
                },
            },
        }
        save_path = os.path.join(self.output_dir, 'suppression_metrics.json')
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return save_path

    def save_interim(self, epoch):
        """训练中周期性保存，防丢失"""
        os.makedirs(self.output_dir, exist_ok=True)
        data = {
            'current_epoch': epoch,
            'epoch_loss_summary': self.epoch_loss_summary,
            'epoch_alpha_summary': self.epoch_alpha_summary,
        }
        save_path = os.path.join(self.output_dir, f'suppression_metrics_ep{epoch}.json')
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
