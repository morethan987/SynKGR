"""
训练过程中采集抑制效果指标
用于可视化验证大损失约束和自适应消息聚合对低置信度辅助三元组的抑制效果

分箱策略：语义分区 + 区内等频混合分箱
  - Low 区 [0, 0.5): KG-BERT 认为可能错误，等频 2 子区间
  - Mid 区 [0.5, 0.9): 不确定区间，等频 2 子区间
  - High 区 [0.9, 1.0): KG-BERT 认为可能正确，等频 3 子区间
"""

import os
import json
import numpy as np
from collections import defaultdict

LOW_THRESHOLD = 0.5
HIGH_THRESHOLD = 0.9
LOW_SUB_BINS = 2
MID_SUB_BINS = 2
HIGH_SUB_BINS = 3


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

        if aux_confidence_map:
            self.bin_boundaries = self._compute_bin_boundaries(
                list(aux_confidence_map.values())
            )
        else:
            self.bin_boundaries = self._default_boundaries()
        self.bin_labels = self._make_bin_labels()

        self._print_bin_info()

        # Loss Restraint: 全局累积统计
        self.accumulated_drop_count = {}
        self.accumulated_loss_sum = {}
        self.accumulated_count = {}
        self.epoch_loss_summary = []

        # Attention Alpha: epoch 级全量记录
        self.epoch_alpha_summary = []

        # 最后一轮 epoch 的原始 alpha 分布（供 violin/box plot 使用）
        self.last_epoch_alpha_raw = {}

    # ---- 分箱 ----

    @staticmethod
    def _compute_bin_boundaries(confidence_values):
        """
        语义分区 + 区内等频混合分箱
        Args:
            confidence_values: 所有辅助三元组的置信度列表
        Returns:
            boundaries: [b0, b1, ..., bN] 升序边界列表
        """
        vals = np.sort(np.array(confidence_values))

        low_vals = vals[vals < LOW_THRESHOLD]
        mid_vals = vals[(vals >= LOW_THRESHOLD) & (vals < HIGH_THRESHOLD)]
        high_vals = vals[vals >= HIGH_THRESHOLD]

        regions = [
            (low_vals, LOW_SUB_BINS),
            (mid_vals, MID_SUB_BINS),
            (high_vals, HIGH_SUB_BINS),
        ]

        boundaries = [0.0]
        for region_vals, n_sub in regions:
            if len(region_vals) == 0:
                continue
            percentiles = np.linspace(0, 100, n_sub + 1)
            edges = np.percentile(region_vals, percentiles)
            boundaries.extend(edges[1:].tolist())

        boundaries[-1] = 1.001
        return boundaries

    @staticmethod
    def _default_boundaries():
        return [0.0, 0.25, 0.5, 0.7, 0.85, 0.93, 0.97, 1.001]

    def _bin_key(self, confidence):
        for i in range(len(self.bin_boundaries) - 1):
            lo = self.bin_boundaries[i]
            hi = self.bin_boundaries[i + 1]
            if lo <= confidence < hi:
                return f"{lo:.4f}-{hi:.4f}"
        last_lo = self.bin_boundaries[-2]
        last_hi = self.bin_boundaries[-1]
        return f"{last_lo:.4f}-{last_hi:.4f}"

    def _make_bin_labels(self):
        labels = []
        for i in range(len(self.bin_boundaries) - 1):
            lo = self.bin_boundaries[i]
            hi = self.bin_boundaries[i + 1]
            if hi <= LOW_THRESHOLD:
                region = 'Low'
            elif lo >= HIGH_THRESHOLD:
                region = 'High'
            else:
                region = 'Mid'
            labels.append({
                'key': f"{lo:.4f}-{hi:.4f}",
                'region': region,
                'lo': lo,
                'hi': hi,
            })
        return labels

    def _print_bin_info(self):
        print(f"MetricsCollector: {len(self.bin_labels)} bins (hybrid semantic + quantile)")
        for lbl in self.bin_labels:
            print(f"  [{lbl['lo']:.4f}, {lbl['hi']:.4f})  region={lbl['region']}")

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
        bins = {lbl['key']: {'total': 0, 'dropped': 0, 'loss_sum': 0.0}
                for lbl in self.bin_labels}
        for key, count in self.accumulated_count.items():
            conf = self.aux_confidence_map.get(key, 0.5)
            bin_key = self._bin_key(conf)
            if bin_key not in bins:
                continue
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
            **{lbl['key']: [] for lbl in self.bin_labels}
        }

        for i in range(num_edges):
            src = src_cpu[i].item()
            dst = dst_cpu[i].item()
            rel = rel_cpu[i].item()
            key = (src, rel, dst)
            alpha_val = alpha_cpu[i].item()

            if key in self.aux_triple_set:
                conf = self.aux_confidence_map.get(key, 0.5)
                bin_key = self._bin_key(conf)
                if bin_key in aux_alphas:
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
            'bin_config': {
                'boundaries': self.bin_boundaries,
                'labels': self.bin_labels,
                'method': 'semantic_quantile_hybrid',
            },
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
            'bin_config': {
                'boundaries': self.bin_boundaries,
                'labels': self.bin_labels,
                'method': 'semantic_quantile_hybrid',
            },
            'epoch_loss_summary': self.epoch_loss_summary,
            'epoch_alpha_summary': self.epoch_alpha_summary,
        }
        save_path = os.path.join(self.output_dir, f'suppression_metrics_ep{epoch}.json')
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
