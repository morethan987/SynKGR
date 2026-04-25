"""
可视化两个技术点对辅助三元组的抑制效果
输入: MetricsCollector 保存的 suppression_metrics.json
输出: 多张分析图表

核心规则：
- MIN_SAMPLES: 每 epoch 平均样本数低于此阈值的 bin 视为统计不可靠，不纳入可视化
- SMOOTH_WINDOW: 训练动态曲线的滑动平均窗口大小
"""

import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_theme(style="whitegrid", font_scale=1.2)

MIN_SAMPLES = 100
SMOOTH_WINDOW = 15


def load_metrics(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return (
        data.get('epoch_loss_summary', []),
        data.get('epoch_alpha_summary', []),
        data.get('last_epoch_alpha_raw', {}),
    )


def _get_sorted_bins(records, key='confidence_bin'):
    """从数据中自动提取并排序所有非空的 bin"""
    seen = set()
    for r in records:
        val = r.get(key, '')
        if val and val != 'original':
            seen.add(val)
    return sorted(seen, key=lambda b: float(b.split('-')[0]))


def _get_bin_sample_size(loss_summary, bin_key):
    """获取某个 bin 的典型每 epoch 样本数（取中位数）"""
    counts = [r['total_count'] for r in loss_summary
              if r['confidence_bin'] == bin_key and r['total_count'] > 0]
    return int(np.median(counts)) if counts else 0


def _get_reliable_bins(loss_summary):
    """返回统计可靠的 bin 列表（样本量 >= MIN_SAMPLES）及其样本数"""
    all_bins = _get_sorted_bins(loss_summary, 'confidence_bin')
    reliable = []
    for b in all_bins:
        n = _get_bin_sample_size(loss_summary, b)
        if n >= MIN_SAMPLES:
            reliable.append(b)
    skipped = [b for b in all_bins if b not in reliable]
    if skipped:
        for b in skipped:
            n = _get_bin_sample_size(loss_summary, b)
            print(f"  [SKIP] {b}: n={n} < MIN_SAMPLES={MIN_SAMPLES}, 统计不可靠")
    return reliable


def _smooth(values, window=SMOOTH_WINDOW):
    """滑动平均平滑"""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    smoothed = np.convolve(values, kernel, mode='same')
    for i in range(window // 2):
        w = 2 * i + 1
        smoothed[i] = np.mean(values[:w])
        smoothed[-(i + 1)] = np.mean(values[-w:])
    return smoothed.tolist()


def plot_drop_rate_vs_confidence(loss_summary, output_dir):
    """图表1: 大损失约束 vs 置信度 — Bar Chart"""
    if not loss_summary:
        print("[WARN] No loss summary data")
        return

    last_epochs = sorted(set(r['epoch'] for r in loss_summary))[-5:]
    recent = [r for r in loss_summary if r['epoch'] in last_epochs]

    bins_order = _get_reliable_bins(loss_summary)
    if not bins_order:
        print("[WARN] No reliable bins in loss summary")
        return

    data = {b: [] for b in bins_order}
    for r in recent:
        b = r['confidence_bin']
        if b in data:
            data[b].append(r['drop_rate'])

    means = [np.mean(data[b]) if data[b] else 0 for b in bins_order]
    sample_sizes = [_get_bin_sample_size(loss_summary, b) for b in bins_order]

    n = len(bins_order)
    fig_w = max(8, n * 0.8)
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    colors = sns.color_palette("RdYlGn", n)
    bars = ax.bar(range(n), means, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(n))
    labels_with_n = [f"{b}\n(n={s})" for b, s in zip(bins_order, sample_sizes)]
    ax.set_xticklabels(labels_with_n, rotation=45, ha='right')
    ax.set_xlabel('Confidence Bin')
    ax.set_ylabel('Drop Rate')
    ax.set_title('Loss Restraint: Drop Rate vs Discriminator Confidence\n(last 5 evaluation epochs)')
    ax.set_ylim(0, float(max(means)) * 1.2 if max(means) > 0 else 1)

    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f'{val:.3f}', ha='center', va='bottom', fontsize=max(6, 10 - n // 5))

    plt.tight_layout()
    path = f'{output_dir}/fig1_drop_rate_vs_confidence.png'
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")


def plot_alpha_distribution(alpha_summary, alpha_raw, loss_summary, output_dir):
    """图表2: 注意力权重分布 — Box Plot，带样本量标注"""
    distributions = alpha_raw.get('distributions', {})
    epoch_label = alpha_raw.get('epoch', '?')

    reliable_bins = _get_reliable_bins(loss_summary) if loss_summary else _get_sorted_bins(alpha_summary, 'group')

    if distributions:
        ordered_groups = [g for g in reliable_bins if g in distributions] + (
            ['original'] if 'original' in distributions else [])

        labels = []
        plot_data = []
        for g in ordered_groups:
            vals = distributions[g]
            if vals:
                label = f"Original\n(n={len(vals)})" if g == 'original' else f"{g}\n(n={len(vals)})"
                labels.append(label)
                plot_data.append(vals)

        if plot_data:
            n = len(labels)
            fig_w = max(10, n * 1.2)
            fig, ax = plt.subplots(figsize=(fig_w, 6))
            bp = ax.boxplot(plot_data, patch_artist=True, showfliers=False, widths=0.6)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            n_bins = len([l for l in labels if 'Original' not in l])
            colors = list(sns.color_palette("Reds", max(n_bins, 1))) + (
                [sns.color_palette("Blues")[3]] if 'original' in distributions else [])
            for patch, color in zip(bp['boxes'], colors[:len(plot_data)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            # 标注统计摘要
            stats_text = "Median: " + " | ".join(
                f"{np.median(d):.4f}" for d in plot_data)
            ax.text(0.02, 0.97, stats_text, transform=ax.transAxes,
                    fontsize=7, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            ax.set_ylabel('Attention Weight (alpha)')
            ax.set_title(f'Adaptive Aggregation: Alpha Distribution by Confidence Group\n(Epoch {epoch_label})')
            plt.tight_layout()
            path = f'{output_dir}/fig2_alpha_distribution.png'
            plt.savefig(path, dpi=200)
            plt.close()
            print(f"Saved: {path}")
            return

    if not alpha_summary:
        print("[WARN] No alpha summary data")
        return

    last_epoch = max(r['epoch'] for r in alpha_summary)
    recent = [r for r in alpha_summary if r['epoch'] == last_epoch]

    groups = [g for g in reliable_bins] + ['original']

    means = []
    q25_list = []
    q75_list = []
    labels = []
    for g in groups:
        for r in recent:
            if r['group'] == g:
                labels.append(g if g != 'original' else 'Original')
                means.append(r['alpha_mean'])
                q25_list.append(r.get('alpha_q25', r['alpha_mean'] - r['alpha_std']))
                q75_list.append(r.get('alpha_q75', r['alpha_mean'] + r['alpha_std']))
                break

    if not means:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(labels))
    yerr_low = [max(0, m - q25) for m, q25 in zip(means, q25_list)]
    yerr_high = [max(0, q75 - m) for m, q75 in zip(means, q75_list)]
    ax.bar(x, means, yerr=[yerr_low, yerr_high], capsize=5,
           edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Mean Attention Weight (alpha)')
    ax.set_title(f'Adaptive Aggregation: Attention Weight by Confidence Group\n(Epoch {last_epoch})')

    plt.tight_layout()
    path = f'{output_dir}/fig2_alpha_distribution.png'
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")


def plot_training_dynamics(loss_summary, alpha_summary, output_dir):
    """图表3: 训练过程中抑制强度变化 — Line Chart（滑动平均平滑）"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    reliable_bins = _get_reliable_bins(loss_summary)

    if loss_summary and reliable_bins:
        for b in reliable_bins:
            records = [r for r in loss_summary if r['confidence_bin'] == b and r['total_count'] > 0]
            records.sort(key=lambda x: x['epoch'])
            if not records:
                continue
            epochs = [r['epoch'] for r in records]
            drop_rates = [r['drop_rate'] for r in records]
            n = _get_bin_sample_size(loss_summary, b)

            ax1.plot(epochs, drop_rates, alpha=0.15, linewidth=0.8)
            smoothed = _smooth(drop_rates)
            ax1.plot(epochs, smoothed, linewidth=2, label=f"{b} (n={n})")

        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Drop Rate')
        ax1.set_title('Loss Restraint Dynamics (smoothed)')
        ax1.legend(fontsize=8)

    alpha_bins = _get_sorted_bins(alpha_summary, 'group')
    reliable_alpha = [b for b in alpha_bins if b in reliable_bins]

    if alpha_summary:
        groups = reliable_alpha + ['original']
        for g in groups:
            records = [r for r in alpha_summary if r['group'] == g]
            records.sort(key=lambda x: x['epoch'])
            if not records:
                continue
            epochs = [r['epoch'] for r in records]
            alphas = [r['alpha_mean'] for r in records]
            label = f"Original (n={records[0]['count']})" if g == 'original' else g

            ax2.plot(epochs, alphas, alpha=0.15, linewidth=0.8)
            smoothed = _smooth(alphas)
            ax2.plot(epochs, smoothed, linewidth=2, label=label)

        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Mean Attention Weight')
        ax2.set_title('Adaptive Aggregation Dynamics (smoothed)')
        ax2.legend(fontsize=8)

    plt.tight_layout()
    path = f'{output_dir}/fig3_training_dynamics.png'
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")


def _average_epochs(all_epochs, chunk_size=30):
    """将 epoch 分组，每组取平均值，返回标签列表和分组映射"""
    if not all_epochs:
        return [], {}
    if len(all_epochs) <= chunk_size:
        labels = [str(e) for e in all_epochs]
        mapping = {e: i for i, e in enumerate(all_epochs)}
        return labels, mapping

    mapping = {}
    groups = {}
    idx = 0
    for i in range(0, len(all_epochs), chunk_size):
        chunk = all_epochs[i:i + chunk_size]
        lo, hi = chunk[0], chunk[-1]
        label = f"{lo}-{hi}"
        groups[label] = idx
        for e in chunk:
            mapping[e] = idx
        idx += 1
    return list(groups.keys()), mapping


def plot_suppression_heatmap(loss_summary, alpha_summary, output_dir):
    """图表4: 综合抑制效应热力图 — 仅统计可靠的 bin，epoch 分组平均"""
    if not loss_summary or not alpha_summary:
        print("[WARN] No data for combined heatmap")
        return

    active_bins = _get_reliable_bins(loss_summary)
    if not active_bins:
        print("[WARN] No reliable bins in loss summary")
        return

    epochs_loss = sorted(set(r['epoch'] for r in loss_summary))
    epochs_alpha = sorted(set(r['epoch'] for r in alpha_summary))
    common_epochs = sorted(set(epochs_loss) & set(epochs_alpha))

    use_epochs = common_epochs if common_epochs else epochs_loss
    has_alpha = bool(common_epochs)

    labels, epoch_to_group = _average_epochs(use_epochs, chunk_size=30)
    n_groups = len(labels)

    alpha_lookup = {}
    if has_alpha:
        for r in alpha_summary:
            alpha_lookup.setdefault((r['epoch'], r['group']), []).append(r['alpha_mean'])

    matrix = np.zeros((len(active_bins), n_groups))
    counts = np.zeros((len(active_bins), n_groups))
    for r in loss_summary:
        b = r['confidence_bin']
        if b not in active_bins or r['epoch'] not in epoch_to_group:
            continue
        g_idx = epoch_to_group[r['epoch']]
        row = active_bins.index(b)
        dr = r['drop_rate']
        if has_alpha:
            alphas = alpha_lookup.get((r['epoch'], b), [1.0])
            alpha_avg = float(np.mean(alphas))
            suppression = (1.0 - alpha_avg) * dr
        else:
            suppression = dr
        matrix[row, g_idx] += suppression
        counts[row, g_idx] += 1

    mask = counts > 0
    matrix[mask] /= counts[mask]

    fig_h = max(4, len(active_bins) * 0.7)
    fig_w = max(8, n_groups * 1.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    annot_fmt = '.4f' if matrix.max() < 0.01 else '.3f'
    sns.heatmap(matrix, ax=ax, xticklabels=labels, yticklabels=active_bins,
                cmap='YlOrRd', annot=True, fmt=annot_fmt, linewidths=0.5,
                annot_kws={'size': max(7, 11 - len(active_bins) // 3)},
                cbar_kws={'label': 'Suppression Index = (1 - α) × drop_rate'})
    ax.set_xlabel('Epoch Range')
    ax.set_ylabel('Confidence Bin')
    ax.set_title('Combined Suppression Heatmap')

    plt.tight_layout()
    path = f'{output_dir}/fig4_suppression_heatmap.png'
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize suppression metrics')
    parser.add_argument('--input', required=True, help='Path to suppression_metrics.json')
    parser.add_argument('--output', default=None, help='Output directory for figures')
    args = parser.parse_args()

    import os
    output_dir = args.output or os.path.dirname(args.input)
    os.makedirs(output_dir, exist_ok=True)

    loss_summary, alpha_summary, alpha_raw = load_metrics(args.input)

    print(f"Loaded {len(loss_summary)} loss records, {len(alpha_summary)} alpha records")
    print(f"Filtering: MIN_SAMPLES={MIN_SAMPLES}")

    plot_drop_rate_vs_confidence(loss_summary, output_dir)
    plot_alpha_distribution(alpha_summary, alpha_raw, loss_summary, output_dir)
    plot_training_dynamics(loss_summary, alpha_summary, output_dir)
    plot_suppression_heatmap(loss_summary, alpha_summary, output_dir)

    print("All visualizations done.")


if __name__ == '__main__':
    main()
