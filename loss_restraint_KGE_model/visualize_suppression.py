"""
可视化两个技术点对辅助三元组的抑制效果
输入: MetricsCollector 保存的 suppression_metrics.json
输出: 多张分析图表
"""

import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_theme(style="whitegrid", font_scale=1.2)


def load_metrics(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return (
        data.get('epoch_loss_summary', []),
        data.get('epoch_alpha_summary', []),
        data.get('last_epoch_alpha_raw', {}),
    )


def plot_drop_rate_vs_confidence(loss_summary, output_dir):
    """图表1: 大损失约束 vs 置信度 — Bar Chart"""
    if not loss_summary:
        print("[WARN] No loss summary data")
        return

    last_epochs = sorted(set(r['epoch'] for r in loss_summary))[-5:]
    recent = [r for r in loss_summary if r['epoch'] in last_epochs]

    bins_order = [f"{i/5:.1f}-{(i+1)/5:.1f}" for i in range(5)]
    data = {b: [] for b in bins_order}
    for r in recent:
        b = r['confidence_bin']
        if b in data:
            data[b].append(r['drop_rate'])

    means = [np.mean(data[b]) if data[b] else 0 for b in bins_order]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(bins_order, means, color=sns.color_palette("RdYlGn", 5), edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Confidence Bin')
    ax.set_ylabel('Drop Rate')
    ax.set_title('Loss Restraint: Drop Rate vs Discriminator Confidence\n(last 5 evaluation epochs)')
    ax.set_ylim(0, max(means) * 1.2 if max(means) > 0 else 1)

    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    path = f'{output_dir}/fig1_drop_rate_vs_confidence.png'
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")


def plot_alpha_distribution(alpha_summary, alpha_raw, output_dir):
    """图表2: 注意力权重分布 — Box Plot (使用原始分布数据)"""
    distributions = alpha_raw.get('distributions', {})
    epoch_label = alpha_raw.get('epoch', '?')

    if distributions:
        groups = ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0', 'original']
        labels = []
        plot_data = []

        for g in groups:
            if g in distributions and distributions[g]:
                label = 'Original' if g == 'original' else g
                labels.append(label)
                plot_data.append(distributions[g])

        if plot_data:
            fig, ax = plt.subplots(figsize=(9, 5))
            bp = ax.boxplot(plot_data, labels=labels, patch_artist=True, showfliers=False,
                            widths=0.6)
            colors = list(sns.color_palette("Reds", 5)) + [sns.color_palette("Blues")[3]]
            for patch, color in zip(bp['boxes'], colors[:len(plot_data)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax.set_ylabel('Attention Weight (alpha)')
            ax.set_title(f'Adaptive Aggregation: Alpha Distribution by Confidence Group\n(Epoch {epoch_label})')
            plt.xticks(rotation=30)
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

    groups = ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0', 'original']
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
           color=[sns.color_palette("Reds", 5)[i] if i < 5 else sns.color_palette("Blues")[3]
                  for i in range(len(labels))],
           edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30)
    ax.set_ylabel('Mean Attention Weight (alpha)')
    ax.set_title(f'Adaptive Aggregation: Attention Weight by Confidence Group\n(Epoch {last_epoch})')

    plt.tight_layout()
    path = f'{output_dir}/fig2_alpha_distribution.png'
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")


def plot_training_dynamics(loss_summary, alpha_summary, output_dir):
    """图表3: 训练过程中抑制强度变化 — Line Chart"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Drop rate over epochs
    if loss_summary:
        bins_order = [f"{i/5:.1f}-{(i+1)/5:.1f}" for i in range(5)]
        for b in bins_order:
            records = [r for r in loss_summary if r['confidence_bin'] == b]
            records.sort(key=lambda x: x['epoch'])
            if records:
                epochs = [r['epoch'] for r in records]
                drop_rates = [r['drop_rate'] for r in records]
                ax1.plot(epochs, drop_rates, marker='o', markersize=3, label=b)
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Baseline (no restraint)')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Drop Rate')
        ax1.set_title('Loss Restraint Dynamics')
        ax1.legend(fontsize=8)

    # Right: Alpha over epochs
    if alpha_summary:
        groups = ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0', 'original']
        for g in groups:
            records = [r for r in alpha_summary if r['group'] == g]
            records.sort(key=lambda x: x['epoch'])
            if records:
                epochs = [r['epoch'] for r in records]
                alphas = [r['alpha_mean'] for r in records]
                label = 'Original' if g == 'original' else g
                ax2.plot(epochs, alphas, marker='o', markersize=3, label=label)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Mean Attention Weight')
        ax2.set_title('Adaptive Aggregation Dynamics')
        ax2.legend(fontsize=8)

    plt.tight_layout()
    path = f'{output_dir}/fig3_training_dynamics.png'
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")


def plot_suppression_heatmap(loss_summary, alpha_summary, output_dir):
    """图表4: 综合抑制效应热力图 — (1 - alpha_mean) × drop_rate"""
    if not loss_summary or not alpha_summary:
        print("[WARN] No data for combined heatmap")
        return

    bins_order = [f"{i/5:.1f}-{(i+1)/5:.1f}" for i in range(5)]

    # 收集各 epoch 各 bin 的 drop_rate 和 alpha_mean
    epochs_loss = sorted(set(r['epoch'] for r in loss_summary))
    epochs_alpha = sorted(set(r['epoch'] for r in alpha_summary))
    common_epochs = sorted(set(epochs_loss) & set(epochs_alpha))

    if not common_epochs:
        # 回退到仅 drop_rate 热力图
        epochs = epochs_loss
        matrix = np.zeros((5, len(epochs)))
        for i, b in enumerate(bins_order):
            for j, ep in enumerate(epochs):
                for r in loss_summary:
                    if r['confidence_bin'] == b and r['epoch'] == ep:
                        matrix[i, j] = r['drop_rate']

        fig, ax = plt.subplots(figsize=(max(10, len(epochs) * 0.3), 4))
        sns.heatmap(matrix, ax=ax, xticklabels=epochs, yticklabels=bins_order,
                    cmap='YlOrRd', annot=True, fmt='.2f', linewidths=0.5,
                    cbar_kws={'label': 'Drop Rate'})
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Confidence Bin')
        ax.set_title('Loss Restraint Suppression Heatmap')
        plt.tight_layout()
        path = f'{output_dir}/fig4_suppression_heatmap.png'
        plt.savefig(path, dpi=200)
        plt.close()
        print(f"Saved: {path}")
        return

    # 构建查找表
    alpha_lookup = {}
    for r in alpha_summary:
        alpha_lookup[(r['epoch'], r['group'])] = r['alpha_mean']

    loss_lookup = {}
    for r in loss_summary:
        loss_lookup[(r['epoch'], r['confidence_bin'])] = r['drop_rate']

    # 综合指标: suppression = (1 - alpha) * drop_rate
    matrix = np.zeros((5, len(common_epochs)))
    for i, b in enumerate(bins_order):
        for j, ep in enumerate(common_epochs):
            dr = loss_lookup.get((ep, b), 0.0)
            alpha = alpha_lookup.get((ep, b), 1.0)
            matrix[i, j] = (1.0 - alpha) * dr

    fig, ax = plt.subplots(figsize=(max(10, len(common_epochs) * 0.3), 4))
    sns.heatmap(matrix, ax=ax, xticklabels=common_epochs, yticklabels=bins_order,
                cmap='YlOrRd', annot=True, fmt='.3f', linewidths=0.5,
                cbar_kws={'label': 'Suppression Index = (1 - α) × drop_rate'})
    ax.set_xlabel('Epoch')
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

    plot_drop_rate_vs_confidence(loss_summary, output_dir)
    plot_alpha_distribution(alpha_summary, alpha_raw, output_dir)
    plot_training_dynamics(loss_summary, alpha_summary, output_dir)
    plot_suppression_heatmap(loss_summary, alpha_summary, output_dir)

    print("All visualizations done.")


if __name__ == '__main__':
    main()
