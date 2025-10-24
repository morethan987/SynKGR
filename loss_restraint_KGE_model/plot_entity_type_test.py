# 绘制不同度数实体表现差异的代码
# 使用:
# cdko && acko && python loss_restraint_KGE_model/plot_entity_type_test.py


import matplotlib.pyplot as plt
import os
import matplotlib.font_manager as fm
from matplotlib import rcParams

# --- 字体设置开始 ---
# 1. 设置中文字体：思源宋体 (Source Han Serif SC)
#    - 指定字体文件的绝对路径
chinese_font_path = '/data/yitingting/moran/KoGReD/loss_restraint_KGE_model/SourceHanSerifSC-Regular.otf'

# 检查文件是否存在
if os.path.exists(chinese_font_path):
    # a. 将字体文件添加到 Matplotlib 的缓存中
    fm.fontManager.addfont(chinese_font_path)

    # b. 获取字体内部定义的名称（通常是 "Source Han Serif SC" 或 "Source Han Serif SC Regular"）
    #    这里我们假设它的 family name 是 'Source Han Serif SC'
    #    你可能需要根据实际情况调整这个名称。
    #    如果设置不成功，可以尝试打印 fm.FontProperties(fname=chinese_font_path).get_name() 来确认。
    chinese_font_name = 'Source Han Serif SC'

    # c. 设置全局中文字体为衬线字体 (Serif)，因为宋体属于衬线字体
    rcParams['font.serif'] = [chinese_font_name, 'Times New Roman', 'Times', 'serif']
    rcParams['font.family'] = 'serif'
else:
    print(f"Warning: Chinese font file not found at {chinese_font_path}, using default system font.")
    # 如果文件不存在，回退到默认设置
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman', 'Times', 'serif']


# 2. 设置英文和数字字体：Times New Roman
#    由于中文字体设置中已经把 Times New Roman 作为备选，这里再次强调设置衬线字体族
#    英文和数字（非中文）将优先使用衬线字体族的第一个有效字体 (通常是 Times New Roman)
#    注意：中文文本会使用 'Source Han Serif SC'
#    数字和英文文本会使用 'Times New Roman' (因为它是衬线字体的第二优先选项，而 Source Han Serif SC 不包含 Times New Roman)
rcParams['font.sans-serif'] = ['Arial', 'sans-serif'] # 确保无衬线字体不会干扰

# 解决负号显示问题 (如果图中有负号)
rcParams['axes.unicode_minus'] = False

# --- 字体设置结束 ---


# CompGCN 的 MRR 数据
compgcn_mrr = [0.14, 0.16, 0.14, 0.17, 0.16, 0.26, 0.43, 0.54, 0.48, 0.53]

# 'ours' 模型的 MRR 数据
ours_mrr = [0.31, 0.28, 0.23, 0.20, 0.18, 0.37, 0.46, 0.68, 0.53, 0.71]

# 实体度数 (degree of entities) 的 x 轴标签
degree_of_entities = [
    '[0, 4)',
    '[4, 8)',
    '[8, 12)',
    '[12, 16)',
    '[16, 20)',
    '[20, 50)',
    '[50, 100)',
    '[100, 200)',
    '[200, 350)',
    '[350, max]'
]

# 创建图形和轴对象
fig, ax = plt.subplots(figsize=(8, 6))

# 绘制 CompGCN 的折线图
line1, = ax.plot(degree_of_entities, compgcn_mrr,
                 marker='o',  # 圆形标记
                 linestyle='-', # 实线
                 color='tab:blue', # 蓝色
                 label='CompGCN')

# 绘制 'ours' 的折线图
line2, = ax.plot(degree_of_entities, ours_mrr,
                 marker='o',
                 linestyle='-',
                 color='tab:red', # 红色
                 label='KoGReD')

# 在数据点上添加数值标签
# CompGCN 标签
for x, y in zip(degree_of_entities, compgcn_mrr):
    # 格式化数值到两位小数，并微调位置
    ax.text(x, y - 0.02, f'{y:.2f}',
            ha='center',
            va='top',
            color='tab:blue',
            fontsize=13)

# 'KoGReD' 标签
for x, y in zip(degree_of_entities, ours_mrr):
    # 格式化数值到两位小数，并微调位置
    ax.text(x, y + 0.015, f'{y:.2f}',
            ha='center',
            va='bottom',
            color='tab:red',
            fontsize=13)

# 设置 y 轴标签
# 因为设置了全局字体，这里的 '平均倒数排名' 会使用宋体
ax.set_ylabel('平均倒数排名', fontsize=15)

# 设置 x 轴标签
# 会使用宋体
ax.set_xlabel('实体度数', fontsize=15)

ax.tick_params(axis='x', labelsize=13)
# 旋转 x 轴刻度标签，避免重叠
plt.xticks(rotation=40, ha='center')

ax.tick_params(axis='y', labelsize=13)
# 设置 y 轴的刻度范围和刻度
ax.set_yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
# y 轴的刻度标签 (数字) 会使用 Times New Roman
ax.set_yticklabels(['0.2', '0.3', '0.4', '0.5', '0.6', '0.7'])
# 调整 y 轴的显示范围，使其与图更接近
ax.set_ylim(min(min(compgcn_mrr), min(ours_mrr)) - 0.05, max(max(compgcn_mrr), max(ours_mrr)) + 0.05)


# 添加图例
# 图例中的文本 'CompGCN' 和 'KoGReD' 会使用 Times New Roman
ax.legend(handles=[line1, line2], loc='upper left', fontsize=13)

# 调整布局以确保所有元素都可见
plt.tight_layout()

# 保存图片
plt.savefig('assets/fb_entity_type_test.png', dpi=300)
