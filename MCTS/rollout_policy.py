from collections import defaultdict
import math
import random
import numpy as np
from typing import TYPE_CHECKING, List, Tuple

from setup_logger import setup_logger, rank_logger
if TYPE_CHECKING:
    from node import SearchNode

class UCB1Policy:
    """
    一个MCTS的Rollout策略，它使用上下文多臂老虎机算法（基于UCB1）进行在线学习。

    这个类的实例应该在多个MCTS任务之间共享，以便积累知识。
    """
    def __init__(self, rank):
        self.logger = setup_logger(self.__class__.__name__)
        self.rank = rank
        # 存储每个 (状态, 动作) 对的累计奖励
        # 结构: self.q_values[state_key][action_key] = total_reward
        self.q_values = defaultdict(lambda: defaultdict(float))

        # 存储每个 (状态, 动作) 对被选择的次数
        # 结构: self.counts[state_key][action_key] = count
        self.counts = defaultdict(lambda: defaultdict(int))

        # 存储每个状态被访问的总次数
        self.state_total_counts = defaultdict(int)

    def _get_state_key(self, node: 'SearchNode') -> str:
        """将当前节点的状态转化为一个离散的、可作为字典键的字符串。"""

        # 1. 离散化候选实体的数量 (这是最重要的上下文)
        num_candidates = len(node.unfiltered_entities)
        if num_candidates > 1000:
            size_bucket = "large"  # 候选集 > 1000
        elif num_candidates > 100:
            size_bucket = "medium" # 候选集 101-1000
        else:
            size_bucket = "small"  # 候选集 <= 100

        # 2. (可选) 可以在此加入更多上下文信息，例如在MCTS树中的深度
        #    为此，你需要在 SearchNode 中实现一个 get_depth() 方法
        # depth = node.get_depth()
        # depth_bucket = f"depth:{depth // 2}" # 每2层深度作为一个桶

        return f"size:{size_bucket}"

    def get_action(self, node: 'SearchNode', potential_actions: list) -> type:
        """根据UCB1算法选择最佳动作（即节点类型）。"""
        state_key = self._get_state_key(node)
        total_visits_at_state = self.state_total_counts[state_key]

        # 如果这是一个全新的状态，没有任何历史数据，则随机探索
        if total_visits_at_state == 0:
            return random.choice(potential_actions)

        best_action_class = None
        max_ucb_score = -1.0

        for action_class in potential_actions:
            action_key = action_class.__name__  # 例如 "KGENode", "GraphNode"

            # 如果某个动作在这个状态下从未被尝试过，优先选择它（无限大UCB值）
            if self.counts[state_key][action_key] == 0:
                return action_class

            # 1. 计算“利用”项 (Exploitation): 该动作的平均奖励
            average_reward = self.q_values[state_key][action_key] / self.counts[state_key][action_key]

            # 2. 计算“探索”项 (Exploration): 不确定性带来的奖励加成
            exploration_bonus = math.sqrt(
                2 * math.log(total_visits_at_state) / self.counts[state_key][action_key]
            )

            ucb_score = average_reward + exploration_bonus

            if ucb_score > max_ucb_score:
                max_ucb_score = ucb_score
                best_action_class = action_class

        return best_action_class

    def update(self, rollout_path: List[Tuple['SearchNode', type]], reward: float):
        """使用一次完整Rollout的结果来更新策略模型的权重。"""
        for state_node, action_class in rollout_path:
            state_key = self._get_state_key(state_node)
            action_key = action_class.__name__

            # 更新访问次数
            self.counts[state_key][action_key] += 1
            self.state_total_counts[state_key] += 1

            # 使用增量方式更新Q值（平均奖励），以保证数值稳定性
            # Q_new = Q_old + (reward - Q_old) / N
            q_old = self.q_values[state_key][action_key]
            n = self.counts[state_key][action_key]
            self.q_values[state_key][action_key] += (reward - q_old) / n
        rank_logger(self.logger, self.rank)(f"Updated policy with reward: {reward}, path length: {len(rollout_path)}")

    def get_state(self) -> dict:
        """
        返回策略的当前状态，以便于序列化保存。
        将 defaultdict 转换为普通 dict 以便 JSON 兼容。
        """
        return {
            "q_values": dict(self.q_values),
            "counts": dict(self.counts),
            "state_total_counts": dict(self.state_total_counts),
        }

    def load_state(self, state: dict):
        """
        从一个字典加载策略的状态。
        """
        # 从加载的普通 dict 恢复 defaultdict 的内容
        self.q_values.update(state.get("q_values", {}))
        self.counts.update(state.get("counts", {}))
        self.state_total_counts.update(state.get("state_total_counts", {}))
        rank_logger(self.logger, self.rank)("Successfully loaded rollout policy state from checkpoint.")

class EnhancedUCB1Policy:
    """
    增强版UCB1 Rollout策略

    改进点：
    1. 更细粒度的状态分桶（5个桶 → 10个桶）
    2. 加入深度信息
    3. 加入稀疏实体度数信息
    4. 更稳定的Q值更新方式
    """

    def __init__(self, rank):
        self.logger = setup_logger(self.__class__.__name__)
        self.rank = rank

        # 存储每个 (状态, 动作) 对的平均奖励
        self.q_values = defaultdict(lambda: defaultdict(float))

        # 存储每个 (状态, 动作) 对被选择的次数
        self.counts = defaultdict(lambda: defaultdict(int))

        # 存储每个状态被访问的总次数
        self.state_total_counts = defaultdict(int)

    def _get_state_key(self, node: 'SearchNode') -> str:
        """
        将当前节点的状态转化为离散的键

        改进：使用更细粒度的分桶和多维度信息
        """
        # 1. 候选实体数量分桶（更细粒度）
        num_candidates = len(node.unfiltered_entities)

        if num_candidates > 10000:
            size_bucket = "huge"      # > 10000
        elif num_candidates > 5000:
            size_bucket = "xlarge"    # 5001-10000
        elif num_candidates > 2000:
            size_bucket = "large"     # 2001-5000
        elif num_candidates > 1000:
            size_bucket = "medium+"   # 1001-2000
        elif num_candidates > 500:
            size_bucket = "medium"    # 501-1000
        elif num_candidates > 200:
            size_bucket = "medium-"   # 201-500
        elif num_candidates > 100:
            size_bucket = "small+"    # 101-200
        elif num_candidates > 50:
            size_bucket = "small"     # 51-100
        elif num_candidates > 20:
            size_bucket = "tiny"      # 21-50
        else:
            size_bucket = "minimal"   # <= 20

        # 2. 搜索深度分桶
        depth = self._get_depth(node)
        if depth == 0:
            depth_bucket = "root"
        elif depth <= 2:
            depth_bucket = "shallow"
        else:
            depth_bucket = "deep"

        # 3. 稀疏实体度数分桶（可选，如果计算开销可接受）
        try:
            degree = len(node.data_loader.get_one_hop_neighbors(node.sparse_entity))
            if degree > 100:
                degree_bucket = "high_degree"
            elif degree > 20:
                degree_bucket = "medium_degree"
            else:
                degree_bucket = "low_degree"
        except:
            degree_bucket = "unknown"

        # 组合所有维度
        return f"size:{size_bucket}|depth:{depth_bucket}|deg:{degree_bucket}"

    def _get_depth(self, node: 'SearchNode') -> int:
        """计算节点深度"""
        depth = 0
        temp_node = node
        while temp_node.parent is not None:
            depth += 1
            temp_node = temp_node.parent
        return depth

    def get_action(self, node: 'SearchNode', potential_actions: list) -> type:
        """根据UCB1算法选择最佳动作"""
        state_key = self._get_state_key(node)
        total_visits_at_state = self.state_total_counts[state_key]

        # 如果这是全新状态，随机探索
        if total_visits_at_state == 0:
            return random.choice(potential_actions)

        best_action_class = None
        max_ucb_score = -1.0

        for action_class in potential_actions:
            action_key = action_class.__name__

            # 如果某个动作在这个状态下从未被尝试过，优先选择它
            if self.counts[state_key][action_key] == 0:
                return action_class

            # 计算UCB值
            average_reward = self.q_values[state_key][action_key]

            exploration_bonus = math.sqrt(
                2 * math.log(total_visits_at_state) / self.counts[state_key][action_key]
            )

            ucb_score = average_reward + exploration_bonus

            if ucb_score > max_ucb_score:
                max_ucb_score = ucb_score
                best_action_class = action_class

        return best_action_class

    def update(self, rollout_path: List[Tuple['SearchNode', type]], reward: float):
        """
        使用rollout结果更新策略

        改进：使用增量方式更新Q值，数值更稳定
        """
        for state_node, action_class in rollout_path:
            state_key = self._get_state_key(state_node)
            action_key = action_class.__name__

            # 更新访问次数
            self.counts[state_key][action_key] += 1
            self.state_total_counts[state_key] += 1

            # 增量更新Q值：Q_new = Q_old + (reward - Q_old) / N
            n = self.counts[state_key][action_key]
            q_old = self.q_values[state_key][action_key]
            self.q_values[state_key][action_key] = q_old + (reward - q_old) / n

        # 定期日志（包含更多诊断信息）
        if self.state_total_counts and sum(self.state_total_counts.values()) % 100 == 0:
            # 统计各个动作的使用频率
            action_totals = defaultdict(int)
            for state_counts in self.counts.values():
                for action, count in state_counts.items():
                    action_totals[action] += count

            rank_logger(self.logger, self.rank)(
                f"UCB1 update: reward={reward:.4f}, path_len={len(rollout_path)}, "
                f"total_states={len(self.state_total_counts)}, "
                f"action_distribution={dict(action_totals)}"
            )

    def get_state(self) -> dict:
        """序列化策略状态"""
        return {
            "q_values": {k: dict(v) for k, v in self.q_values.items()},
            "counts": {k: dict(v) for k, v in self.counts.items()},
            "state_total_counts": dict(self.state_total_counts),
        }

    def load_state(self, state: dict):
        """从字典加载策略状态"""
        # 加载Q值
        for state_key, action_dict in state.get("q_values", {}).items():
            self.q_values[state_key].update(action_dict)

        # 加载计数
        for state_key, action_dict in state.get("counts", {}).items():
            self.counts[state_key].update(action_dict)

        # 加载状态总计数
        self.state_total_counts.update(state.get("state_total_counts", {}))

        total_updates = sum(self.state_total_counts.values())
        rank_logger(self.logger, self.rank)(
            f"Loaded UCB1 policy: {total_updates} total updates, "
            f"{len(self.state_total_counts)} unique states"
        )

    def get_diagnostics(self) -> dict:
        """获取诊断信息"""
        # 统计每个状态的最优动作
        state_preferences = {}
        for state_key, action_q_values in self.q_values.items():
            if action_q_values:
                best_action = max(action_q_values.items(), key=lambda x: x[1])
                state_preferences[state_key] = best_action[0]

        # 统计全局动作分布
        action_totals = defaultdict(int)
        for state_counts in self.counts.values():
            for action, count in state_counts.items():
                action_totals[action] += count

        return {
            'total_states': len(self.state_total_counts),
            'total_updates': sum(self.state_total_counts.values()),
            'action_distribution': dict(action_totals),
            'sample_preferences': dict(list(state_preferences.items())[:5])
        }

class LinUCBRolloutPolicy:
    """
    基于LinUCB的MCTS Rollout策略
    使用线性模型估计每个动作的期望奖励：
    E[r|x,a] ≈ θ_a^T · x

    """

    def __init__(self, rank, alpha=1.0, lambda_reg=1.0):
        """
        Args:
            rank: 进程rank
            alpha: 探索参数（推荐1.0-2.0）
            lambda_reg: 正则化参数，防止矩阵奇异（推荐0.1-1.0）
        """
        self.logger = setup_logger(self.__class__.__name__)
        self.rank = rank
        self.alpha = alpha
        self.lambda_reg = lambda_reg

        # 特征维度（修复后）
        self.feature_dim = 5

        # 为每个动作维护LinUCB模型
        self.A = defaultdict(lambda: self.lambda_reg * np.identity(self.feature_dim))
        self.b = defaultdict(lambda: np.zeros(self.feature_dim))

        # 统计信息
        self.action_counts = defaultdict(int)
        self.total_updates = 0

        # 用于特征归一化的统计量（在线更新）
        self.feature_stats = {
            'log_candidates_max': 5.0,  # log10(100000)
            'depth_max': 10.0,
            'degree_max': 2.0  # log10(100)
        }

    def _get_feature_vector(self, node: 'SearchNode') -> np.ndarray:
        """
        提取状态特征向量（移除数据泄露）

        特征设计原则：
        1. 只包含当前状态的客观信息
        2. 不包含任何动作的历史统计
        3. 保持数值稳定性
        """
        features = []

        # 特征1: 候选实体数量（对数尺度，归一化）
        num_candidates = len(node.unfiltered_entities)
        log_candidates = math.log10(max(num_candidates, 1))
        features.append(log_candidates / self.feature_stats['log_candidates_max'])

        # 特征2: 搜索深度（归一化）
        depth = self._get_depth(node)
        features.append(min(depth / self.feature_stats['depth_max'], 1.0))

        # 特征3: 候选数相对根节点的比例
        root = node._get_root()
        root_size = len(root.unfiltered_entities)
        ratio = num_candidates / max(root_size, 1)
        features.append(ratio)

        # 特征4: 稀疏实体的度数（对数尺度）
        degree = len(node.data_loader.get_one_hop_neighbors(node.sparse_entity))
        log_degree = math.log10(max(degree, 1))
        features.append(log_degree / self.feature_stats['degree_max'])

        # 特征5: 深度与候选数的交互项（捕捉非线性关系）
        # 例如：深度大+候选数多 可能需要不同策略
        features.append(features[1] * features[0])  # depth * log_candidates

        return np.array(features, dtype=np.float64)

    def _get_depth(self, node: 'SearchNode') -> int:
        """计算节点深度"""
        depth = 0
        temp_node = node
        while temp_node.parent is not None:
            depth += 1
            temp_node = temp_node.parent
        return depth

    def get_action(self, node: 'SearchNode', potential_actions: list) -> type:
        """
        使用LinUCB算法选择动作

        对每个动作a：
        1. 计算预测奖励：θ_a^T · x
        2. 计算置信区间：alpha * sqrt(x^T · A_a^{-1} · x)
        3. 选择 UCB = 预测 + 置信区间 最大的动作
        """
        x = self._get_feature_vector(node)

        best_action = None
        max_ucb = -float('inf')

        for action_class in potential_actions:
            action_name = action_class.__name__

            try:
                # 计算参数估计 θ_a = A_a^{-1} · b_a
                A_inv = np.linalg.inv(self.A[action_name])
                theta = A_inv @ self.b[action_name]

                # 预测奖励
                predicted_reward = theta @ x

                # 置信区间
                confidence_radius = self.alpha * np.sqrt(
                    max(x @ A_inv @ x, 0.0)  # 防止数值误差导致负数
                )

                ucb_value = predicted_reward + confidence_radius

            except np.linalg.LinAlgError:
                # 矩阵奇异，使用随机策略
                rank_logger(self.logger, self.rank)(
                    f"Warning: Singular matrix for {action_name}, using random selection"
                )
                ucb_value = random.random()

            if ucb_value > max_ucb:
                max_ucb = ucb_value
                best_action = action_class

        return best_action

    def update(self, rollout_path: List[Tuple['SearchNode', type]], reward: float):
        """
        更新LinUCB模型参数

        对路径中的每个(状态, 动作)对：
        1. 提取特征向量x
        2. 更新设计矩阵：A_a += x · x^T
        3. 更新奖励向量：b_a += r · x
        """
        self.total_updates += 1

        # 奖励塑形：给不同深度的节点不同权重
        # 深度越深的决策影响越小，应该获得衰减的奖励
        path_length = len(rollout_path)

        for idx, (state_node, action_class) in enumerate(rollout_path):
            action_name = action_class.__name__
            x = self._get_feature_vector(state_node)

            # 计算该节点的有效奖励（深度衰减）
            # 最后一个节点权重1.0，往前逐渐衰减到0.5
            decay_factor = 0.5 + 0.5 * (idx + 1) / path_length
            effective_reward = reward * decay_factor

            # LinUCB更新
            self.A[action_name] += np.outer(x, x)
            self.b[action_name] += effective_reward * x

            # 统计
            self.action_counts[action_name] += 1

        # 定期日志（包含更多诊断信息）
        if self.total_updates % 50 == 0:
            # 计算当前策略的偏好（哪个动作的预测奖励最高）
            action_preferences = {}
            for action_name in ['GraphNode', 'KGENode', 'LLMNode']:
                try:
                    A_inv = np.linalg.inv(self.A[action_name])
                    theta = A_inv @ self.b[action_name]
                    avg_pred = np.mean(theta)  # 粗略的平均预测
                    action_preferences[action_name] = f"{avg_pred:.4f}"
                except:
                    action_preferences[action_name] = "N/A"

            rank_logger(self.logger, self.rank)(
                f"LinUCB #{self.total_updates}: reward={reward:.4f}, "
                f"path_len={len(rollout_path)}, "
                f"counts={dict(self.action_counts)}, "
                f"preferences={action_preferences}"
            )

    def get_state(self) -> dict:
        """序列化状态"""
        return {
            'A': {k: v.tolist() for k, v in self.A.items()},
            'b': {k: v.tolist() for k, v in self.b.items()},
            'action_counts': dict(self.action_counts),
            'total_updates': self.total_updates,
            'feature_dim': self.feature_dim,
            'alpha': self.alpha,
            'lambda_reg': self.lambda_reg,
            'feature_stats': self.feature_stats
        }

    def load_state(self, state: dict):
        """加载状态"""
        for action_name, matrix in state.get('A', {}).items():
            self.A[action_name] = np.array(matrix)
        for action_name, vector in state.get('b', {}).items():
            self.b[action_name] = np.array(vector)

        self.action_counts.update(state.get('action_counts', {}))
        self.total_updates = state.get('total_updates', 0)
        self.feature_dim = state.get('feature_dim', self.feature_dim)
        self.alpha = state.get('alpha', self.alpha)
        self.lambda_reg = state.get('lambda_reg', self.lambda_reg)
        self.feature_stats = state.get('feature_stats', self.feature_stats)

        rank_logger(self.logger, self.rank)(
            f"Loaded LinUCB: {self.total_updates} updates, "
            f"alpha={self.alpha}, lambda={self.lambda_reg}"
        )

    def get_diagnostics(self) -> dict:
        """获取诊断信息（用于调试）"""
        diagnostics = {
            'total_updates': self.total_updates,
            'action_counts': dict(self.action_counts),
            'matrix_conditions': {}
        }

        # 检查矩阵条件数（数值稳定性指标）
        for action_name in ['GraphNode', 'KGENode', 'LLMNode']:
            try:
                cond = np.linalg.cond(self.A[action_name])
                diagnostics['matrix_conditions'][action_name] = float(cond)
            except:
                diagnostics['matrix_conditions'][action_name] = float('inf')

        return diagnostics
