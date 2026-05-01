import os
from typing import List, Dict, Set, Tuple, Optional

from kg_data_loader import KGDataLoader
from model_calls import OpenKEClient
from mcts_tree import MCTS
from node import SearchRootNode, Context
from rollout_policy import UCB1Policy, LinUCBRolloutPolicy, MomentumRewardPolicy
from base_discriminator import BaseDiscriminator
from setup_logger import setup_logger, rank_logger


class KGEnhancer:
    """知识图谱增强器：为稀疏结点搜索新的三元组关系"""

    def __init__(self,
                 rank: int,
                 entity2name_path: str,
                 entity2embedding_path: str,
                 relation2id_path: str,
                 entity2id_path: str,
                 output_folder: str,
                 entity2description_path: str,
                 kg_path: str,
                 budget_per_entity: int = 1000,
                 mcts_iterations: int = 50,
                 leaf_threshold: int = 32,
                 exploration_weight: float = 1.0,
                 llm_path: str = "path/to/llm",
                 kge_path: str = "path/to/kge/model",
                 lora_path: str = None,
                 embedding_path: str = None,
                 device: str = "cuda",
                 dtype: str = "float32",
                 batch_size: int = 16,
                 no_sample: bool = True,
                 discriminator_type: str = "llm",
                 kgbert_model_dir: str = None,
                 kgbert_data_dir: str = None,
                 kge_discriminator_path: str = None,
                 valid_path: str = None,
                 target_depth: int = 4,
    ):
        """
        初始化知识图谱增强器

        Args:
            entity2name_path: 实体到名称映射文件路径
            relation2id_path: 关系到ID映射文件路径
            entity2description_path: 实体描述文件路径
            train_kg_path: 训练知识图谱文件路径
            budget_per_entity: 每个实体的分类器调用预算
            mcts_iterations: MCTS迭代次数
            leaf_threshold: 叶子结点候选实体数量阈值
            exploration_weight: UCT探索权重
        """
        self.logger = setup_logger(self.__class__.__name__)
        self.rank = rank
        self.output_folder = output_folder
        self.local_discovered_triplets = set()
        self.target_depth = target_depth
        self.valid_path = valid_path

        # 配置参数
        self.budget_per_entity = budget_per_entity
        self.mcts_iterations = mcts_iterations
        self.leaf_threshold = leaf_threshold
        self.exploration_weight = exploration_weight

        # 初始化策略类
        self.rollout_policy = MomentumRewardPolicy(rank=self.rank, exploration_factor=1.414)
        # self.rollout_policy = None # 纯随机
        # self.rollout_policy = UCB1Policy(rank=self.rank, exploration_factor=1.414)
        # self.rollout_policy = LinUCBRolloutPolicy(
        #     rank=self.rank,
        #     alpha=1.5,        # 稍微激进一点的探索
        #     lambda_reg=0.5    # 正则化参数
        # )

        # 初始化数据加载器
        self.logger.info("Loading knowledge graph data...")
        self.data_loader = KGDataLoader(
            entity2name_path=entity2name_path,
            entity2embedding_path=entity2embedding_path,
            relation2id_path=relation2id_path,
            entity2id_path=entity2id_path,
            entity2description_path=entity2description_path,
            kg_path=kg_path
        )

        # 初始化OpenKE客户端
        self.logger.info("Initializing OpenKE client...")
        self.kge_model = OpenKEClient(
            path=kge_path,
            model_name="RotatE",
            rank=self.rank
        )

        # 初始化三元组判别器
        self.logger.info(f"Initializing triplet discriminator (type={discriminator_type})...")
        self.triplet_discriminator = self._create_discriminator(
            discriminator_type=discriminator_type,
            llm_path=llm_path,
            lora_path=lora_path,
            embedding_path=embedding_path,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
            kgbert_model_dir=kgbert_model_dir,
            kgbert_data_dir=kgbert_data_dir,
            kge_discriminator_path=kge_discriminator_path,
        )

        # 初始化MCTS
        self.mcts = MCTS(
            rank=self.rank,
            exploration_weight=exploration_weight,
            rollout_policy=self.rollout_policy
        )

        self.all_entities = set(self.data_loader.entity2name.keys())

        self.logger.info("KGEnhancer initialized successfully")

    def _create_discriminator(
        self,
        discriminator_type: str,
        llm_path: str,
        lora_path: str,
        embedding_path: str,
        device: str,
        dtype: str,
        batch_size: int,
        kgbert_model_dir: str = None,
        kgbert_data_dir: str = None,
        kge_discriminator_path: str = None,
    ) -> BaseDiscriminator:
        """根据类型创建判别器实例"""
        if discriminator_type == "llm":
            from LLM_Discriminator.discriminator import TriplesDiscriminator
            discriminator = TriplesDiscriminator(
                llm_path=llm_path,
                lora_path=lora_path,
                embedding_path=embedding_path,
                device=device,
                dtype=dtype,
                batch_size=batch_size
            )
            calibration_data = self._prepare_llm_calibration_data()
            if calibration_data:
                import random as _random
                max_calibration = 2000
                if len(calibration_data) > max_calibration:
                    _random.shuffle(calibration_data)
                    calibration_data = calibration_data[:max_calibration]
                self.logger.info(
                    f"Calibrating LLM discriminator with {len(calibration_data)} samples")
                discriminator.calibrate(calibration_data)
            else:
                self.logger.warning(
                    "No calibration data for LLM discriminator, using default threshold")
            return discriminator
        elif discriminator_type == "kgbert":
            from kgbert_discriminator import KGBERTDiscriminator
            discriminator = KGBERTDiscriminator(
                model_dir=kgbert_model_dir,
                data_dir=kgbert_data_dir,
                batch_size=batch_size,
                device=device,
            )
            discriminator.set_id_mappings(
                id2entity=self.data_loader.id2entity,
                id2relation=self.data_loader.id2relation,
            )
            return discriminator
        elif discriminator_type == "kge":
            from kge_discriminator import KGEDiscriminator
            return KGEDiscriminator(
                model_path=kge_discriminator_path,
                model_name="RotatE",
                device=device,
                batch_size=batch_size,
            )
        elif discriminator_type == "random":
            from random_discriminator import RandomDiscriminator
            return RandomDiscriminator(positive_rate=0.5)
        else:
            raise ValueError(
                f"Unknown discriminator type: '{discriminator_type}'. "
                f"Supported: 'llm', 'kgbert', 'kge', 'random'"
            )

    def _load_valid_triple_ids(self):
        """加载验证集三元组并转换为 embedding ID 格式，用于 LLM 判别器校准"""
        valid_path = self.valid_path
        if valid_path is None or not os.path.isfile(valid_path):
            return []

        entity2id = self.data_loader.entity2id
        relation2id = self.data_loader.relation2id
        valid_ids = []

        with open(valid_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 3:
                    h_id = entity2id.get(parts[0])
                    r_id = relation2id.get(parts[1])
                    t_id = entity2id.get(parts[2])
                    if h_id is not None and r_id is not None and t_id is not None:
                        valid_ids.append((h_id, r_id, t_id))

        return valid_ids

    def _prepare_llm_calibration_data(self, num_neg_per_positive: int = 10):
        """
        准备 LLM 判别器校准数据（正样本 + 随机负样本）。
        使用验证集三元组作为正样本，随机替换尾实体生成负样本。
        """
        import random

        valid_path = self.valid_path
        if valid_path is None or not os.path.isfile(valid_path):
            return []

        entity2id = self.data_loader.entity2id
        relation2id = self.data_loader.relation2id
        entity2name = self.data_loader.entity2name
        all_entity_ids = list(entity2id.values())

        calibration_samples = []

        with open(valid_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) != 3:
                    continue
                h_str, r_str, t_str = parts
                h_id = entity2id.get(h_str)
                r_id = relation2id.get(r_str)
                t_id = entity2id.get(t_str)
                if h_id is None or r_id is None or t_id is None:
                    continue

                input_text = "The input triple: \n( {head}, {rel}, {tail} )\n".format(
                    head=entity2name.get(h_str, h_str).replace('_', ' '),
                    rel=r_str.replace('/', ' '),
                    tail=entity2name.get(t_str, t_str).replace('_', ' '),
                )
                calibration_samples.append({
                    "input": input_text,
                    "embedding_ids": [h_id, r_id, t_id],
                    "label": 1,
                })

                for _ in range(num_neg_per_positive):
                    neg_t_id = random.choice(all_entity_ids)
                    if neg_t_id == t_id:
                        continue
                    neg_t_str = self.data_loader.id2entity.get(neg_t_id, "")
                    if not neg_t_str:
                        continue
                    neg_input_text = "The input triple: \n( {head}, {rel}, {tail} )\n".format(
                        head=entity2name.get(h_str, h_str).replace('_', ' '),
                        rel=r_str.replace('/', ' '),
                        tail=entity2name.get(neg_t_str, neg_t_str).replace('_', ' '),
                    )
                    calibration_samples.append({
                        "input": neg_input_text,
                        "embedding_ids": [h_id, r_id, neg_t_id],
                        "label": 0,
                    })

        self.logger.info(
            f"Prepared {len(calibration_samples)} calibration samples for LLM discriminator")
        return calibration_samples

    def enhance_entity_relation(self, sparse_entity: str, position: str, relation: str) -> Set[Tuple[str, str, str]]:
        """
        为指定的稀疏实体-位置-关系组合搜索正确的三元组

        Args:
            sparse_entity: 稀疏实体ID
            position: 实体在三元组中的位置 ('head' 或 'tail')
            relation: 关系类型

        Returns:
            发现的正确三元组列表 [(head, relation, tail), ...]
        """
        rank_logger(self.logger, self.rank)(
            f"Starting enhancement for {sparse_entity}-{position}-{relation}")

        # 获取所有候选目标实体（除了稀疏实体本身）
        candidate_entities = self.all_entities - {sparse_entity}

        rank_logger(self.logger, self.rank)(
            f"Total candidate entities: {len(candidate_entities)}")

        # 构建上下文信息
        context = Context(
            rank=self.rank,
            sparse_entity=sparse_entity,
            position=position,
            relation=relation,
            unfiltered_entities=candidate_entities,
            output_folder=self.output_folder,
            data_loader=self.data_loader,
            triplet_discriminator=self.triplet_discriminator,
            kge_model=self.kge_model,
            leaf_threshold=self.leaf_threshold,
            parent=None
        )

        # 创建搜索根节点
        root_node = SearchRootNode(context=context, target_depth=self.target_depth)

        # 重置MCTS状态
        self.mcts.reset()

        # 记录已发现的正确三元组
        discovered_triplets = []
        budget_used = 0

        # MCTS搜索循环
        for iteration in range(self.mcts_iterations):
            if budget_used >= self.budget_per_entity:
                rank_logger(self.logger, self.rank)(
                    f"Budget exhausted after {iteration} iterations")
                break

            rank_logger(self.logger, self.rank)(f"MCTS iteration {iteration + 1}/{self.mcts_iterations}, "
                             f"budget used: {budget_used}/{self.budget_per_entity}")

            # 执行一次MCTS迭代
            triplets_found, budget_increment = self.mcts.do_iteration(root_node)

            # 更新统计信息
            discovered_triplets.extend(triplets_found)
            budget_used += budget_increment

            rank_logger(self.logger, self.rank)(f"Iteration {iteration + 1} found {len(triplets_found)} triplets, "
                             f"used {budget_increment} budget")

        # 去重
        discovered_triplets = set(discovered_triplets)

        # 更新本地存储
        self.local_discovered_triplets.update(discovered_triplets)

        rank_logger(self.logger, self.rank)(f"Enhancement completed: found {len(discovered_triplets)} unique triplets, "
                         f"total budget used: {budget_used}")

        return discovered_triplets

    def get_statistics(self) -> Dict:
        """获取增强过程的统计信息"""
        return {
            "total_entities": len(self.data_loader.entity2name),
            "total_relations": len(self.data_loader.relation2id),
            "total_kg_triplets": len(self.data_loader.kg_triplets)
        }
