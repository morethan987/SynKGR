import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Any

from MCTS.base_discriminator import BaseDiscriminator
from MCTS.prompts import ALPACA_PROMPT


class TriplesDiscriminator(BaseDiscriminator):

    TRUE_TOKEN_ID = 5852
    FALSE_TOKEN_ID = 7700

    def __init__(
        self,
        llm_path: str,
        lora_path: str,
        embedding_path: str,
        device: str = "",
        dtype: str = "fp16",
        batch_size: int = 8,
        threshold: float = 0.5,
    ):
        """
        Initializes the TriplesDiscriminator.

        Args:
            llm_path (str): Path to the base language model (e.g., Alpaca-7B).
            lora_path (str): Path to the trained LoRA weights.
            embedding_path (str): Path to the knowledge graph embeddings file (.pth).
            device (str, optional): The device to run the model on ('npu', 'cuda', 'cpu').
            dtype (str, optional): The data type for model computations ('bf16', 'fp16', 'fp32').
            batch_size (int, optional): The batch size for judging triples.
        """
        self.llm_path = llm_path
        self.lora_path = lora_path
        self.embedding_path = embedding_path
        self.batch_size = batch_size
        self.default_threshold = threshold
        self.device = device or ('npu' if torch.npu.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

        # Set torch dtype based on the string input
        self.torch_dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }.get(dtype, torch.bfloat16)

        self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self):
        """
        Loads the tokenizer, base model, LoRA weights, and KG embeddings,
        exactly following the logic of the provided test script.
        """
        print(f"Loading model components to device: {self.device}")

        # 1. Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_path, local_files_only=True)

        # 2. Load base model
        model = AutoModelForCausalLM.from_pretrained(
            self.llm_path,
            torch_dtype=self.torch_dtype,
            local_files_only=True
        ).to(self.device)

        # 3. Apply LoRA weights (PeftModel)
        self.model = PeftModel.from_pretrained(
            model,
            self.lora_path,
            torch_dtype=self.torch_dtype,
        ).to(self.device)

        # 4. Configure token IDs, matching the test script
        self.model.config.pad_token_id = self.tokenizer.pad_token_id = 0  # unk
        self.model.config.bos_token_id = 1
        self.model.config.eos_token_id = 2

        # 5. Load knowledge graph embeddings
        self.kg_embeddings = torch.load(self.embedding_path, map_location=self.device)

        # 6. Set model to evaluation mode
        self.model.eval()
        print("Model and tokenizer loaded successfully.")

    def _get_p_true_batch(self, batch_data: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Forward pass to get P(True) for a batch of triples.

        Returns:
            torch.Tensor of shape (batch_size,) with P(True) for each triple
        """
        prompts = [
            ALPACA_PROMPT.format(input=item["input"], output="")
            for item in batch_data
        ]

        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True
        )
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        token_embeds = self.model.model.model.embed_tokens(input_ids).to(self.torch_dtype)

        embedding_ids = torch.LongTensor(
            [item["embedding_ids"] for item in batch_data]
        ).to(self.device)
        prefix_embeds = self.kg_embeddings(embedding_ids).to(self.torch_dtype)

        input_embeds = torch.cat((prefix_embeds, token_embeds), dim=1)

        prefix_len = prefix_embeds.shape[1]
        prefix_attention = torch.ones(
            attention_mask.shape[0], prefix_len,
            dtype=attention_mask.dtype, device=self.device
        )
        full_attention_mask = torch.cat([prefix_attention, attention_mask], dim=1)

        outputs = self.model(
            inputs_embeds=input_embeds,
            attention_mask=full_attention_mask,
        )

        last_logits = outputs.logits[:, -1, :]
        logit_true = last_logits[:, self.TRUE_TOKEN_ID]
        logit_false = last_logits[:, self.FALSE_TOKEN_ID]
        binary_logits = torch.stack([logit_true, logit_false], dim=-1)
        p_true = torch.softmax(binary_logits, dim=-1)[:, 0]

        return p_true

    def judge_batch(self, triples_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Judges a list of triples using forward() + logit extraction.

        Args:
            triples_list: Each item contains "input" and "embedding_ids".

        Returns:
            List of dicts with "triple_str", "is_correct", and "confidence".
        """
        results = []
        with torch.no_grad():
            for i in range(0, len(triples_list), self.batch_size):
                batch_data = triples_list[i:i + self.batch_size]

                p_true = self._get_p_true_batch(batch_data)

                for j, score in enumerate(p_true):
                    results.append({
                        "triple_str": batch_data[j].get("input", ""),
                        "is_correct": score.item() >= self.default_threshold
                    })

        return results

    def calibrate(
        self,
        samples_with_labels: List[Dict[str, Any]],
    ):
        """
        在带标签的样本上搜索最优 P(True) 阈值。

        Args:
            samples_with_labels: 每个元素包含 "input", "embedding_ids", "label" (1=正, 0=负)
        """
        all_scores = []
        all_labels = []

        with torch.no_grad():
            total_batches = (len(samples_with_labels) + self.batch_size - 1) // self.batch_size
            for batch_idx, i in enumerate(range(0, len(samples_with_labels), self.batch_size)):
                batch = samples_with_labels[i:i + self.batch_size]
                batch_data = [{"input": s["input"], "embedding_ids": s["embedding_ids"]} for s in batch]
                p_true = self._get_p_true_batch(batch_data)
                for j, score in enumerate(p_true):
                    all_scores.append(score.item())
                    all_labels.append(batch[j]["label"])
                if (batch_idx + 1) % 50 == 0 or batch_idx == total_batches - 1:
                    print(f"[LLM Discriminator] Calibration progress: {batch_idx + 1}/{total_batches} batches")

        scores_arr = np.array(all_scores)
        labels_arr = np.array(all_labels)

        best_thresh = self.default_threshold
        best_f1 = 0.0

        candidate_thresholds = np.percentile(scores_arr, np.arange(5, 96, 2))
        for thresh in candidate_thresholds:
            preds = (scores_arr >= thresh).astype(int)
            tp = np.sum((preds == 1) & (labels_arr == 1))
            fp = np.sum((preds == 1) & (labels_arr == 0))
            fn = np.sum((preds == 0) & (labels_arr == 1))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else 0.0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = float(thresh)

        self.default_threshold = best_thresh
        print(f"[LLM Discriminator] Calibrated threshold: {best_thresh:.4f} (F1={best_f1:.4f})")
        print(f"  Score stats: min={scores_arr.min():.4f}, max={scores_arr.max():.4f}, "
              f"mean={scores_arr.mean():.4f}")
