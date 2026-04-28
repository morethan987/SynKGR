import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Any

from MCTS.base_discriminator import BaseDiscriminator
from MCTS.prompts import ALPACA_PROMPT


class TriplesDiscriminator(BaseDiscriminator):
    def __init__(
        self,
        llm_path: str,
        lora_path: str,
        embedding_path: str,
        device: str = None,
        dtype: str = "bf16",
        batch_size: int = 8,
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

    def judge_batch(self, triples_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Judges a list of triples in batches under the simplified assumption that
        each triple corresponds to exactly 3 embedding IDs.

        Args:
            triples_list (List[Dict[str, Any]]): A list where each item is a
                dictionary containing "input" and "embedding_ids" (assumed to be a list of 3 IDs).

        Returns:
            List[Dict[str, Any]]: A list of prediction results.
        """
        results = []
        with torch.no_grad():
            for i in range(0, len(triples_list), self.batch_size):
                batch_data = triples_list[i:i + self.batch_size]

                prompts = [
                    ALPACA_PROMPT.format(input=item["input"], output="") for item in batch_data
                ]
                embedding_ids = torch.LongTensor([item["embedding_ids"] for item in batch_data]).to(self.device)

                inputs = self.tokenizer(
                    prompts, return_tensors="pt", padding=True, truncation=True, max_length=512
                )
                input_ids = inputs.input_ids.to(self.device)

                token_embeds = self.model.model.model.embed_tokens(input_ids).to(self.torch_dtype)
                prefix_embeds = self.kg_embeddings(embedding_ids).to(self.torch_dtype)
                input_embeds = torch.cat((prefix_embeds, token_embeds), dim=1)

                true_token_id = self.tokenizer.encode("True", add_special_tokens=False)[0]
                false_token_id = self.tokenizer.encode("False", add_special_tokens=False)[0]

                outputs = self.model(
                    inputs_embeds=input_embeds,
                    output_hidden_states=False,
                )
                last_logits = outputs.logits[:, -1, :]
                true_logit = last_logits[:, true_token_id]
                false_logit = last_logits[:, false_token_id]
                probs = torch.softmax(torch.stack([true_logit, false_logit], dim=-1), dim=-1)
                confidences = probs[:, 0]

                for j in range(len(batch_data)):
                    confidence = confidences[j].item()
                    results.append({
                        "triple_str": batch_data[j]["input"],
                        "is_correct": confidence >= 0.5,
                        "confidence": confidence,
                    })

        return results
