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
        device: str = "",
        dtype: str = "fp16",
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
        Judges a list of triples in batches using model.generate(),
        aligned with test_finetuned_llm.py.

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

                generate_ids = self.model.generate(
                    inputs_embeds=input_embeds,
                    attention_mask=full_attention_mask,
                    max_new_tokens=16,
                )

                contexts = self.tokenizer.batch_decode(
                    input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                responses = self.tokenizer.batch_decode(
                    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )

                for j in range(len(batch_data)):
                    response = responses[j].replace(contexts[j], "").strip()
                    results.append({
                        "triple_str": batch_data[j]["input"],
                        "is_correct": "True" in response,
                    })

        return results
