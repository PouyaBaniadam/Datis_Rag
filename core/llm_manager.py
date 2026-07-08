import os
import gc
from core.llm_api import OpenRouterLLM


class LLMManager:
    def __init__(self):
        self.current_local_model_name = None
        self.local_pipeline = None
        self.online_llm = OpenRouterLLM()

    def _load_local_model(self, model_name: str):
        try:
            import torch
            from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise Exception(
                "Offline packages (torch, transformers) are not installed yet. Please use the 'online' model for now.")

        if self.current_local_model_name == model_name:
            return

        if self.local_pipeline is not None:
            del self.local_pipeline
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        model_path = os.path.join("models", model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path not found: {model_path}")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16
        )

        self.local_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=1024
        )

        self.current_local_model_name = model_name

    def generate(self, prompt: str, model_type: str) -> str:
        if model_type not in ["qwen", "gemma"]:
            return self.online_llm.generate(prompt, model=model_type)
        else:
            self._load_local_model(model_type)

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]

            formatted_prompt = self.local_pipeline.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            outputs = self.local_pipeline(formatted_prompt, return_full_text=False)
            return outputs[0]['generated_text'].strip()

    def generate_stream(self, prompt: str, model_type: str):
        if model_type not in ["qwen", "gemma"]:
            yield from self.online_llm.generate_stream(prompt, model=model_type)
        else:
            yield self.generate(prompt, model_type)