import os
import gc
from core.llm_api import OpenRouterLLM

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class LLMManager:
    def __init__(self):
        self.current_local_model_name = None
        self.local_llm = None
        self.online_llm = OpenRouterLLM()

    def _load_local_model(self, model_type: str):
        try:
            from llama_cpp import Llama
        except ImportError:
            raise Exception("Please install llama-cpp-python: pip install llama-cpp-python")

        if self.current_local_model_name == model_type and self.local_llm is not None:
            return

        if self.local_llm is not None:
            del self.local_llm
            gc.collect()

        if "gemma" in model_type.lower():
            model_path = os.path.join(BASE_DIR, "models", "gemma_4_e4b", "gemma-4-E4B_q4_0-it.gguf")
        elif "mistral" in model_type.lower():
            model_path = os.path.join(BASE_DIR, "models", "mistral_7b_instruct_v0.1", "mistral-7b-v0.1.Q4_0.gguf")
        else:
            raise FileNotFoundError(f"Local model path configuration not found for: {model_type}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"GGUF model file not found at: {model_path}")

        self.local_llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=4,
            verbose=False
        )
        self.current_local_model_name = model_type

    def _format_prompt(self, prompt: str, model_type: str) -> str:
        system_prompt = "شما یک دستیار هوشمند و دقیق به زبان فارسی هستید."
        if "gemma" in model_type.lower():
            return f"<start_of_turn>user\n{system_prompt}\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        else:
            return f"<s>[INST] {system_prompt}\n{prompt} [/INST]"

    def generate(self, prompt: str, model_type: str, api_key: str = None) -> str:
        if "gemma" not in model_type.lower() and "mistral" not in model_type.lower():
            return self.online_llm.generate(prompt, model=model_type, api_key=api_key)
        else:
            self._load_local_model(model_type)
            formatted_prompt = self._format_prompt(prompt, model_type)
            output = self.local_llm(
                formatted_prompt,
                max_tokens=1024,
                stop=["<end_of_turn>", "</s>"],
                temperature=0.2
            )
            return output["choices"][0]["text"].strip()

    def generate_stream(self, prompt: str, model_type: str, api_key: str = None):
        if "gemma" not in model_type.lower() and "mistral" not in model_type.lower():
            yield from self.online_llm.generate_stream(prompt, model=model_type, api_key=api_key)
        else:
            self._load_local_model(model_type)
            formatted_prompt = self._format_prompt(prompt, model_type)
            stream = self.local_llm(
                formatted_prompt,
                max_tokens=1024,
                stop=["<end_of_turn>", "</s>"],
                temperature=0.2,
                stream=True
            )
            for chunk in stream:
                token = chunk["choices"][0]["text"]
                if token:
                    yield token