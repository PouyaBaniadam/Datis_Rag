import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('OPENAI_API_KEY')

class OpenRouterLLM:
    def __init__(
        self,
        api_key: str = API_KEY,
        model: str = "gapgpt-qwen-3.5",
        base_url: str = "https://api.gapgpt.app/v1",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        system_prompt: str = "شما یک دستیار فارسی هستید که پاسخ‌ها را به صورت خلاصه اما دقیق ارائه می‌دهد."
    ):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=60
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt

    def generate(self, prompt: str, model: str = None, api_key: str = None) -> str:
        target_model = model if model else self.model
        client = OpenAI(api_key=api_key, base_url=self.client.base_url) if api_key else self.client
        response = client.chat.completions.create(
            model=target_model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content.strip()

    def generate_stream(self, prompt: str, model: str = None, api_key: str = None):
        target_model = model if model else self.model
        client = OpenAI(api_key=api_key, base_url=self.client.base_url) if api_key else self.client
        response = client.chat.completions.create(
            model=target_model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True
        )
        for chunk in response:
            if chunk.choices:
                content = chunk.choices[0].delta.content
                if content:
                    yield content