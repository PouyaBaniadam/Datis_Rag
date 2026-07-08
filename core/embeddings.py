import numpy as np
from typing import List
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv('OPENAI_API_KEY')

class Embedder:
    def __init__(
        self,
        embedder_type: str = 'api',
        model_name: str = "intfloat/multilingual-e5-small",
        device: str = 'cpu',
        model_name_api: str = 'text-embedding-3-small',
        base_url: str = 'https://api.gapgpt.app/v1',
        api_key: str = API_KEY
    ):
        self.embedder_type = embedder_type
        if self.embedder_type != 'api':
            from sentence_transformers import SentenceTransformer
            self.device = device
            self.model = SentenceTransformer(model_name, device=device)
            print(f"[Embedder] Model loaded on {self.device}")

        self.model_name = model_name_api
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )

    def embed_text_api(self, texts: List[str], model: str = None) -> np.ndarray:
        target_model = model if model else self.model_name
        prefixed_texts = [f"passage: {t}" for t in texts]
        response = self.client.embeddings.create(
            model=target_model,
            input=prefixed_texts
        )
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings)

    def embed_query_api(self, query: str, model: str = None) -> np.ndarray:
        target_model = model if model else self.model_name
        prefixed_query = f"query: {query}"
        response = self.client.embeddings.create(
            model=target_model,
            input=[prefixed_query]
        )
        embedding = response.data[0].embedding
        return np.array(embedding)