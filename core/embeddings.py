import numpy as np
from typing import List
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv('OPENAI_API_KEY')
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
        self.device = device
        self.model_name = model_name_api
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )

    def _load_local_model(self):
        if not hasattr(self, 'model'):
            from sentence_transformers import SentenceTransformer
            local_model_path = os.path.join(BASE_DIR, "models", "multilingual-e5-small")
            self.model = SentenceTransformer(local_model_path, device=self.device)

    def embed_text_api(self, texts: List[str], model: str = None, api_key: str = None) -> np.ndarray:
        target_model = model if model else self.model_name
        client = OpenAI(api_key=api_key, base_url=self.client.base_url) if api_key else self.client
        prefixed_texts = [f"passage: {t}" for t in texts]
        response = client.embeddings.create(
            model=target_model,
            input=prefixed_texts
        )
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings)

    def embed_query_api(self, query: str, model: str = None, api_key: str = None) -> np.ndarray:
        target_model = model if model else self.model_name
        client = OpenAI(api_key=api_key, base_url=self.client.base_url) if api_key else self.client
        prefixed_query = f"query: {query}"
        response = client.embeddings.create(
            model=target_model,
            input=[prefixed_query]
        )
        embedding = response.data[0].embedding
        return np.array(embedding)

    def embed_text_local(self, texts: List[str]) -> np.ndarray:
        self._load_local_model()
        prefixed_texts = [f"passage: {t}" for t in texts]
        embeddings = self.model.encode(prefixed_texts, normalize_embeddings=True)
        return np.array(embeddings)

    def embed_query_local(self, query: str) -> np.ndarray:
        self._load_local_model()
        prefixed_query = f"query: {query}"
        embedding = self.model.encode([prefixed_query], normalize_embeddings=True)[0]
        return np.array(embedding)