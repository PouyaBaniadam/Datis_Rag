import json
import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from core.embeddings import Embedder


class Retriever:

    def __init__(
            self,
            chunks_path: str = "Data/chunks.json",
            metadata_path: str = "Data/metadata.json",
            embeddings_path: str = "Data/embeddings.npy",
            device: Optional[str] = None,
            normalize: bool = True,
    ):
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks: List[str] = json.load(f)

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata: List[Dict] = json.load(f)

        self.embeddings: np.ndarray = np.load(embeddings_path).astype("float32")
        assert len(self.chunks) == self.embeddings.shape[0], "Chunks and embeddings count mismatch"

        if normalize:
            faiss.normalize_L2(self.embeddings)

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)

        self.embedder = Embedder(device=device)

        print(f"[Retriever] Loaded {len(self.chunks)} chunks with metadata and embeddings")

    def retrieve(
            self,
            query: str,
            embedding_type: str = "api",
            top_k: int = 5,
            abs_min_score: float = 0.30,
            rel_score_drop: float = 0.7,
            fallback_top_k: int = 3,
    ) -> List[Dict]:
        index_dim = self.index.d

        target_model = embedding_type
        if target_model == "api":
            target_model = "text-embedding-3-small"

        if target_model == "local":
            query_emb = self.embedder.embed_query_local(query).astype("float32")
        else:
            query_emb = self.embedder.embed_query_api(query, model=target_model).astype("float32")

        if query_emb.shape[0] != index_dim:
            print(f"[Retriever] Dimension mismatch: query is {query_emb.shape[0]} but FAISS expects {index_dim}.")
            print(f"[Retriever] Automatically falling back to index-compatible model...")

            if index_dim == 1536:
                query_emb = self.embedder.embed_query_api(query, model="text-embedding-3-small").astype("float32")
            elif index_dim == 384:
                query_emb = self.embedder.embed_query_local(query).astype("float32")
            elif index_dim == 3072:
                query_emb = self.embedder.embed_query_api(query, model="text-embedding-3-large").astype("float32")
            else:
                query_emb = self.embedder.embed_query_api(query, model="text-embedding-3-small").astype("float32")

        faiss.normalize_L2(query_emb.reshape(1, -1))

        scores, indices = self.index.search(query_emb.reshape(1, -1), self.chunks.__len__())
        scores, indices = scores[0], indices[0]

        max_score = scores[0] if scores.size > 0 else 0.0
        results = []

        for score, idx in zip(scores, indices):
            if score < abs_min_score:
                continue
            if score < max_score * rel_score_drop:
                continue
            results.append({
                "text": self.chunks[idx],
                "score": float(score),
                "metadata": self.metadata[idx],
            })
            if len(results) >= top_k:
                break

        if len(results) == 0:
            for i in range(min(fallback_top_k, len(indices))):
                idx = indices[i]
                results.append({
                    "text": self.chunks[idx],
                    "score": float(scores[i]),
                    "metadata": self.metadata[idx],
                })

        return results