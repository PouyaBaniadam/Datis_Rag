import os
from core.embeddings import Embedder
from core.pdf_to_txt import DockumentProcessor
from pathlib import Path
import numpy as np
import json
from typing import List, Dict
from hazm import sent_tokenize

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC_PATH = Path(os.path.join(BASE_DIR, "Data"))
CHUNKS_PATH = DOC_PATH / "chunks.json"
METADATA_PATH = DOC_PATH / "metadata.json"
EMBEDDINGS_PATH = DOC_PATH / "embeddings.npy"


class EmbeddingBuilder:

    def __init__(self, embedding_type: str = "text-embedding-3-small", chunk_size: int = 700, overlap_sentences: int = 2, batch_size: int = 8):
        self.chunk_size = chunk_size
        self.overlap_sentences = overlap_sentences
        self.batch_size = batch_size
        self.embedder = Embedder(
            model_name_api=embedding_type
        )
        self.embedding_type = embedding_type

    def chunk_doc(self, pages: List[Dict], doc_id):
        chunks = []
        metadata = []
        doc_char_offset = 0

        for page in pages:
            page_number = page["page_number"]
            sentences = sent_tokenize(page['text'])
            current_chunk = []
            char_start = doc_char_offset

            for _, sent in enumerate(sentences):
                current_chunk.append(sent)
                current_len = sum(len(s) for s in current_chunk) + len(current_chunk)

                if current_len >= self.chunk_size:
                    chunk_text = " ".join(current_chunk)
                    chunks.append(chunk_text)

                    metadata.append({
                        "doc_id": doc_id,
                        "char_start": char_start,
                        "char_end": char_start + len(chunk_text),
                        "sentences": current_chunk.copy(),
                        "page_number": page_number
                    })

                    overlap_sents = current_chunk[-self.overlap_sentences:] if self.overlap_sentences > 0 else []
                    char_start += len(
                        " ".join(current_chunk[:-self.overlap_sentences])) if self.overlap_sentences > 0 else len(
                        chunk_text)
                    current_chunk = overlap_sents

            if current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)
                metadata.append({
                    "doc_id": doc_id,
                    "char_start": char_start,
                    "char_end": char_start + len(chunk_text),
                    "sentences": current_chunk.copy(),
                    "page_number": page_number
                })

            doc_char_offset += len(page['text']) + 1

        return chunks, metadata

    def embed_chunks(self, chunks: List[str]) -> np.ndarray:
        all_embeddings = []
        total_chunks = len(chunks)
        print(f"[Build] Total chunks to embed: {total_chunks}")

        for i in range(0, total_chunks, self.batch_size):
            batch = chunks[i:i + self.batch_size]
            current_end = min(i + self.batch_size, total_chunks)
            print(f"[Build] Sending batch {i // self.batch_size + 1}: processing chunks {i} to {current_end}...")

            if self.embedding_type == "local":
                emb = self.embedder.embed_text_local(batch)
            else:
                emb = self.embedder.embed_text_api(batch, model=self.embedding_type)
            all_embeddings.append(emb)

        print("[Build] All embeddings received successfully.")
        return np.vstack(all_embeddings)

    def build(self):
        print("[Building] Starting pipeline...")
        all_chunks = []
        all_metadata = []
        all_embeddings = []

        pdf_files = list(DOC_PATH.glob("*.pdf"))
        if not pdf_files:
            print("[Build] Warning: No PDF files found in Data/ directory.")
            return

        for pdf_path in pdf_files:
            print(f"[Build] Processing PDF: {pdf_path.name}")
            doc_id = pdf_path.name

            processed_path = DOC_PATH / f"processed_{doc_id}.json"

            self.pdf_processor = DockumentProcessor(
                pdf_path=pdf_path,
                output_path=processed_path
            )
            self.pdf_processor.pdf_to_json()

            pages = json.loads(processed_path.read_text(encoding="utf-8"))

            chunks, metadata = self.chunk_doc(pages, doc_id)
            embeddings = self.embed_chunks(chunks)

            all_chunks.extend(chunks)
            all_metadata.extend(metadata)
            all_embeddings.append(embeddings)

        np.save(EMBEDDINGS_PATH, np.vstack(all_embeddings))

        with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, ensure_ascii=False)

        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(all_metadata, f, ensure_ascii=False)

        print("[Build] Database build process finished successfully.")