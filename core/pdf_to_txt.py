import fitz
import re
import json
from pathlib import Path


class DockumentProcessor:
    def __init__(self, pdf_path:Path,output_path:Path):
        self.pdf_path = pdf_path
        self.output_path = output_path

    def clean_text(text: str) -> str:
        """
        Clean PDF text:
        - Remove bullets, numbering, stray symbols
        - Merge broken lines into paragraphs
        - Keep Persian punctuation
        """

        if not text:
            return ""

        # Remove bullets and special symbols
        text = re.sub(r"[•▪◦★✦]+", " ", text)

        # Remove numbering like 1. 1) 1: etc.
        text = re.sub(r"\b\d+[\.\،\:\)\-]\s*", " ", text)

        # Remove isolated single digits or stray symbols on lines
        text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^[\(\)\[\]]+\s*$", "", text, flags=re.MULTILINE)

        # Merge lines not ending with sentence-ending punctuation
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        merged = []
        buffer = ""
        for line in lines:
            if buffer:
                if not re.search(r"[.!؟]", buffer[-1]):
                    buffer += " " + line
                else:
                    merged.append(buffer)
                    buffer = line
            else:
                buffer = line
        if buffer:
            merged.append(buffer)

        # Normalize spaces
        merged = [re.sub(r"\s+", " ", l) for l in merged]
        return "\n".join(merged)


    def pdf_to_json(self):
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"{self.pdf_path} not found")

        doc = fitz.open(self.pdf_path)
        pages_data = []

        for page_number, page in enumerate(doc, start=1):
            text = page.get_text()
            if not text.strip():
                continue
            cleaned = DockumentProcessor.clean_text(text)
            pages_data.append({
                "page_number": page_number,
                "text": cleaned
            })

        doc.close()

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(pages_data, f, ensure_ascii=False, indent=2)

        print(f"[Done] Saved cleaned text with {len(pages_data)} pages to {self.output_path}")
