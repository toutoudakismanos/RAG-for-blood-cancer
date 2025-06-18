# rag_model.py
"""
Υλοποίηση pipeline RAG με FAISS + Open-Source generator
"""
import os
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Ρυθμίσεις
EN_FILTERED = Path('./assignment_corpus/en/filtered')
EL_FILTERED = Path('./assignment_corpus/el/filtered')  # δημιουργείται από τα επόμενα βήματα
EMB_MODEL_EN = 'all-mpnet-base-v2'
EMB_MODEL_MULTI = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
GEN_MODEL = 't5-base'

class MedicalRAG:
    def __init__(self, emb_model_name, gen_model_name, index_path=None):
        # Φόρτωση embedding
        self.embedder = SentenceTransformer(emb_model_name)
        # Φόρτωση generator
        self.tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
        self.generator = AutoModelForSeq2SeqLM.from_pretrained(gen_model_name)
        # FAISS index
        self.dim = self.embedder.get_sentence_embedding_dimension()
        if index_path and Path(index_path).exists():
            self.index = faiss.read_index(index_path)
        else:
            self.index = faiss.IndexFlatL2(self.dim)
            self._build_index(EN_FILTERED, emb_model_name)

    def _build_index(self, docs_folder: Path, emb_model_name: str):
        texts = []
        for f in docs_folder.glob('*.txt'):
            texts.append(f.read_text(encoding='utf-8'))
        embeddings = self.embedder.encode(texts, convert_to_numpy=True)
        self.index.add(embeddings)
        # Προαιρετικά: αποθήκευση
        faiss.write_index(self.index, 'faiss_index.idx')

    def retrieve(self, query: str, k: int = 5):
        q_emb = self.embedder.encode([query], convert_to_numpy=True)
        D, I = self.index.search(q_emb, k)
        return I[0]  # δείκτες εγγράφων

    def generate(self, query: str, docs: list):
        # Συνένωση query+docs
        input_text = query + ' \n'.join(docs)
        inputs = self.tokenizer(input_text, return_tensors='pt', truncation=True, max_length=512)
        outputs = self.generator.generate(**inputs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def answer(self, query: str, docs_folder: Path, k: int = 5):
        idxs = self.retrieve(query, k)
        docs = []
        files = list(docs_folder.glob('*.txt'))
        for i in idxs:
            docs.append(files[i].read_text(encoding='utf-8'))
        return self.generate(query, docs)

# Παράδειγμα χρήσης:
if __name__ == '__main__':
    rag = MedicalRAG(EMB_MODEL_EN, GEN_MODEL)
    q = "What are the symptoms of leukemia?"
    ans = rag.answer(q, EN_FILTERED)
    print("Q:", q)
    print("A:", ans)
