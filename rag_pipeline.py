# File: rag_pipeline.py
# Description: Retrieval with DPR + Generation with T5 for hematologic cancer corpus

import os
from pathlib import Path
import faiss
import numpy as np
import torch
from datasets import Dataset, load_from_disk
from transformers import (
    DPRContextEncoder, DPRContextEncoderTokenizer,
    DPRQuestionEncoder, DPRQuestionEncoderTokenizer,
    T5ForConditionalGeneration, T5Tokenizer
)

# Directories
EN_CORPUS_DIR = Path("./assignment_corpus/en/mayoclinic_filtered")
EL_CORPUS_DIR = Path("./assignment_corpus/el/translated_filtered")
INDEX_DIR = Path("./faiss_index")
INDEX_DIR.mkdir(exist_ok=True)

# Model checkpoints
CTX_MODEL = 'facebook/dpr-ctx_encoder-single-nq-base'
Q_MODEL = 'facebook/dpr-question_encoder-single-nq-base'
T5_MODEL = 't5-base'

# Initialize DPR tokenizers & encoders
dpr_ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(CTX_MODEL)
dpr_ctx_encoder = DPRContextEncoder.from_pretrained(CTX_MODEL)
dpr_q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(Q_MODEL)
dpr_q_encoder = DPRQuestionEncoder.from_pretrained(Q_MODEL)

# Initialize T5 for generation
t5_tokenizer = T5Tokenizer.from_pretrained(T5_MODEL)
t5_model = T5ForConditionalGeneration.from_pretrained(T5_MODEL)

def build_faiss(texts, prefix: str):
    """Encode texts, build FAISS index, save index and dataset."""
    enc = dpr_ctx_tokenizer(texts, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
    with torch.no_grad():
        out = dpr_ctx_encoder(**enc)
    embeds = out.pooler_output.cpu().numpy()
    embeds = np.ascontiguousarray(embeds)
    dim = embeds.shape[1]

    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeds)
    index.add(embeds)
    idx_path = INDEX_DIR / f"{prefix}.index"
    faiss.write_index(index, str(idx_path))

    titles = [f"{prefix}_{i}" for i in range(len(texts))]
    ds = Dataset.from_dict({"title": titles, "text": texts, "embeddings": embeds.tolist()})
    ds_path = INDEX_DIR / f"{prefix}_dataset"
    ds.save_to_disk(str(ds_path))

    return idx_path, ds_path

def retrieve(question: str, ds_path: str, idx_path: Path, k: int = 5) -> list:
    """Retrieve top-k texts for question."""
    ds = load_from_disk(str(ds_path))
    q_enc = dpr_q_tokenizer(question, return_tensors='pt', truncation=True, max_length=64)
    with torch.no_grad():
        q_out = dpr_q_encoder(**q_enc)
    q_emb = q_out.pooler_output.cpu().numpy()
    index = faiss.read_index(str(idx_path))
    faiss.normalize_L2(q_emb)
    _, ids = index.search(q_emb, k)
    return [ds[int(i)]['text'] for i in ids[0]]

def answer(question: str, ds_path: str, idx_path: Path) -> str:
    """Retrieve contexts and generate answer with T5."""
    contexts = retrieve(question, ds_path, idx_path)
    if not contexts:
        return "Δεν βρέθηκαν σχετικά έγγραφα."
    prompt = f"question: {question}  context: {'  '.join(contexts)}"
    inputs = t5_tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        outputs = t5_model.generate(
            **inputs,
            max_length=200,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
            repetition_penalty=2.0
        )
    return t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    # Build for English
    en_texts = [p.read_text(encoding='utf-8') for p in EN_CORPUS_DIR.glob('*.txt')]
    en_idx, en_ds = build_faiss(en_texts, 'en')
    # Build for Greek
    el_texts = [p.read_text(encoding='utf-8') for p in EL_CORPUS_DIR.glob('*.txt')]
    el_idx, el_ds = build_faiss(el_texts, 'el')

    print("--- English Q&A ---")
    for q in [
        "What is the treatment for leukemia?",
        "How is lymphoma diagnosed?",
        "What are the symptoms of myeloma?",
        "Explain stem cell transplantation.",
        "What is CAR-T therapy?"
    ]:
        print(f"Q: {q}")
        print(f"A: {answer(q, en_ds, en_idx)}\n")

    print("--- Greek Q&A ---")
    for q in ["Τι είναι οξεία μυελογενής λευχαιμία;", "Ποια είναι τα συμπτώματα του μυελώματος;"]:
        print(f"Q: {q}")
        print(f"A: {answer(q, el_ds, el_idx)}\n")

if __name__ == "__main__":
    main()
