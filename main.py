import os
import re
import json
from pathlib import Path
from tqdm.auto import tqdm

# NLP and embedding libraries
from transformers import MarianMTModel, MarianTokenizer, AutoTokenizer, AutoModel
import torch
import faiss

# --------------------
# Configuration
# --------------------
INPUT_DIR = Path('./assignment_corpus/en/mayoclinic')
TRANSLATED_DIR = Path('./assignment_corpus/el/translated')
FILTERED_EN_DIR = Path('./assignment_corpus/en/filtered')
FILTERED_EL_DIR = Path('./assignment_corpus/el/filtered')
VECTOR_DB_PATH = Path('./vector_db/faiss_index.bin')
EMBEDDINGS_PATH = Path('./vector_db/embeddings.npy')
DOC_METRICS_PATH = Path('./vector_db/doc_metadata.json')

# Ensure directories exist
for d in [TRANSLATED_DIR, FILTERED_EN_DIR, FILTERED_EL_DIR, VECTOR_DB_PATH.parent]:
    d.mkdir(parents=True, exist_ok=True)

# --------------------
# Term lists and regex patterns
# --------------------
EN_TERMS = [
    "hematologic", "blood cancer", "hematological neoplasm", "leukemia", "lymphoma", "myeloma",
    "acute myeloid", "chronic lymphocytic", "CLL", "acute lymphoblastic", "chronic myelogenous",
    "Hodgkin", "nonHodgkin", "diffuse large B-cell", "myelodysplastic", "myeloproliferative",
    "monoclonal gammopathy", "Waldenström", "CAR-NK", "CAR-T", "stem cell transplant",
    "minimal residual disease", "hematopoiesis", "bone marrow"
]
EL_TERMS = [
    r"αιματολογικ[αοςη]* καρκίνος", r"καρκίνος του αίματος", r"λευχαιμία", r"λέμφωμα", r"μυέλωμα",
    r"αιμοποίηση", r"μυελός των οστών", r"βλαστοκύτταρα", r"πλασματοκύτταρα", r"Hodgkin",
    r"λέμφωμα|λεμφικός|λεμφικά κύτταρα|λεμφαδένες|λεμφαδενικός|λεμφαδενίτιδα|λεμφική κακοήθεια",
    r"λευχαιμία|λευκά αιμοσφαίρια|λευκοκύτταρα|αιματολογική κακοήθεια|διαταραχές του μυελού των οστών",
    r"μυέλωμα|πολλαπλούν μυέλωμα|νεοπλασία πλασματοκυττάρων",
    r"μυελός|αιμοποιητικό σύστημα|αιμοποιητικός ιστός",
    r"οξεία μυελογενής λευχαιμία|AML|Χρόνια λεμφοκυτταρική λευχαιμία|CLL",
    r"οξεία λεμφοβλαστική λευχαιμία|χρόνια μυελογενής λευχαιμία|CML",
    r"διάχυτο μεγαλοκυτταρικό λέμφωμα Β|DLBCL",
    r"μυελοδυσπλαστικά σύνδρομα|μυελοϋπερπλαστικά νοσήματα|μονοκλωνική γαμμαπάθεια|μακροσφαιριναιμία Waldenström",
    r"CAR-NK|CAR-T|μεταμόσχευση αιμοποιητικών βλαστοκυττάρων|ελάχιστη υπολειπόμενη νόσος"
]
# Compile regex
EN_PATTERN = re.compile(r"(" + r"|".join([re.escape(t) for t in EN_TERMS]) + r")", re.IGNORECASE)
EL_PATTERN = re.compile(r"(" + r"|".join(EL_TERMS) + r")", re.IGNORECASE)

# --------------------
# 1. Data Filtering & Preprocessing
# --------------------

def filter_documents(input_dir: Path, output_dir: Path, pattern: re.Pattern):
    """
    Read all text files from input_dir, filter paragraphs containing any term in pattern,
    and save filtered texts to output_dir preserving filenames.
    """
    for txt_file in input_dir.rglob('*.txt'):
        with open(txt_file, 'r', encoding='utf-8') as f:
            content = f.read()
        matches = pattern.findall(content)
        if matches:
            out_path = output_dir / txt_file.name
            with open(out_path, 'w', encoding='utf-8') as out:
                out.write(content)

# Filter English source
filter_documents(INPUT_DIR, FILTERED_EN_DIR, EN_PATTERN)

# --------------------
# 2. Translation (en -> el)
# --------------------

tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-el')
model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-el')

def translate_file(in_path: Path, out_path: Path):
    """
    Translate an English text file to Greek and save.
    """
    with open(in_path, 'r', encoding='utf-8') as f:
        src = f.read().splitlines()
    batch = tokenizer.prepare_seq2seq_batch(src, return_tensors='pt')
    translated = model.generate(**batch)
    tgt = tokenizer.batch_decode(translated, skip_special_tokens=True)
    with open(out_path, 'w', encoding='utf-8') as out:
        out.write("\n".join(tgt))

# Translate filtered English docs
for en_file in tqdm(list(FILTERED_EN_DIR.glob('*.txt')), desc="Translating to Greek"):
    out_file = TRANSLATED_DIR / en_file.name
    translate_file(en_file, out_file)

# --------------------
# 3. RAG Pipeline
# --------------------

# 3.1 Embed documents and build FAISS
embed_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
embed_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')


def embed_texts(texts):
    """Compute sentence embeddings for a list of texts"""
    inputs = embed_tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = embed_model(**inputs)
    # mean pooling
    embeddings = model_output.last_hidden_state.mean(dim=1)
    return embeddings.cpu().numpy()

# Collect docs
docs = []  # list of (id, text, lang)
for lang_dir, lang in [(FILTERED_EN_DIR, 'en'), (TRANSLATED_DIR, 'el')]:
    for f in lang_dir.glob('*.txt'):
        text = f.read_text(encoding='utf-8')
        docs.append({'id': f.name, 'text': text, 'lang': lang})

# Embed and index
texts = [d['text'] for d in docs]
embs = embed_texts(texts)

# Save embeddings
import numpy as np
np.save(EMBEDDINGS_PATH, embs)
with open(DOC_METRICS_PATH, 'w', encoding='utf-8') as jm:
    json.dump(docs, jm, ensure_ascii=False, indent=2)

# Build FAISS index
dim = embs.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embs)
faiss.write_index(index, str(VECTOR_DB_PATH))

# --------------------
# 4. RAG Query Pipeline
# --------------------

def retrieve(query: str, top_k: int = 5):
    """Return top_k docs for a query"""
    q_emb = embed_texts([query])[0]
    I, D = index.search(np.array([q_emb]), top_k)
    results = []
    for idx in I[0]:
        results.append(docs[idx])
    return results

# Example usage
def answer_question(question: str):
    """Retrieve and generate an answer using retrieved docs"""
    results = retrieve(question)
    # For simplicity, concatenate top docs as context
    context = "\n\n".join([r['text'] for r in results])
    # Here you could plug in a generator model, e.g., T5 or GPT
    # For brevity, we return the context as "answer"
    return context

if __name__ == '__main__':
    # Example: test retrieval
    for q in [
        "What is acute myeloid leukemia?",
        "Τι είναι η οξεία μυελογενής λευχαιμία;"
    ]:
        print("Q:", q)
        print("A:", answer_question(q)[:500], "...\n")
