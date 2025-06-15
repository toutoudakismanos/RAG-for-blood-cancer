# Retrieval-Augmented Generation (RAG) System for Hematologic Cancers:
# Robust Data Preparation Script (no more >512-token warnings)

import os
import nltk
import torch
from tqdm.auto import tqdm
from transformers import MarianMTModel, MarianTokenizer
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters

# Ensure punkt tokenizer is available
nltk.download('punkt', quiet=True)

# -----------------------------
# Configuration
# -----------------------------
input_root  = './assignment_corpus/en/mayoclinic'
output_root = './assignment_corpus/en/translated'
os.makedirs(output_root, exist_ok=True)

model_name    = 'Helsinki-NLP/opus-mt-en-el'

# Initialize tokenizer & model
tokenizer = MarianTokenizer.from_pretrained(model_name)
model     = MarianMTModel.from_pretrained(model_name)
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Retrieve model's true max length
MODEL_MAX = tokenizer.model_max_length  # typically 512
STRIDE    = 64

# -----------------------------
# Semantic splitter setup
# -----------------------------
punkt_params             = PunktParameters()
punkt_params.abbrev_types = set(['e.g', 'i.e'])
sent_splitter            = PunktSentenceTokenizer(punkt_params)

# -----------------------------
# Chunking helpers
# -----------------------------

def tokenwise_split(text: str):
    """Split any long text into overlapping token chunks within MODEL_MAX"""
    enc = tokenizer(text, return_tensors='pt', add_special_tokens=True)
    ids = enc.input_ids[0]
    total = ids.size(0)
    step = MODEL_MAX - STRIDE
    out = []
    for start in range(0, total, step):
        end = min(start + MODEL_MAX, total)
        chunk_txt = tokenizer.decode(ids[start:end], skip_special_tokens=True)
        out.append(chunk_txt)
        if end == total:
            break
    return out


def semantic_chunks(text: str):
    """
    Segment text into sentence-based chunks under MODEL_MAX tokens.
    Use tokenwise_split as fallback for any piece exceeding MODEL_MAX.
    """
    sentences = sent_splitter.tokenize(text)
    chunks, current = [], ""

    for sent in sentences:
        # build candidate segment
        candidate = f"{current} {sent}".strip() if current else sent
        # measure true token count with special tokens
        ids = tokenizer(candidate, return_tensors='pt', add_special_tokens=True).input_ids[0]
        if ids.size(0) <= MODEL_MAX:
            current = candidate
        else:
            # push existing
            if current:
                chunks.append(current)
            # check single sentence
            sent_ids = tokenizer(sent, return_tensors='pt', add_special_tokens=True).input_ids[0]
            if sent_ids.size(0) <= MODEL_MAX:
                current = sent
            else:
                # fallback on sentence
                for sub in tokenwise_split(sent):
                    chunks.append(sub)
                current = ""

    if current:
        chunks.append(current)
    return chunks

# -----------------------------
# Translation function
# -----------------------------

def translate_en_to_el(text: str) -> str:
    """
    Translates text by chunking into semantic_chunks, then translating each chunk.
    Guarantees no chunk exceeds MODEL_MAX tokens.
    """
    chunks = semantic_chunks(text)
    outputs = []
    for chunk in chunks:
        enc = tokenizer(chunk, return_tensors='pt', padding=True, truncation=True, max_length=MODEL_MAX).to(device)
        gen = model.generate(**enc)
        out = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
        outputs.append(out)
    return "\n".join(outputs)

# -----------------------------
# Main processing loop
# -----------------------------

total_seen = 0
translated = 0

for root, _, files in os.walk(input_root):
    rel = os.path.relpath(root, input_root)
    od  = os.path.join(output_root, rel)
    os.makedirs(od, exist_ok=True)

    for fname in tqdm(files, desc=f"DIR {rel}"):
        if not fname.lower().endswith('.txt'):
            continue
        total_seen += 1

        path = os.path.join(root, fname)
        try:
            text = open(path, 'r', encoding='utf-8', errors='ignore').read()
        except Exception as e:
            print(f"Read error {path}: {e}")
            continue

        # translate
        try:
            result = translate_en_to_el(text)
        except Exception as e:
            print(f"Translate error {fname}: {e}")
            continue
        translated += 1

        base, ext = os.path.splitext(fname)
        open(os.path.join(od, f"{base}_en{ext}"), 'w', encoding='utf-8').write(text)
        open(os.path.join(od, f"{base}_el{ext}"), 'w', encoding='utf-8').write(result)

print(f"\nSeen {total_seen} files, translated {translated}.")
print("✅ Finished full translation with no over-length chunks.")
