"""
Script για φιλτράρισμα αγγλικών κειμένων σχετικά με αιματολογικούς καρκίνους
και μετάφραση στα ελληνικά, με chunking μεγάλων κειμένων.
"""
import os
import re
from pathlib import Path
from transformers import MarianMTModel, MarianTokenizer

# Ρυθμίσεις φακέλων
INPUT_DIR = Path('./assignment_corpus/en/mayoclinic')
FILTERED_DIR = Path('./assignment_corpus/en/filtered')
TRANSLATED_DIR = Path('./assignment_corpus/en/translated')

# Λίστα όρων για φιλτράρισμα (regex, case-insensitive)
EN_TERMS = [
    r"hematologic?", r"blood cancer", r"hematological neoplasm", r"leukemia", r"lymphoma",
    r"myeloma", r"acute myeloid", r"chronic lymphocytic|CLL", r"acute lymphoblastic",
    r"chronic myelogenous|CML", r"Hodgkin", r"non[- ]?Hodgkin", r"diffuse large B-cell",
    r"myelodysplastic", r"myeloproliferative", r"monoclonal gammopathy", r"Waldenström",
    r"CAR[- ]NK|CAR[- ]T", r"stem cell transplant", r"minimal residual disease",
    r"hematopoiesis", r"bone marrow"
]
FILTER_PATTERN = re.compile('|'.join(EN_TERMS), flags=re.IGNORECASE)

# Chunking  : max tokens per chunk (~500 tokens ~ 2000 chars)
MAX_CHARS = 2000  

# Προφόρτωση MarianMT για en->el
MODEL_NAME = 'Helsinki-NLP/opus-mt-en-el'
tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
model = MarianMTModel.from_pretrained(MODEL_NAME)

def chunk_text(text, max_chars=MAX_CHARS):
    """
    Κόβει κείμενο σε chunks έως max_chars χωρίς να σπάει λέξη.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text) and not text[end].isspace():
            end = text.rfind(' ', start, end)
            if end < start:
                end = min(start + max_chars, len(text))
        chunks.append(text[start:end].strip())
        start = end
    return chunks


def filter_and_translate_file(in_path: Path, out_filtered: Path, out_translated: Path):
    text = in_path.read_text(encoding='utf-8')
    # Απλό clean-up
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Φιλτράρισμα
    if not FILTER_PATTERN.search(text):
        # Δεν βρέθηκε όρος, skip
        return False

    # Αποθήκευση φιλτραρισμένου
    out_filtered.write_text(text, encoding='utf-8')

    # Μετάφραση με chunking
    chunks = chunk_text(text)
    translated_chunks = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors='pt', truncation=True, max_length=512)
        translated = model.generate(**inputs)
        tgt = tokenizer.decode(translated[0], skip_special_tokens=True)
        translated_chunks.append(tgt)
    full_translation = "\n\n".join(translated_chunks)

    # Αποθήκευση μετάφρασης
    out_translated.write_text(full_translation, encoding='utf-8')
    return True


def main():
    FILTERED_DIR.mkdir(parents=True, exist_ok=True)
    TRANSLATED_DIR.mkdir(parents=True, exist_ok=True)

    for file in INPUT_DIR.glob('*.txt'):
        filtered_path = FILTERED_DIR / file.name
        translated_path = TRANSLATED_DIR / file.name
        ok = filter_and_translate_file(file, filtered_path, translated_path)
        status = 'Processed' if ok else 'Skipped (no terms)'
        print(f"{file.name}: {status}")

if __name__ == '__main__':
    main()
