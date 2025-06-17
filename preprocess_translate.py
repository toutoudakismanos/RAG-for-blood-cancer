# Description: Filtering English medical texts for hematologic cancer topics and translating to Greek

import os
import re
import glob
from pathlib import Path
from typing import List

from transformers import MarianMTModel, MarianTokenizer

# Paths
RAW_EN_DIR = Path("./assignment_corpus/en/mayoclinic")
EN_FILTERED_DIR = Path("./assignment_corpus/en/mayoclinic_filtered")
EL_TRANSLATED_DIR = Path("./assignment_corpus/el/translated")
EL_FILTERED_DIR = Path("./assignment_corpus/el/translated_filtered")
for d in [EN_FILTERED_DIR, EL_TRANSLATED_DIR, EL_FILTERED_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# English filter terms (regex patterns)
EN_TERMS = [
    r"hematologic?", r"blood cancer", r"hematological neoplasm", r"leukemia", r"lymphoma",
    r"myeloma", r"acute myeloid", r"chronic lymphocytic", r"CLL", r"acute lymphoblastic",
    r"chronic myelogenous", r"Hodgkin", r"non[- ]?Hodgkin", r"diffuse large B[- ]cell",
    r"myelodysplastic", r"myeloproliferative", r"monoclonal gammopathy", r"Waldenström",
    r"CAR[- ]NK", r"CAR[- ]T", r"stem cell transplant", r"minimal residual disease",
    r"hematopoiesis", r"bone marrow"
]
EN_PATTERN = re.compile(r"(" + r"|".join(EN_TERMS) + r")", re.IGNORECASE)

# Greek filter terms (regex patterns)
GR_TERMS = [
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
GR_PATTERN = re.compile(r"(" + r"|".join(GR_TERMS) + r")", re.IGNORECASE)

# Initialize translation model (MarianMT en->el)
MODEL_NAME = "Helsinki-NLP/opus-mt-en-el"
tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
model = MarianMTModel.from_pretrained(MODEL_NAME)


def filter_text(text: str, pattern: re.Pattern) -> bool:
    """Return True if text matches any regex term."""
    return bool(pattern.search(text))


def translate_text(text: str) -> str:
    """Translate English text block to Greek using MarianMT."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)


def process_documents():
    # Step 1: filter English raw docs and save
    for file in glob.glob(str(RAW_EN_DIR / "*.txt")):
        path = Path(file)
        text = path.read_text(encoding="utf-8")
        if filter_text(text, EN_PATTERN):
            out_en = EN_FILTERED_DIR / path.name
            out_en.write_text(text, encoding="utf-8")
            print(f"Saved filtered English: {out_en}")

    # Step 2: translate filtered English -> Greek and save
    for file in glob.glob(str(EN_FILTERED_DIR / "*.txt")):
        path = Path(file)
        english = path.read_text(encoding="utf-8")
        greek = translate_text(english)
        out_el = EL_TRANSLATED_DIR / path.name.replace('.txt', '_el.txt')
        out_el.write_text(greek, encoding="utf-8")
        print(f"Saved translated Greek: {out_el}")

    # Step 3: filter translated Greek and save
    for file in glob.glob(str(EL_TRANSLATED_DIR / "*.txt")):
        path = Path(file)
        greek = path.read_text(encoding="utf-8")
        if filter_text(greek, GR_PATTERN):
            out_el_filt = EL_FILTERED_DIR / path.name
            out_el_filt.write_text(greek, encoding="utf-8")
            print(f"Saved filtered Greek: {out_el_filt}")

if __name__ == "__main__":
    process_documents()