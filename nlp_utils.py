from transformers import pipeline

# Initialise la pipeline NER
ner = pipeline("ner", grouped_entities=True)

def analyze_text(text):
    return ner(text)
