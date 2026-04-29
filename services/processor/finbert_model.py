import os
from functools import lru_cache

from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
env_path = os.path.join(BASE_DIR, ".env")
load_dotenv(env_path)

FINBERT_MODEL_NAME = "ProsusAI/finbert"


@lru_cache(maxsize=1)
def get_finbert_pipeline():
    """
    Lazily load FinBERT sentiment pipeline once.
    """
    print("Loading FinBERT model... (first call will be slow)")

    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL_NAME)

    nlp = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        top_k=None,
        truncation=True,
    )

    return nlp


def normalize_outputs(outputs):
    """
    Normalize different Hugging Face pipeline output shapes into list[dict].
    """

    if isinstance(outputs, dict):
        return [outputs]

    if isinstance(outputs, list):
        if not outputs:
            return []

        first = outputs[0]

        if isinstance(first, dict):
            return outputs

        if isinstance(first, list):
            return first

    return []


def analyze_headline(headline: str):
    """
    Run FinBERT on a single headline and return (label, score).

    label:
    - bullish
    - bearish
    - neutral

    score:
    - positive_score - negative_score
    """

    nlp = get_finbert_pipeline()

    text = headline.strip() if headline else ""

    if not text:
        return "neutral", 0.0

    outputs = nlp(text)
    class_outputs = normalize_outputs(outputs)

    if not class_outputs:
        return "neutral", 0.0

    scores = {
        item.get("label", "").lower(): float(item.get("score", 0.0))
        for item in class_outputs
        if isinstance(item, dict)
    }

    pos = scores.get("positive", 0.0)
    neg = scores.get("negative", 0.0)
    neu = scores.get("neutral", 0.0)

    score = float(pos - neg)

    if pos >= neg and pos >= neu:
        label = "bullish"
    elif neg >= pos and neg >= neu:
        label = "bearish"
    else:
        label = "neutral"

    return label, score