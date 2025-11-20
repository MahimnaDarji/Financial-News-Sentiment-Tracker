import os
from functools import lru_cache

from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Load .env (not strictly needed here, but keeps it consistent)
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
        return_all_scores=True,
    )
    return nlp


def analyze_headline(headline: str):
    """
    Run FinBERT on a single headline and return (label, score):

    - label: 'bullish' | 'bearish' | 'neutral'
    - score: float in roughly [-1, 1] = positive_score - negative_score
    """
    nlp = get_finbert_pipeline()

    # Truncate overly long text to keep model happy
    text = headline.strip()
    if not text:
        return "neutral", 0.0
    if len(text) > 512:
        text = text[:512]

    outputs = nlp(text)[0]  # list of dicts for each class

    # FinBERT labels: POSITIVE / NEGATIVE / NEUTRAL
    scores = {o["label"].lower(): o["score"] for o in outputs}

    pos = scores.get("positive", 0.0)
    neg = scores.get("negative", 0.0)
    neu = scores.get("neutral", 0.0)

    # Continuous sentiment score: positive - negative
    score = float(pos - neg)

    # Discrete label mapped to finance-style terms
    if pos >= neg and pos >= neu:
        label = "bullish"
    elif neg >= pos and neg >= neu:
        label = "bearish"
    else:
        label = "neutral"

    return label, score
