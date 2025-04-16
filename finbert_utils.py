import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
from logger import get_logger

logger = get_logger("finbert_utils")

MODEL_NAME = "yiyanghkust/finbert-tone"

try:
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    logger.info("FinBERT model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load FinBERT model: {e}")
    tokenizer = None
    model = None

def estimate_sentiment(headlines):
    """
    Estimate sentiment from a list of headlines using FinBERT.
    
    Returns:
        (sentiment_score, sentiment_label)
    """
    if not tokenizer or not model or not headlines:
        return 0.0, "neutral"

    try:
        inputs = tokenizer(headlines, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1).numpy()
        avg_probs = np.mean(probs, axis=0)
        sentiment_score = float(np.max(avg_probs))
        sentiment_label = ["negative", "neutral", "positive"][np.argmax(avg_probs)]
        return sentiment_score, sentiment_label
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        return 0.0, "neutral"
