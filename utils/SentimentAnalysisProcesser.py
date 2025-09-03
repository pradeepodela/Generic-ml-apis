
from __future__ import annotations

import json
import logging
import os
import sys
from functools import lru_cache
from typing import Any, Dict, List
import time 
import numpy as np
import torch
from scipy.special import softmax
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

logger = logging.getLogger("sentiment_service")
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# Model name can be overridden via environment variable so you can AB‑test
# or roll back without touching the code.
MODEL_NAME = os.getenv("SENTIMENT_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest")


def preprocess(text: str) -> str:
    """Apply the light Twitter‑style preprocessing used by CardiffNLP.

    * User mentions -> @user
    * Links          -> http
    """
    return " ".join(
        "@user" if tok.startswith("@") and len(tok) > 1 else "http" if tok.startswith("http") else tok
        for tok in text.split()
    )


@lru_cache(maxsize=1)
def _load_resources():  # noqa: D401  ‑ do not need docstring style check here
    """Singleton that loads and returns tokenizer, config, model, device."""
    logger.info("Loading model resources for '%s'…", MODEL_NAME)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    config = AutoConfig.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    logger.info("Model loaded on %s", device)
    return tokenizer, config, model, device


def _predict(texts: List[str]) -> List[List[Dict[str, Any]]]:
   
    """Batch‑predict sentiment probabilities for *raw* texts (no prep)."""
    tokenizer, config, model, device = _load_resources()

    cleaned = [preprocess(t) for t in texts]
    encoded = tokenizer(cleaned, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        logits = model(**encoded).logits

    probs = softmax(logits.cpu().numpy(), axis=1)
    ranks = np.flip(np.argsort(probs, axis=1), axis=1)

    outputs: List[List[Dict[str, Any]]] = []
    for i in range(len(texts)):
        sample: List[Dict[str, Any]] = []
        for r, idx in enumerate(ranks[i]):
            sample.append({
                "rank": r + 1,
                "label": config.id2label[idx],
                "score": round(float(probs[i, idx]), 4),
            })
        outputs.append(sample)
    end_time = time.time()
    # logger.info("Processed %d texts in %.4f seconds (%.4f per text)", len(texts), end_time - start_time, (end_time - start_time) / len(texts))
    return outputs


def analyze(text: str) -> str:
    """Public helper for single‑string inference, returns formatted JSON."""
    start_time = time.time()
    response = _predict([text])
    outputs = response[0]
    end_time = time.time()

    # Create a dictionary with all the results
    result_data = {
        "sentiment_scores": outputs,
        "total_processing_time": round(end_time - start_time, 4)
    }

    # Convert the final dictionary to a JSON string
    return json.dumps(result_data, indent=4)

# ---------------------------------------------------------------------------
# CLI helper — useful in Docker health‑checks and quick experiments
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    text = "Covid cases are increasing fast!"
    print(analyze(text))
    