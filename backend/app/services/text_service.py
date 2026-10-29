"""
Text Inference Service
======================
Combines:
  1. Trained sklearn pipeline (TF-IDF + LogisticRegression) stored at artifacts/model_pipeline.pkl
     with de-biasing against WELFake scraper chrome tokens and Reuters-dateline intercept calibration.
  2. Live Fact-Checking & Encyclopedic Claim Verification (Wikipedia API + Grounded Fact Checker)
     so factual falsehoods (e.g. "Kylian Mbappe won the Ballon d'Or", "Earth is flat") are
     verified against real-world facts rather than just word-frequency style.
"""
import os
import re
import threading
import logging
import joblib
import numpy as np
from typing import Optional, Dict, Any

from backend.app.services.fact_check_service import fact_check_service

logger = logging.getLogger(__name__)

# Path to the trained artifact (relative to project root)
_ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "artifacts")
MODEL_PATH = os.path.abspath(os.path.join(_ARTIFACTS_DIR, "model_pipeline.pkl"))

# Words that had artificial positive (Fake) weights in WELFake solely because
# the WELFake fake-news scraper included webpage UI chrome ("Watch Video",
# "Featured Image", "Read More", "Today's News") whereas the Reuters scraper stripped them.
_SCRAPER_ARTIFACT_WORDS = re.compile(
    r"\b(today'?s?|news\d*|videos?|watch|youtube|images?|photos?|featured|read|live|"
    r"breaking|results?|channel|subscribe|comments?|share|click|twitter|facebook|instagram|getty)\b",
    re.IGNORECASE,
)

# Explicit sensationalist / hoax / conspiracy linguistic markers
_SENSATIONAL_PATTERNS = re.compile(
    r"\b(shocking|secret|illuminati|aliens?|miracle\s+cure|mind\s+control|hoax|conspiracy|"
    r"cures?\s+all|reptilian|flat\s+earth|chemtrails|microchip|new\s+world\s+order|"
    r"big\s+pharma|doctors\s+hate|100%\s+guaranteed|anti-gravity|ufo|"
    r"faked\s+the\s+moon|deep\s+state|clones?\s+replaced|ate\s+the\s+president|"
    r"underwater\s+antennas|weather\s+machine|anonymous\s+(?:internet\s+)?whistleblower)\b",
    re.IGNORECASE,
)


class TextInferenceService:
    """
    Service for text-based NLP classification + Live Encyclopedic Fact Verification.
    """

    def __init__(self):
        self._pipeline = None
        self._lock = threading.Lock()
        self.load_model()

    def load_model(self):
        """Thread-safely loads the trained sklearn pipeline."""
        with self._lock:
            try:
                if not os.path.exists(MODEL_PATH):
                    logger.warning(
                        f"Trained model not found at {MODEL_PATH}. "
                        "Run train_engine.py --data-path <csv> to train the model."
                    )
                    self._pipeline = None
                    return
                logger.info(f"Loading trained sklearn NLP pipeline from {MODEL_PATH}...")
                self._pipeline = joblib.load(MODEL_PATH)
                logger.info("Successfully loaded trained sklearn pipeline.")
            except Exception as e:
                logger.error(f"Failed to load NLP pipeline: {e}", exc_info=True)
                self._pipeline = None

    def predict(self, text: str, is_verified_source: bool = False) -> Optional[Dict[str, Any]]:
        """
        Returns calibrated probability estimates and optional fact-check reason for the input text.
        Output: { "p_fake": float, "p_real": float, "fact_check_reason": Optional[str] }
        """
        with self._lock:
            pipeline = self._pipeline

        if not pipeline:
            logger.warning("Prediction requested but no trained model is loaded.")
            return None

        try:
            raw_text = text.strip()
            if not raw_text:
                return None

            # 1. Strip URLs, hashtags, pipe separators, and WELFake scraper-chrome artifact tokens
            cleaned = re.sub(r"https?://\S+", " ", raw_text)
            cleaned = re.sub(r"#\w+", " ", cleaned)
            cleaned = re.sub(r"\|+", ", ", cleaned)
            cleaned = _SCRAPER_ARTIFACT_WORDS.sub(" ", cleaned)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()

            if len(cleaned.split()) < 3:
                cleaned = raw_text

            word_count = len(raw_text.split())
            if word_count < 2:
                return {"p_fake": 0.25, "p_real": 0.75, "fact_check_reason": None}

            # 2. Compute feature log-odds from the trained TF-IDF + LogisticRegression pipeline
            if hasattr(pipeline, "named_steps") and "features" in pipeline.named_steps and "classifier" in pipeline.named_steps:
                feats = pipeline.named_steps["features"]
                clf = pipeline.named_steps["classifier"]
                X = feats.transform([cleaned])
                dot = float((X @ clf.coef_.T)[0, 0])
                # Neutralize WELFake's +1.0504 Reuters-dateline intercept bias
                logit = (dot - 0.65) * 0.80
            else:
                proba = pipeline.predict_proba([cleaned])[0]
                p_raw = float(np.clip(proba[1], 1e-4, 1.0 - 1e-4))
                logit = float(np.log(p_raw / (1.0 - p_raw)) - 1.2)

            # 3. Evaluate explicit sensationalist / conspiracy / hoax linguistic signals
            sensational_hits = len(_SENSATIONAL_PATTERNS.findall(raw_text))
            exclamations = raw_text.count("!")
            if sensational_hits > 0:
                logit += 2.2 * sensational_hits
            if exclamations >= 2:
                logit += 1.2

            # 4. If extracted from a verified news organization / broadcast channel and no hoax markers
            if is_verified_source and sensational_hits == 0:
                logit -= 1.15

            p_fake_style = float(1.0 / (1.0 + np.exp(-logit)))

            # 5. Live Factual Claim Verification (for claims not coming from a verified publisher URL)
            fact_reason = None
            p_fake = p_fake_style
            if not is_verified_source and sensational_hits == 0:
                fact_res = fact_check_service.verify_claim(raw_text)
                if fact_res is not None:
                    fc_prob = float(fact_res["p_fake"])
                    fact_reason = fact_res.get("reason")
                    if fact_res.get("is_fake"):
                        # Factual contradiction overrides stylistic neutrality
                        p_fake = max(fc_prob, 0.75 * fc_prob + 0.25 * p_fake_style)
                    else:
                        # Factually verified true claim
                        p_fake = min(p_fake_style, 0.70 * fc_prob + 0.30 * p_fake_style)

            p_fake = float(np.clip(p_fake, 0.01, 0.99))
            p_real = 1.0 - p_fake

            return {
                "p_fake": round(p_fake, 4),
                "p_real": round(p_real, 4),
                "fact_check_reason": fact_reason,
            }
        except Exception as e:
            logger.error(f"Error during text prediction: {e}", exc_info=True)
            return None

    def reload_model(self):
        """Hot-reload the model after retraining without restarting the server."""
        logger.info("Reloading NLP model...")
        self.load_model()


# Global instance for text inference
text_service = TextInferenceService()
