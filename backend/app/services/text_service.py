"""
Text Inference Service
======================
Uses the trained sklearn pipeline (TF-IDF + LogisticRegression) stored at
artifacts/model_pipeline.pkl.  This is the model produced by train_engine.py
and is purpose-trained on real fake-news datasets — unlike the generic zero-shot
NLI model that was used previously (which caused universal truths to be marked fake).
"""
import os
import threading
import logging
import joblib
from typing import Optional, Dict

logger = logging.getLogger(__name__)

# Path to the trained artifact (relative to project root)
_ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "artifacts")
MODEL_PATH = os.path.abspath(os.path.join(_ARTIFACTS_DIR, "model_pipeline.pkl"))


class TextInferenceService:
    """
    Service for text-based NLP classification.
    Loads the trained sklearn pipeline from artifacts/ and returns calibrated
    probability that the input text is fake (p_fake) or real (p_real).
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

    def predict(self, text: str) -> Optional[Dict[str, float]]:
        """
        Returns calibrated probability estimates for the input text.
        Output: { "p_fake": float, "p_real": float } or None if model unavailable.
        """
        with self._lock:
            pipeline = self._pipeline

        if not pipeline:
            logger.warning("Prediction requested but no trained model is loaded.")
            return None

        try:
            clean_text = text.strip()
            if not clean_text:
                return None

            # predict_proba returns [p_real, p_fake] (class 0=Real, 1=Fake)
            proba = pipeline.predict_proba([clean_text])[0]
            p_fake = float(proba[1])
            p_real = float(proba[0])

            # ── OOD guard: only for very tiny fragments (< 4 words) ──
            # The model handles short factual sentences well.
            # Only 1-3 word inputs are truly out-of-distribution.
            word_count = len(clean_text.split())
            if word_count < 4:
                p_fake = 0.38
                p_real = 0.62
            # ─────────────────────────────────────────────────────────────────

            return {
                "p_fake": round(p_fake, 4),
                "p_real": round(p_real, 4),
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
