import cv2
import numpy as np
from PIL import Image, ImageChops
import io
import tempfile
import os
import logging
import threading
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class VisionForensicService:
    """
    Multimodal visual forensics service.

    Techniques used:
    ─────────────────────────────────────────────────────────────────
    IMAGES:
      1. Deep Learning (ViT) — AI-generated image detector
         Model: umm-maybe/AI-image-detector (artificial vs human)
      2. Error Level Analysis (ELA) — detects JPEG re-compression
         anomalies that indicate photo editing/splicing
      3. DCT Frequency Analysis — GAN-generated images have
         characteristic high-frequency artifacts in the DCT domain

    VIDEOS:
      1. Per-frame ELA via the image pipeline (1 FPS sampling)
      2. Temporal consistency variance — sudden ELA jumps between
         frames indicate inserted/deepfaked segments
      3. Laplacian sharpness consistency — deepfakes often have
         uneven sharpness across frames (face region vs background)
    ─────────────────────────────────────────────────────────────────
    All scores are in [0.0, 1.0] where 1.0 = highly manipulated.
    """

    def __init__(self):
        self._pipe = None
        self._lock = threading.Lock()
        self.load_model()

    def load_model(self):
        """Thread-safely loads the Hugging Face vision transformer model."""
        with self._lock:
            try:
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                from transformers import pipeline
                logger.info("Loading Deep Learning Vision pipeline (AI Image Detection)...")
                self._pipe = pipeline(
                    "image-classification",
                    model="umm-maybe/AI-image-detector"
                )
                logger.info("Successfully loaded Vision pipeline.")
            except Exception as e:
                logger.info(f"Using OpenCV ELA + DCT frequency forensics (ViT offline/lightweight mode: {e})")
                self._pipe = None

    # ── Helpers ────────────────────────────────────────────────────────────

    def _bytes_to_pil(self, image_bytes: bytes) -> Image.Image:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    def _bytes_to_cv2(self, image_bytes: bytes) -> Optional[np.ndarray]:
        nparr = np.frombuffer(image_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # ── DCT Frequency Analysis ─────────────────────────────────────────────

    def _dct_frequency_score(self, img_cv2: np.ndarray) -> float:
        """
        GAN-artifact score via DCT analysis.
        GAN-generated images show abnormal high-frequency energy distribution
        (checkerboard pattern) in the DCT domain.
        Returns score in [0.0, 1.0].
        """
        try:
            gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY).astype(np.float32)
            h, w = gray.shape
            cy, cx = h // 2, w // 2
            half = 128
            patch = gray[max(0, cy - half):cy + half, max(0, cx - half):cx + half]
            if patch.shape[0] < 32 or patch.shape[1] < 32:
                return 0.0

            dct = cv2.dct(patch)
            total = float(np.sum(np.abs(dct))) + 1e-6
            h_p, w_p = dct.shape
            hf_energy = float(np.sum(np.abs(dct[h_p // 2:, w_p // 2:])))

            hf_ratio = hf_energy / total
            # Natural photos: ~5–10% energy in HF; GANs push this higher
            hf_baseline = 0.08
            hf_deviation = max(0.0, hf_ratio - hf_baseline) / (1.0 - hf_baseline)
            return float(np.clip(hf_deviation * 3.0, 0.0, 1.0))
        except Exception as e:
            logger.debug(f"DCT analysis failed: {e}")
            return 0.0

    # ── ELA (Error Level Analysis) ─────────────────────────────────────────

    def _ela_score(self, original_pil: Image.Image) -> float:
        """
        Resave at quality=90 and measure pixel-level compression discrepancy.
        Edited/spliced regions show higher residual error.
        Returns score in [0.0, 1.0].
        """
        try:
            buf = io.BytesIO()
            original_pil.save(buf, "JPEG", quality=90)
            buf.seek(0)
            resaved = Image.open(buf).convert("RGB")

            diff = ImageChops.difference(original_pil, resaved)
            diff_np = np.array(diff, dtype=np.float32)

            mean_err = float(np.mean(diff_np))
            std_err = float(np.std(diff_np))

            # Typical untampered JPEG: mean≈2–5, std≈1–3
            mean_score = float(np.clip(mean_err / 20.0, 0.0, 1.0))
            std_score = float(np.clip(std_err / 15.0, 0.0, 1.0))
            return float(0.6 * mean_score + 0.4 * std_score)
        except Exception as e:
            logger.debug(f"ELA failed: {e}")
            return 0.0

    # ── Main Image Analysis ────────────────────────────────────────────────

    def analyze_image_ela(self, image_bytes: bytes, is_editorial_thumbnail: bool = False) -> float:
        """
        Full image forensics pipeline:
          1. ViT deep-learning classifier (AI-generated detection, skipped for broadcast/article graphic thumbnails)
          2. ELA (editing/splicing detection)
          3. DCT frequency analysis (GAN artifact detection)
        Returns combined manipulation score [0.0, 1.0].
        """
        try:
            original = self._bytes_to_pil(image_bytes)
            img_cv2 = self._bytes_to_cv2(image_bytes)

            # ── Stage 1: Deep Learning (ViT) ───────────────────────────────
            dl_score = None
            if not is_editorial_thumbnail:
                with self._lock:
                    pipe = self._pipe

                if pipe:
                    result = pipe(original)
                    score_map = {item["label"].lower(): float(item["score"]) for item in result}

                    for fake_label in ["artificial", "fake", "ai-generated", "generated"]:
                        if fake_label in score_map:
                            dl_score = float(np.clip(score_map[fake_label], 0.0, 1.0))
                            break

                    if dl_score is None:
                        for real_label in ["human", "real", "authentic"]:
                            if real_label in score_map:
                                dl_score = float(np.clip(1.0 - score_map[real_label], 0.0, 1.0))
                                break

            # ── Stage 2: ELA ────────────────────────────────────────────────
            ela = self._ela_score(original)

            # ── Stage 3: DCT frequency ──────────────────────────────────────
            dct = self._dct_frequency_score(img_cv2) if img_cv2 is not None else 0.0

            # ── Fusion ──────────────────────────────────────────────────────
            if dl_score is not None:
                combined = 0.60 * dl_score + 0.25 * ela + 0.15 * dct
            else:
                combined = 0.55 * ela + 0.45 * dct

            return float(np.clip(combined, 0.0, 1.0))

        except Exception as e:
            logger.error(f"Failed image analysis: {e}", exc_info=True)
            return 0.0

    # ── Laplacian Sharpness Consistency ───────────────────────────────────

    def analyze_laplacian_sharpness(self, image_bytes: bytes) -> float:
        """
        Measures LOCAL sharpness inconsistency across image tiles.
        Spliced/composited images have mismatched sharpness regions
        (e.g., sharp pasted object on blurry background).
        Returns a continuous anomaly score in [0.0, 1.0].
        """
        try:
            img = self._bytes_to_cv2(image_bytes)
            if img is None:
                return 0.0

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            tile_h = h // 4
            tile_w = w // 4
            variances = []

            for r in range(4):
                for c in range(4):
                    tile = gray[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w]
                    if tile.size > 0:
                        variances.append(cv2.Laplacian(tile, cv2.CV_64F).var())

            if not variances:
                return 0.0

            mean_var = float(np.mean(variances))
            std_var = float(np.std(variances))

            if mean_var < 1.0:
                return 0.0

            # Inconsistency = std / mean; natural photos ≈ 0.3–0.8; spliced > 1.0
            inconsistency = std_var / (mean_var + 1e-6)
            return float(np.clip((inconsistency - 0.5) / 1.5, 0.0, 1.0))

        except Exception as e:
            logger.error(f"Failed Laplacian analysis: {e}", exc_info=True)
            return 0.0

    # ── Video Analysis ─────────────────────────────────────────────────────

    def analyze_video(self, video_bytes: bytes, max_frames: int = 20) -> Dict[str, float]:
        """
        Frame-by-frame video forensics:
          - Full image pipeline on sampled frames (1 FPS, up to max_frames)
          - Temporal ELA variance: sudden jumps = inserted/deepfaked segments
          - Laplacian consistency: deepfakes often have uneven blur per frame
        Returns dict with mean_ela and temporal_variance, both in [0.0, 1.0].
        """
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(video_bytes)
                temp_path = tmp.name

            cap = cv2.VideoCapture(temp_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0:
                fps = 30.0

            ela_scores: List[float] = []
            lap_scores: List[float] = []
            frame_count = 0

            while cap.isOpened() and len(ela_scores) < max_frames:
                frame_id = int(frame_count * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ret, frame = cap.read()
                if not ret:
                    break

                ok, buf = cv2.imencode(".jpg", frame)
                if ok:
                    frame_bytes = buf.tobytes()
                    ela_scores.append(self.analyze_image_ela(frame_bytes))
                    lap_scores.append(self.analyze_laplacian_sharpness(frame_bytes))

                frame_count += 1

            cap.release()

            if not ela_scores:
                return {"mean_ela": 0.0, "temporal_variance": 0.0}

            mean_ela = float(np.mean(ela_scores))

            # Temporal ELA variance: deepfakes = inconsistent ELA across frames
            temporal_var = float(np.clip(np.var(ela_scores) * 20.0, 0.0, 1.0))

            # Sharpness consistency across frames
            lap_consistency = float(np.clip(np.std(lap_scores) * 4.0, 0.0, 1.0))

            temporal_variance = float(
                np.clip(0.5 * temporal_var + 0.5 * lap_consistency, 0.0, 1.0)
            )

            logger.info(
                f"Video forensics: frames={len(ela_scores)} "
                f"mean_ela={mean_ela:.4f} temporal_var={temporal_variance:.4f}"
            )

            return {
                "mean_ela": round(mean_ela, 4),
                "temporal_variance": round(temporal_variance, 4),
            }

        except Exception as e:
            logger.error(f"Failed video analysis: {e}", exc_info=True)
            return {"mean_ela": 0.0, "temporal_variance": 0.0}
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass


# Global instance
vision_service = VisionForensicService()
