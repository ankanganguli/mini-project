import cv2
import numpy as np
from PIL import Image, ImageChops
import io
import re
import tempfile
import os
import logging
import threading
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Explicit AI generator and photo-manipulation software markers in file headers / EXIF / PNG chunks
_AI_AND_EDIT_SOFTWARE_MARKERS = re.compile(
    rb"(midjourney|stable\s*diffusion|dall[-_ ]?e|comfyui|automatic1111|novelai|"
    rb"adobe\s*firefly|firefly|imagen|flux\.1|leonardo\.ai|bing\s*image\s*creator|"
    rb"c2pa|trainedalgorithmicmedia|digital\s*source\s*type|photoshop|lightroom|"
    rb"gimp|canva|picsart|faceapp|remini|inshot|snapseed|pixelmator|affinity\s*photo|"
    rb"ai[-_ ]generated|dreamstudio|runwayml|pika\s*labs|sora|kling)",
    re.IGNORECASE,
)

# Standard AI diffusion model output resolutions (width, height)
_AI_DIFFUSION_DIMENSIONS = {
    (512, 512), (768, 768), (1024, 1024), (2048, 2048),
    (512, 768), (768, 512), (768, 1024), (1024, 768),
    (1024, 1536), (1536, 1024), (1024, 1792), (1792, 1024),
    (896, 1152), (1152, 896), (832, 1216), (1216, 832),
    (1344, 768), (768, 1344), (640, 1536), (1536, 640),
}

# Real physical camera EXIF tag IDs (Make, Model, ExposureTime, FNumber, ISOSpeedRatings, DateTimeOriginal, FocalLength, LensModel)
_CAMERA_EXIF_TAGS = {271, 272, 33434, 33437, 34855, 36867, 37386, 42036}


class VisionForensicService:
    """
    Multimodal visual forensics service.

    Techniques used:
    ─────────────────────────────────────────────────────────────────
    IMAGES:
      1. Deep Learning (ViT) — AI-generated image detector (when available)
      2. Localized Block-Grid Error Level Analysis (ELA) — detects splicing,
         inpainting, text overlays, and localized JPEG re-compression anomalies
      3. SRM High-Pass Sensor Noise Residual Analysis — detects lack of physical
         camera PRNU shot noise (characteristic of AI diffusion models like
         Midjourney/DALL-E/Flux) and regional noise inconsistency
      4. EXIF / Header / C2PA / Resolution Provenance Forensics — inspects binary
         headers for AI/editor software signatures, missing camera hardware tags,
         and synthetic diffusion grid dimensions
      5. DCT Frequency & Color-Channel Coherence Analysis — detects GAN/diffusion
         spectral anomalies and unnatural RGB channel locking
    ─────────────────────────────────────────────────────────────────
    All scores are in [0.0, 1.0] where 1.0 = highly manipulated.
    """

    def __init__(self):
        self._pipe = None
        self._lock = threading.Lock()
        self.load_model()

    def load_model(self):
        """Thread-safely loads the Hugging Face vision transformer model if available."""
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
                logger.info(f"Using multi-signal OpenCV/SRM/ELA/EXIF forensics (ViT offline/lightweight mode: {e})")
                self._pipe = None

    # ── Helpers ────────────────────────────────────────────────────────────

    def _bytes_to_pil(self, image_bytes: bytes) -> Image.Image:
        try:
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Corrupted or unsupported image format: {e}")

    def _bytes_to_cv2(self, image_bytes: bytes) -> Optional[np.ndarray]:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Corrupted or unsupported image format.")
        return img

    # ── Metadata, EXIF & Header Signature Analysis ─────────────────────────

    def _metadata_and_signature_score(self, image_bytes: bytes, pil_img: Image.Image) -> float:
        """
        Inspects binary headers, PNG chunks, and EXIF metadata for:
          - Explicit AI generator / photo editor signatures (C2PA, Midjourney, DALL-E, Photoshop, etc.)
          - Presence vs absence of real physical camera hardware EXIF tags
          - AI diffusion aspect ratios / exact synthetic resolutions
        Returns anomaly score in [0.0, 1.0].
        """
        try:
            header_slice = image_bytes[:65536]
            if _AI_AND_EDIT_SOFTWARE_MARKERS.search(header_slice):
                return 0.92

            # Check PIL info dictionary (PNG text chunks like 'parameters', 'prompt', 'Software')
            raw_pil = Image.open(io.BytesIO(image_bytes))
            info_str = " ".join(f"{k}:{v}" for k, v in (raw_pil.info or {}).items()).encode("utf-8", errors="ignore")
            if _AI_AND_EDIT_SOFTWARE_MARKERS.search(info_str):
                return 0.92
            if any(k.lower() in ("parameters", "prompt", "workflow", "dream", "sd-metadata") for k in (raw_pil.info or {})):
                return 0.95

            # Check EXIF camera hardware tags
            exif_data = raw_pil.getexif() if hasattr(raw_pil, "getexif") else {}
            camera_tag_hits = sum(1 for tag_id in _CAMERA_EXIF_TAGS if tag_id in exif_data)

            # Check if EXIF Software tag (305) has editing/AI software
            software_val = str(exif_data.get(305, "")).encode("utf-8", errors="ignore")
            if software_val and _AI_AND_EDIT_SOFTWARE_MARKERS.search(software_val):
                return 0.90

            w, h = pil_img.size
            img_format = (raw_pil.format or "").upper()

            # Real camera photos with 2+ hardware EXIF tags (Make, Model, Exposure, ISO) are strongly authentic
            if camera_tag_hits >= 2:
                return 0.05

            score = 0.52  # Baseline when all physical camera hardware EXIF is stripped/missing
            if (w, h) in _AI_DIFFUSION_DIMENSIONS or (w == h and w in (512, 640, 768, 960, 1024, 1280, 1440, 1536, 2048)):
                score += 0.32
            elif w % 64 == 0 and h % 64 == 0 and max(w, h) <= 2048:
                # AI latent diffusion models generate dimensions in multiples of 64
                score += 0.22
            elif w % 8 == 0 and h % 8 == 0 and camera_tag_hits == 0:
                score += 0.12

            if img_format in ("PNG", "WEBP") and camera_tag_hits == 0:
                score += 0.18

            return float(np.clip(score, 0.0, 1.0))
        except Exception as e:
            logger.debug(f"Metadata forensic check failed: {e}")
            return 0.30

    # ── SRM Sensor Noise Residual & AI Smoothness Forensics ────────────────

    def _srm_noise_anomaly_score(self, img_cv2: np.ndarray) -> float:
        """
        Applies a 2nd-order Spatial Rich Model (SRM) high-pass residual filter
        to extract micro-texture / sensor noise residuals.
        Detects:
          1. AI diffusion synthetic smoothness paired with hyper-sharp synthetic edges
          2. Regional splicing noise inconsistency across 32x32 mid-tone blocks
          3. Unnatural inter-channel RGB residual correlation
        Returns score in [0.0, 1.0].
        """
        try:
            img_f = img_cv2.astype(np.float32)
            kernel = np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=np.float32) / 4.0

            res_b = cv2.filter2D(img_f[:, :, 0], -1, kernel)
            res_g = cv2.filter2D(img_f[:, :, 1], -1, kernel)
            res_r = cv2.filter2D(img_f[:, :, 2], -1, kernel)

            gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY).astype(np.float32)
            res_gray = cv2.filter2D(gray, -1, kernel)

            h, w = gray.shape
            block_size = 32
            patch_vars = []
            for y in range(0, h - block_size + 1, block_size):
                for x in range(0, w - block_size + 1, block_size):
                    g_patch = gray[y:y + block_size, x:x + block_size]
                    mean_lum = float(np.mean(g_patch))
                    # Ignore pure black/white saturated blocks
                    if 20.0 < mean_lum < 235.0:
                        r_patch = res_gray[y:y + block_size, x:x + block_size]
                        patch_vars.append(float(np.var(r_patch)))

            if len(patch_vars) < 4:
                return 0.35

            patch_vars_np = np.array(patch_vars, dtype=np.float32)
            p25 = float(np.percentile(patch_vars_np, 25))
            p50 = float(np.percentile(patch_vars_np, 50))
            p95 = float(np.percentile(patch_vars_np, 95))

            # 1. Dynamic range of noise residual across patches:
            # Spliced images and AI renders (smooth skin/sky + sharp synthetic detail) have high p95/p25 ratio
            spread_ratio = p95 / (p25 + 0.15)
            spread_score = float(np.clip((spread_ratio - 6.0) / 22.0, 0.0, 1.0))

            # 2. Lack of natural sensor shot noise in smooth mid-tone patches (AI diffusion smoothness)
            smooth_anomaly = float(np.clip((1.8 - p25) / 1.8, 0.0, 1.0)) if p25 < 1.8 else 0.0

            # 3. Inter-channel noise correlation (AI generators synthesize R, G, B simultaneously from latent space)
            flat_r = res_r.flatten()[::4]
            flat_g = res_g.flatten()[::4]
            if flat_r.size > 100 and np.std(flat_r) > 1e-4 and np.std(flat_g) > 1e-4:
                corr_rg = abs(float(np.corrcoef(flat_r, flat_g)[0, 1]))
            else:
                corr_rg = 0.5
            corr_score = float(np.clip((corr_rg - 0.72) / 0.25, 0.0, 1.0))

            # 4. Coefficient of variation of residual energy (splicing mismatch)
            cv_noise = float(np.std(patch_vars_np) / (p50 + 0.25))
            cv_score = float(np.clip((cv_noise - 1.5) / 3.5, 0.0, 1.0))

            combined_srm = max(
                0.40 * spread_score + 0.35 * smooth_anomaly + 0.25 * corr_score,
                0.50 * cv_score + 0.30 * spread_score + 0.20 * corr_score,
            )
            return float(np.clip(combined_srm, 0.0, 1.0))
        except Exception as e:
            logger.debug(f"SRM noise analysis failed: {e}")
            return 0.30

    # ── DCT Frequency Analysis ─────────────────────────────────────────────

    def _dct_frequency_score(self, img_cv2: np.ndarray) -> float:
        """
        GAN & AI Diffusion frequency-domain artifact score via 2D DCT analysis.
        Checks both abnormal high-frequency energy (GAN checkerboard) and
        abnormal mid-to-high frequency roll-off (diffusion synthesis).
        Returns score in [0.0, 1.0].
        """
        try:
            gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY).astype(np.float32)
            h, w = gray.shape
            cy, cx = h // 2, w // 2
            half = min(128, h // 2, w // 2)
            patch = gray[max(0, cy - half):cy + half, max(0, cx - half):cx + half]
            # Ensure even dimensions for cv2.dct
            ph, pw = (patch.shape[0] // 2) * 2, (patch.shape[1] // 2) * 2
            patch = patch[:ph, :pw]
            if ph < 32 or pw < 32:
                return 0.0

            dct = np.abs(cv2.dct(patch))
            total = float(np.sum(dct)) + 1e-6
            hf_energy = float(np.sum(dct[ph // 2:, pw // 2:]))
            mf_energy = float(np.sum(dct[ph // 4:ph // 2, pw // 4:pw // 2]))

            hf_ratio = hf_energy / total
            mf_ratio = mf_energy / total

            # High HF energy (GAN checkerboard / over-sharpened splice)
            if hf_ratio > 0.075:
                hf_score = float(np.clip((hf_ratio - 0.075) * 8.0, 0.0, 1.0))
            # Ultra-low HF energy relative to MF (AI diffusion smoothing)
            elif hf_ratio < 0.022 and mf_ratio > 0.04:
                hf_score = float(np.clip((0.025 - hf_ratio) * 28.0, 0.0, 0.85))
            else:
                hf_score = 0.15

            return float(np.clip(hf_score, 0.0, 1.0))
        except Exception as e:
            logger.debug(f"DCT analysis failed: {e}")
            return 0.0

    # ── ELA (Localized Block-Grid Error Level Analysis) ────────────────────

    def _ela_score(self, original_pil: Image.Image) -> float:
        """
        Resaves at quality=90 and measures both global compression error AND
        localized 16x16 block-grid ELA anomalies (95th percentile vs median block error).
        Spliced/edited regions and pasted overlays cause strong localized block spikes.
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
            max_err = float(np.percentile(diff_np, 99.5))

            global_score = float(np.clip(0.5 * (mean_err / 12.0) + 0.5 * (std_err / 10.0), 0.0, 1.0))

            # Localized 16x16 block analysis to catch regional splicing / face swaps / text edits
            h, w, _ = diff_np.shape
            bsize = 16
            block_means = []
            for y in range(0, h - bsize + 1, bsize):
                for x in range(0, w - bsize + 1, bsize):
                    block_means.append(float(np.mean(diff_np[y:y + bsize, x:x + bsize])))

            if len(block_means) >= 8:
                bm = np.array(block_means, dtype=np.float32)
                b_med = float(np.median(bm))
                b_p95 = float(np.percentile(bm, 95))
                b_max = float(np.max(bm))
                local_ratio = b_p95 / (b_med + 0.75)
                peak_ratio = b_max / (b_med + 1.0)
                local_score = float(np.clip((local_ratio - 2.2) / 3.5, 0.0, 1.0))
                peak_score = float(np.clip((peak_ratio - 4.0) / 8.0, 0.0, 1.0))
            else:
                local_score = 0.0
                peak_score = 0.0

            extreme_pixel_score = float(np.clip((max_err - 28.0) / 45.0, 0.0, 1.0))

            return float(np.clip(max(global_score, 0.45 * local_score + 0.35 * peak_score + 0.20 * extreme_pixel_score), 0.0, 1.0))
        except Exception as e:
            logger.debug(f"ELA failed: {e}")
            return 0.0

    # ── Main Image Analysis ────────────────────────────────────────────────

    def analyze_image_ela(self, image_bytes: bytes, is_editorial_thumbnail: bool = False) -> float:
        """
        Full image forensics pipeline:
          1. ViT deep-learning classifier (when available)
          2. Localized Block-Grid ELA (editing/splicing detection)
          3. SRM High-Pass Sensor Noise Residual (AI diffusion & inpainting detection)
          4. Metadata / EXIF / AI Dimension Provenance
          5. DCT frequency analysis (GAN/diffusion artifact detection)
        Returns combined manipulation score [0.0, 1.0].
        """
        original = self._bytes_to_pil(image_bytes)
        img_cv2 = self._bytes_to_cv2(image_bytes)

        try:
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

            # ── Stage 2: Localized + Global ELA ────────────────────────────
            ela = self._ela_score(original)

            # ── Stage 3: DCT frequency ─────────────────────────────────────
            dct = self._dct_frequency_score(img_cv2) if img_cv2 is not None else 0.0

            if is_editorial_thumbnail:
                # For broadcast/news article thumbnails, avoid penalizing missing camera EXIF
                return float(np.clip(0.55 * ela + 0.45 * dct, 0.0, 0.45))

            # ── Stage 4: SRM Sensor Noise & AI Diffusion Texture ───────────
            srm = self._srm_noise_anomaly_score(img_cv2) if img_cv2 is not None else 0.35

            # ── Stage 5: EXIF / Software Signature / AI Resolution ─────────
            meta = self._metadata_and_signature_score(image_bytes, original)

            # If real physical camera EXIF hardware provenance is present and no heavy splicing ELA:
            if meta <= 0.10 and ela < 0.48:
                return float(np.clip(0.55 * ela + 0.30 * srm + 0.15 * meta, 0.05, 0.32))

            # ── Multi-Signal Fusion ────────────────────────────────────────
            heuristic_fusion = (
                0.34 * meta
                + 0.30 * srm
                + 0.24 * ela
                + 0.12 * dct
            )
            # If the top two forensic signals both show manipulation/synthetic traits, boost score
            strong_signals = sorted([ela, srm, meta, dct], reverse=True)
            top2_mean = (strong_signals[0] + strong_signals[1]) / 2.0
            forensic_score = max(heuristic_fusion, 0.90 * top2_mean if strong_signals[1] >= 0.30 else heuristic_fusion)

            if dl_score is not None:
                combined = max(0.55 * dl_score + 0.45 * forensic_score, forensic_score if dl_score > 0.30 else 0.35 * dl_score + 0.65 * forensic_score)
            else:
                combined = forensic_score

            return float(np.clip(combined, 0.0, 1.0))

        except Exception as e:
            logger.error(f"Failed image analysis: {e}", exc_info=True)
            return 0.0

    # ── Laplacian Sharpness & Edge-Coherence Consistency ──────────────────

    def analyze_laplacian_sharpness(self, image_bytes: bytes) -> float:
        """
        Measures LOCAL sharpness inconsistency across 4x4 image tiles plus
        high-frequency edge kurtosis.
        Spliced/composited images and AI-generated portraits (hyper-sharp subject
        on synthetic smooth background) have high tile sharpness disparity.
        Returns a continuous anomaly score in [0.0, 1.0].
        """
        img = self._bytes_to_cv2(image_bytes)
        if img is None:
            raise ValueError("Corrupted or unsupported image format.")

        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            tile_h = max(1, h // 4)
            tile_w = max(1, w // 4)
            variances = []

            for r in range(4):
                for c in range(4):
                    tile = gray[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w]
                    if tile.size > 0:
                        variances.append(float(cv2.Laplacian(tile, cv2.CV_64F).var()))

            if not variances:
                return 0.0

            mean_var = float(np.mean(variances))
            std_var = float(np.std(variances))
            p90_var = float(np.percentile(variances, 90))
            p20_var = float(np.percentile(variances, 20))

            if mean_var < 0.5:
                return 0.55  # Unnaturally flat / synthetic blank image

            # Inconsistency = std / mean; natural camera scenes ≈ 0.25–0.55; spliced/AI > 0.65
            inconsistency = std_var / (mean_var + 1e-6)
            inc_score = float(np.clip((inconsistency - 0.35) / 1.1, 0.0, 1.0))

            # Ratio between sharpest and smoothest tiles
            ratio_score = float(np.clip(((p90_var / (p20_var + 1.0)) - 4.0) / 18.0, 0.0, 1.0))

            return float(np.clip(max(inc_score, 0.65 * inc_score + 0.35 * ratio_score), 0.0, 1.0))

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
            mean_lap = float(np.mean(lap_scores))

            temporal_variance = float(
                np.clip(max(0.5 * temporal_var + 0.5 * lap_consistency, 0.7 * mean_lap), 0.0, 1.0)
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
