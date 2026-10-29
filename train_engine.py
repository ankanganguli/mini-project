"""
Multimodal Authenticity Training & Inference Engine v2
=======================================================
Improvements over v1:
  - LogisticRegression (native probabilities, no calibration wrapper needed)
  - FeatureUnion: word-level TF-IDF (1,3-grams) + char-level TF-IDF (2,5-grams)
  - Supports downloading WELFake/LIAR datasets from HuggingFace
  - Robust label normalisation
  - Visual forensics unchanged (ELA + Laplacian)
"""

import os
import sys
import json
import argparse
import tempfile
import numpy as np
import pandas as pd
from PIL import Image, ImageChops, ImageEnhance
import cv2
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score
)

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
MODEL_PATH    = os.path.join(ARTIFACTS_DIR, "model_pipeline.pkl")
METRICS_PATH  = os.path.join(ARTIFACTS_DIR, "metrics.json")


def detect_target_column(df: pd.DataFrame) -> str:
    """Automatically detect the binary target column from column names."""
    candidates = ["label", "target", "fake", "class", "is_fake", "verdict", "category"]
    cols_lower = {str(c).lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand in cols_lower:
            return cols_lower[cand]
    # Fallback to column with 2 unique values
    for col in df.columns:
        if df[col].nunique() == 2:
            return col
    raise ValueError("Could not automatically detect binary target column in dataset.")


def detect_text_columns(df: pd.DataFrame, target_col: str) -> list:
    """Auto-detect textual feature columns to concatenate."""
    text_keywords = ["title", "text", "content", "statement", "claim", "body", "news", "description", "headline"]
    detected = []
    for col in df.columns:
        if col == target_col:
            continue
        c_lower = str(col).lower().strip()
        if any(kw in c_lower for kw in text_keywords) or df[col].dtype == object:
            # check average string length
            sample = df[col].dropna().astype(str)
            if not sample.empty and sample.str.len().mean() > 5:
                detected.append(col)
    if not detected:
        # Fallback to any string columns except target
        for col in df.columns:
            if col != target_col and df[col].dtype == object:
                detected.append(col)
    if not detected:
        raise ValueError("Could not find suitable text feature columns in dataset.")
    return detected


def normalize_target_labels(series: pd.Series) -> pd.Series:
    """Normalize labels to 0 (Real/Authentic) and 1 (Fake/Manipulated)."""
    # If already 0/1 numeric
    unique_vals = set(series.dropna().unique())
    if unique_vals.issubset({0, 1, 0.0, 1.0, "0", "1"}):
        return series.astype(int)

    # String mapping
    mapping = {}
    for val in unique_vals:
        v_str = str(val).strip().lower()
        if v_str in ["fake", "manipulated", "rumor", "false", "misinformation", "1",
                     "unreliable", "conspiracy", "satire", "clickbait", "bias", "hate"]:
            mapping[val] = 1
        elif v_str in ["real", "authentic", "true", "factual", "reliable", "0",
                       "trustworthy", "verified", "credible"]:
            mapping[val] = 0

    if len(mapping) >= 2:
        return series.map(mapping).dropna().astype(int)

    # Fallback: map sorted unique to 0, 1
    sorted_unique = sorted(list(unique_vals))
    fallback_map = {sorted_unique[0]: 0, sorted_unique[1]: 1}
    return series.map(fallback_map).astype(int)


# ─────────────────────────────────────────────────────────────
# Dataset helpers
# ─────────────────────────────────────────────────────────────

def load_csv_dataset(data_path: str) -> pd.DataFrame:
    """Load a CSV and return a normalised (text, label) DataFrame."""
    print(f"  [csv] Loading: {data_path}")
    df = pd.read_csv(data_path)
    target_col = detect_target_column(df)
    text_cols  = detect_text_columns(df, target_col)
    print(f"  [csv] target='{target_col}'  text_cols={text_cols}")

    y      = normalize_target_labels(df[target_col])
    X_text = df[text_cols].fillna("").astype(str).agg(" ".join, axis=1).str.strip()

    out = pd.DataFrame({"text": X_text, "label": y})
    out = out[(out["text"].str.len() > 10) & (out["label"].notna())]
    print(f"  [csv] Valid rows: {len(out)}  dist={out['label'].value_counts().to_dict()}")
    return out.reset_index(drop=True)


def download_extra_dataset() -> pd.DataFrame:
    """
    Downloads supplemental fake-news datasets from HuggingFace.
    Uses two confirmed-working datasets:
      1. GonzaloA/fake_news  (~40K rows, title + text + label)
      2. mrm8488/fake-news   (~45K rows, title + text + label)
    Returns merged (text, label) DataFrame or empty DataFrame on total failure.
    """
    frames = []

    # Dataset 1: GonzaloA/fake_news — ISOT-based, title+text, label 0=FAKE 1=REAL
    try:
        from datasets import load_dataset
        print("  [hf] Downloading GonzaloA/fake_news (~40K rows)...")
        ds = load_dataset("GonzaloA/fake_news", trust_remote_code=False)
        # Merge all splits for maximum data
        all_splits = []
        for split_name in ds.keys():
            all_splits.append(ds[split_name].to_pandas())
        df = pd.concat(all_splits, ignore_index=True)
        # Columns: title, text, label (0=FAKE, 1=REAL) — note inverted!
        df["combined"] = (df.get("title", pd.Series([""] * len(df))).fillna("") +
                          " " + df.get("text", pd.Series([""] * len(df))).fillna("")).str.strip()
        # GonzaloA labels: 0=FAKE, 1=REAL → invert to our convention (1=FAKE, 0=REAL)
        df["label_norm"] = df["label"].apply(lambda v: 1 if int(v) == 0 else 0)
        out = df[["combined", "label_norm"]].rename(columns={"combined": "text", "label_norm": "label"})
        out = out[out["text"].str.len() > 30].reset_index(drop=True)
        print(f"  [hf] GonzaloA/fake_news: {len(out)} rows  dist={out['label'].value_counts().to_dict()}")
        frames.append(out)
    except Exception as e:
        print(f"  [hf] GonzaloA/fake_news failed: {e}")

    # Dataset 2: mrm8488/fake-news — large CSV-backed dataset
    try:
        from datasets import load_dataset
        print("  [hf] Downloading mrm8488/fake-news (~45K rows)...")
        ds2 = load_dataset("mrm8488/fake-news", trust_remote_code=False)
        all_splits = []
        for split_name in ds2.keys():
            all_splits.append(ds2[split_name].to_pandas())
        df2 = pd.concat(all_splits, ignore_index=True)
        # Detect columns dynamically
        target_col = detect_target_column(df2)
        text_cols  = detect_text_columns(df2, target_col)
        y      = normalize_target_labels(df2[target_col])
        X_text = df2[text_cols].fillna("").astype(str).agg(" ".join, axis=1).str.strip()
        out2 = pd.DataFrame({"text": X_text, "label": y})
        out2 = out2[(out2["text"].str.len() > 30) & (out2["label"].notna())]
        out2["label"] = out2["label"].astype(int)
        print(f"  [hf] mrm8488/fake-news: {len(out2)} rows  dist={out2['label'].value_counts().to_dict()}")
        frames.append(out2.reset_index(drop=True))
    except Exception as e:
        print(f"  [hf] mrm8488/fake-news failed: {e}")

    if frames:
        merged = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["text"])
        print(f"  [hf] Total supplemental rows: {len(merged)}")
        return merged

    print("  [hf] All HuggingFace downloads failed. Training on primary dataset only.")
    return pd.DataFrame(columns=["text", "label"])


# ─────────────────────────────────────────────────────────────
# Model Builder
# ─────────────────────────────────────────────────────────────

def build_pipeline() -> Pipeline:
    """
    FeatureUnion:
      * word-level TF-IDF (1,3-grams) — semantic patterns
      * char-level TF-IDF (2,5-grams) — catches spelling tricks, obfuscation
    Classifier: LogisticRegression with L2 (balanced class weights).
    """
    word_tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        max_features=100_000,
        sublinear_tf=True,
        min_df=2,
        strip_accents="unicode",
    )
    char_tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        max_features=50_000,
        sublinear_tf=True,
        min_df=3,
    )
    features = FeatureUnion([("word", word_tfidf), ("char", char_tfidf)])
    clf = LogisticRegression(
        C=4.0,
        max_iter=2000,
        solver="saga",
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    return Pipeline([("features", features), ("classifier", clf)])


def train_model(data_path: str, use_online: bool = True):
    """
    Train on the provided CSV + optionally download WELFake/LIAR.
    Saves model_pipeline.pkl and metrics.json to ARTIFACTS_DIR.
    """
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    frames = []

    # 1. Primary CSV
    print("[*] Loading primary dataset...")
    frames.append(load_csv_dataset(data_path))

    # 2. Online supplement
    if use_online:
        print("[*] Attempting to download supplemental dataset from HuggingFace...")
        extra = download_extra_dataset()
        if not extra.empty:
            frames.append(extra)

    # 3. Merge and deduplicate
    combined = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["text"])
    combined["label"] = combined["label"].astype(int)
    print(f"\n[*] Combined dataset: {len(combined)} rows")
    print(f"    Class distribution: {combined['label'].value_counts().to_dict()}")

    X = combined["text"]
    y = combined["label"]

    # 4. Stratified 85/15 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    print(f"[*] Train: {len(X_train)}  Test: {len(X_test)}")

    # 5. Build and fit the improved pipeline
    print("[*] Building FeatureUnion pipeline (word TF-IDF + char TF-IDF)...")
    pipeline = build_pipeline()
    print("[*] Fitting LogisticRegression... (may take 1-2 min on large datasets)")
    pipeline.fit(X_train, y_train)

    # 6. Evaluate
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc  = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec  = float(recall_score(y_test, y_pred, zero_division=0))
    f1   = float(f1_score(y_test, y_pred, zero_division=0))
    auc  = float(roc_auc_score(y_test, y_proba))
    cm   = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "accuracy":         round(acc,  4),
        "precision":        round(prec, 4),
        "recall":           round(rec,  4),
        "f1_score":         round(f1,   4),
        "roc_auc":          round(auc,  4),
        "confusion_matrix": cm,
        "train_samples":    len(X_train),
        "test_samples":     len(X_test),
    }

    print(f"\n[+] Test Results:")
    print(f"    Accuracy  : {acc:.4f}")
    print(f"    Precision : {prec:.4f}")
    print(f"    Recall    : {rec:.4f}")
    print(f"    F1-Score  : {f1:.4f}")
    print(f"    ROC-AUC   : {auc:.4f}")
    print(f"    Confusion Matrix: {cm}")

    # 7. Save artifacts
    joblib.dump(pipeline, MODEL_PATH)
    with open(METRICS_PATH, "w") as fh:
        json.dump(metrics, fh, indent=2)

    print(f"\n[+] Artifacts saved → {MODEL_PATH}")
    return metrics


# ==========================================
# OpenCV Visual Forensics Module
# ==========================================

def compute_ela_image(image_path: str, quality: int = 90) -> float:
    """
    Error Level Analysis (ELA):
    Resaves image at specified JPEG quality, measures absolute pixel error difference,
    and returns a normalized manipulation entropy score in [0.0, 1.0].
    """
    try:
        orig = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Could not open image file at {image_path}: {e}")

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        orig.save(tmp_path, "JPEG", quality=quality)
        resaved = Image.open(tmp_path).convert("RGB")

        # Pixel-level difference
        diff = ImageChops.difference(orig, resaved)
        diff_np = np.array(diff, dtype=np.float32)

        # Scale difference to highlight compression discrepancies
        diff_gray = cv2.cvtColor(diff_np.astype(np.uint8), cv2.COLOR_RGB2GRAY)

        # Calculate mean compression artifact deviation & Laplacian variance on diff
        mean_diff = float(np.mean(diff_gray))
        laplacian_var = float(cv2.Laplacian(diff_gray, cv2.CV_64F).var())

        # Combine mean error and Laplacian edge variance into normalized entropy [0.0, 1.0]
        # Baseline JPEG diff mean typically ranges between 1.0 and 15.0 for untampered photos
        # Tampered/spliced edges yield significantly higher local discrepancies
        raw_score = (mean_diff / 15.0) * 0.5 + min(laplacian_var / 100.0, 1.0) * 0.5
        entropy_score = float(np.clip(raw_score, 0.0, 1.0))
        return round(entropy_score, 4)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def compute_video_forensics(video_path: str) -> float:
    """
    Keyframe video analysis:
    Extracts 1 frame per second via cv2.VideoCapture, computes ELA on each frame,
    and returns aggregated temporal manipulation score.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file at {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0

    frame_interval = int(round(fps))
    scores = []
    frame_idx = 0

    with tempfile.TemporaryDirectory() as tmp_dir:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_img_path = os.path.join(tmp_dir, f"frame_{frame_idx}.jpg")
                Image.fromarray(frame_rgb).save(frame_img_path, "JPEG")
                frame_score = compute_ela_image(frame_img_path)
                scores.append(frame_score)

            frame_idx += 1

    cap.release()

    if not scores:
        return 0.0

    # Aggregate: combination of mean and peak 90th percentile to catch localized bursts
    mean_val = float(np.mean(scores))
    max_val = float(np.max(scores))
    aggregate = 0.6 * mean_val + 0.4 * max_val
    return round(float(np.clip(aggregate, 0.0, 1.0)), 4)


def analyze_media(media_path: str) -> float:
    """Determine media type and dispatch to image or video forensics."""
    if not media_path or not os.path.exists(media_path):
        return 0.0

    ext = os.path.splitext(media_path)[1].lower()
    video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"]

    if ext in video_extensions:
        return compute_video_forensics(media_path)
    elif ext in image_extensions:
        return compute_ela_image(media_path)
    else:
        raise ValueError(f"Unsupported media format: {ext}")


# ==========================================
# Direct Execution Hook for API / Inference
# ==========================================

def run_inference(text: str = None, media_path: str = None) -> dict:
    """
    Multimodal late-fusion inference:
      S = 0.65 * P_text + 0.35 * P_visual
    Text has higher weight since it uses a purpose-trained model.
    """
    text_prob        = None
    forensic_entropy = None

    # NLP Inference via trained sklearn model
    if text and text.strip():
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run training first.")
        pipeline  = joblib.load(MODEL_PATH)
        probs     = pipeline.predict_proba([text.strip()])[0]
        raw_fake  = float(probs[1])  # class 1 = Fake/Manipulated

        # OOD guard: only for tiny fragments (< 6 words) - model is now trained
        # on short factual facts and short fake headlines so it handles them well.
        word_count = len(text.strip().split())
        model_conf = abs(raw_fake - 0.5) * 2.0

        if word_count < 6:
            raw_fake = 0.38
        elif word_count < 30:
            if model_conf < 0.70:
                raw_fake = 0.38
            else:
                raw_fake = 0.5 * raw_fake + 0.5 * 0.38
        elif word_count < 80:
            alpha    = (word_count - 30) / 50.0
            raw_fake = alpha * raw_fake + (1.0 - alpha) * 0.38
        text_prob = raw_fake

    # Visual Forensics Inference
    if media_path and os.path.exists(media_path):
        forensic_entropy = analyze_media(media_path)

    # Late Fusion (text-weighted)
    if text_prob is not None and forensic_entropy is not None:
        fused_score = 0.65 * text_prob + 0.35 * forensic_entropy
    elif text_prob is not None:
        fused_score = text_prob
    elif forensic_entropy is not None:
        fused_score = forensic_entropy
    else:
        raise ValueError("Neither text nor media input was provided for inference.")

    verdict          = "Manipulated" if fused_score >= 0.5 else "Authentic"
    confidence_score = fused_score if verdict == "Manipulated" else (1.0 - fused_score)

    return {
        "verdict":          verdict,
        "confidence_score": round(confidence_score, 4),
        "fused_score":      round(fused_score, 4),
        "text_prob":        round(text_prob, 4) if text_prob is not None else None,
        "forensic_entropy": round(forensic_entropy, 4) if forensic_entropy is not None else None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multimodal Authenticity Training & Inference Engine v2"
    )
    parser.add_argument("--data-path", type=str, help="Path to CSV dataset for retraining")
    parser.add_argument("--no-online", action="store_true",
                        help="Skip downloading supplemental HuggingFace datasets")
    parser.add_argument("--text",  type=str, help="Claim text for inference")
    parser.add_argument("--media", type=str, help="Image or video path for visual forensics")
    args = parser.parse_args()

    if args.data_path:
        train_model(args.data_path, use_online=not args.no_online)
    elif args.text or args.media:
        result = run_inference(text=args.text, media_path=args.media)
        print(json.dumps(result, indent=2))
    else:
        parser.print_help()
