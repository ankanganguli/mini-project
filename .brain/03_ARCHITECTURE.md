# 03_ARCHITECTURE.md: System Architecture & Orchestration

## 1. Asynchronous Orchestration
- Powered by FastAPI asynchronous worker endpoints.
- Non-blocking I/O handling file uploads, background retraining tasks, and MongoDB operations.

## 2. Parallel Processing Pipelines
1. **NLP Classification Track:**
   - Preprocessing text claims.
   - TF-IDF feature extraction.
   - Calibrated LinearSVC predicting probability $P_{\text{text}} \in [0, 1]$ representing probability of manipulation/fake.
2. **Visual Forensics Track:**
   - Multi-format ingestion (JPEG, PNG, MP4, WebM frame sampling).
   - Error Level Analysis (ELA): computes compression error differences between original and resaved JPEG.
   - Laplacian Frequency Analysis: calculates high-frequency edge variance and entropy to detect tampering/splicing boundaries.
   - Outputs visual tampering probability $P_{\text{visual}} \in [0, 1]$.

## 3. Late Fusion Formula
$$\text{Score } S = w_1 \cdot P_{\text{text}} + w_2 \cdot P_{\text{visual}}$$
- Where $w_1 + w_2 = 1.0$ (default $w_1 = 0.5, w_2 = 0.5$, dynamically adjusted if single modality provided).
- Thresholding:
  - $S \ge 0.5 \implies$ Verdict: **Manipulated / Fake**
  - $S < 0.5 \implies$ Verdict: **Authentic / Real**

## 4. Persistence & Audit Logging
- Every verification request is logged to MongoDB collections (`verifications` and `model_metrics`) with exact input hashes, modality scores, timestamp, and verdict.
