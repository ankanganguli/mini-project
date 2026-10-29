# 07_API_CONTRACT.md: REST API Specifications

## Base URL
`/api/v1`

## Endpoints

### 1. `POST /api/v1/train`
- **Description:** Accepts CSV file upload, validates schema dynamically, starts background retraining, and writes serialized model artifacts (`tfidf_vectorizer.joblib`, `calibrated_svc.joblib`, `metrics.json`).
- **Request:** `multipart/form-data` with `file: UploadFile`.
- **Response:**
```json
{
  "status": "training_initiated",
  "filename": "dataset.csv",
  "message": "Dataset accepted. Model training running asynchronously."
}
```

### 2. `POST /api/v1/verify`
- **Description:** Multimodal verification. Accepts text claim and/or media (image or video), processes through parallel pipelines, applies late fusion, logs to MongoDB, and returns explainable authenticity verdict.
- **Request:** `multipart/form-data`
  - `claim_text` (optional): string
  - `media_file` (optional): file (image or video)
- **Response:**
```json
{
  "verdict": "Manipulated",
  "confidence_score": 0.8845,
  "text_prob": 0.9120,
  "forensic_entropy": 0.8570,
  "timestamp": "2026-09-18T20:38:00.000Z"
}
```

### 3. `GET /api/v1/metrics`
- **Description:** Returns active model validation accuracy, precision, recall, F1, and total verification audit counts.
- **Response:**
```json
{
  "model_accuracy": 0.935,
  "precision": 0.928,
  "recall": 0.941,
  "f1_score": 0.934,
  "total_verifications": 1420,
  "last_trained": "2026-09-18T20:00:00.000Z"
}
```
