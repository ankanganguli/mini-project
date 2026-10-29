# 04_DATA_MODEL.md: Data & Schema Definitions

## 1. Target Binary Classification Mapping
- `0`: Authentic / Real
- `1`: Manipulated / Fake

## 2. Dynamic CSV Ingestion Contract
- Automatically detects candidate text column (longest average string length or column headers like `text`, `content`, `claim`, `title`).
- Automatically maps target labels (e.g. `label`, `target`, `fake`, `real`, `0/1`, `fake/real` strings normalized to integer `0` or `1`).
- Handles class imbalance via `class_weight='balanced'` in LinearSVC.

## 3. Verification Payload Schema
```json
{
  "verdict": "Authentic" | "Manipulated",
  "confidence_score": 0.942,
  "text_prob": 0.915,
  "forensic_entropy": 0.824,
  "timestamp": "2026-09-18T20:38:00.000Z"
}
```

## 4. MongoDB Document Schema
- Collection: `verifications`
  - `_id`: ObjectId
  - `claim_text`: String (optional)
  - `media_type`: "image" | "video" | "none"
  - `media_filename`: String (optional)
  - `text_prob`: Float
  - `forensic_entropy`: Float
  - `confidence_score`: Float
  - `verdict`: String
  - `timestamp`: ISODate
