# 02_TRD.md: Technical Requirements Document

## 1. Technology Stack
- **Runtime & Language:** Python 3.11
- **Backend Framework:** FastAPI (async orchestration, multipart form data handling)
- **Machine Learning & NLP:** Scikit-Learn (TF-IDF vectorizer + CalibratedClassifierCV on LinearSVC for calibrated confidence scores)
- **Computer Vision & Forensics:** OpenCV (`opencv-python-headless` / `opencv-python`), Pillow, NumPy (Error Level Analysis - ELA, Laplacian frequency checks, and edge artifact entropy)
- **Database & Audit Store:** MongoDB via Motor / PyMongo for asynchronous audit trail persistence and analytics
- **Frontend / 3D Canvas:** Vanilla HTML5, Modern CSS, Three.js (r128) for 3D hero visualization

## 2. Dependencies & Environment
- `fastapi`
- `uvicorn[standard]`
- `scikit-learn`
- `joblib`
- `pandas`
- `numpy`
- `opencv-python-headless`
- `pillow`
- `motor`
- `pymongo`
- `python-multipart`
