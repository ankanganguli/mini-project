# Multimodal Authenticity Verification Platform

A production-grade multimodal authenticity verification platform empowering newsrooms, fact-checkers, and end users with transparent, explainable authenticity ratings across text claims, manipulated images, and video frames.

## Features
- **Textual Claim Verification**: Dynamic NLP classification trained on real vs. manipulated text corpuses.
- **Visual Tampering Detection**: OpenCV-driven Error Level Analysis (ELA) and Laplacian frequency checks for splicing and edge anomalies.
- **Video Frame Integrity**: Sampling and visual forensic evaluation across video frames to detect frame-level tampering.
- **Explainability & Transparency**: Multimodal late-fusion scoring with breakdown of visual and textual confidence metrics.

## Tech Stack
- **Backend**: FastAPI, PyTorch, Scikit-learn, OpenCV, Uvicorn
- **Frontend**: Modern responsive web dashboard (HTML/CSS/JS)
- **Containerization**: Docker, Docker Compose, Nginx
 BACKEND: https://mini-project-2-op98.onrender.com/
