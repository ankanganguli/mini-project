# 01_PRD.md: Product Requirements Document

## 1. Overview & Vision
A production-grade multimodal authenticity verification platform empowering newsrooms, fact-checkers, and end users with transparent, explainable authenticity ratings across text claims, manipulated images, and video frames.

## 2. Target Personas & Workflows
- **Fact-Checkers & Analysts:** Detailed breakdown of textual confidence, Error Level Analysis (ELA) heatmaps, Laplacian frequency distributions, and audit trails.
- **Newsrooms:** High-throughput batch API verification with explainable late-fusion scores for editorial decision making.
- **End Users:** Clear verdicts (Authentic vs. Manipulated), confidence gauges, and interactive 3D visual cues indicating deep analysis status.

## 3. Core Functional Requirements
- **Textual Claim Verification:** Dynamic NLP classification trained on real vs. manipulated text corpuses with calibrated probabilities.
- **Visual Tampering Detection:** OpenCV-driven Error Level Analysis (ELA) detecting compression discrepancies and Laplacian frequency checks for splicing and edge anomalies.
- **Video Frame Integrity:** Sampling and visual forensic evaluation across video frames to detect frame-level tampering or deepfake manipulation artifacts.
- **Explainability & Transparency:** Multimodal score breakdown with text probability, visual forensic entropy, and clear decision rationales.
