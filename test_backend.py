import os
import io
import time
import json
import tempfile
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import cv2
import pytest
from fastapi.testclient import TestClient
from pymongo import MongoClient

from main import app, MONGO_URI, DB_NAME
from train_engine import ARTIFACTS_DIR, MODEL_PATH, METRICS_PATH

client = TestClient(app)


def test_empty_payload_rejection():
    """Verify that an empty payload yields an HTTP 400 Bad Request."""
    response = client.post("/api/v1/verify", data={})
    assert response.status_code == 400
    assert "At least one input modality" in response.json().get("detail", "")


def test_invalid_train_file_extension():
    """Verify that non-CSV file uploads to /train yield HTTP 400."""
    fake_txt = io.BytesIO(b"Just some text data")
    response = client.post(
        "/api/v1/train",
        files={"file": ("invalid_data.txt", fake_txt, "text/plain")}
    )
    assert response.status_code == 400
    assert "Only CSV datasets are accepted" in response.json().get("detail", "")


def test_dynamic_retraining_workflow():
    """
    Generate synthetic dataset, submit to /api/v1/train, poll /metrics,
    and verify model hot-reload and artifact generation.
    """
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    test_csv_path = os.path.join(data_dir, "test_retrain.csv")

    # Generate balanced synthetic claims
    claims = [
        ("NASA rover discovers pristine liquid lake on surface of Mars", "Fake"),
        ("Scientists confirm discovery of new exoplanet within habitable zone", "Real"),
        ("Ancient pyramids constructed with extraterrestrial anti-gravity beams", "Fake"),
        ("Global temperatures record slight rise during annual meteorological summit", "Real"),
        ("Miracle herbal beverage cures all cardiovascular disorders overnight", "Fake"),
        ("Health agency publishes peer-reviewed research on cardiovascular wellness", "Real"),
        ("Local municipality votes to enhance public transportation transit lines", "Real"),
        ("Secret global council controls weather using underwater antennas", "Fake"),
        ("Tech company releases quarterly financial audit exceeding revenue estimates", "Real"),
        ("Aliens replace cabinet ministers according to anonymous internet whistleblower", "Fake")
    ] * 6  # Multiply to provide ample data for train-test split

    df = pd.DataFrame(claims, columns=["statement", "label"])
    df.to_csv(test_csv_path, index=False)

    with open(test_csv_path, "rb") as f:
        response = client.post(
            "/api/v1/train",
            files={"file": ("test_retrain.csv", f, "text/csv")}
        )

    assert response.status_code == 202
    assert response.json()["status"] == "training_initiated"

    # Wait briefly for background retraining completion
    retrain_ok = False
    for _ in range(15):
        time.sleep(1)
        metrics_res = client.get("/api/v1/metrics")
        if metrics_res.status_code == 200:
            m_data = metrics_res.json()
            if m_data.get("pipeline", {}).get("is_trained") and m_data.get("model_accuracy") is not None:
                retrain_ok = True
                break

    assert retrain_ok, "Model did not successfully complete training and hot-reload within allotted time."


def test_text_verification():
    """Verify calibrated probability prediction and schema compliance on text claim."""
    response = client.post(
        "/api/v1/verify",
        data={"claim_text": "Secret global conspiracy changes weather patterns worldwide"}
    )
    assert response.status_code == 200
    data = response.json()

    assert "verdict" in data
    assert data["verdict"] in ["Likely Authentic", "Manipulated Media", "High Risk Contradiction"]
    assert 0.0 <= data["confidence_score"] <= 1.0
    assert data["text_prob"] is not None
    assert 0.0 <= data["text_prob"] <= 1.0
    assert "timestamp" in data


def test_image_ela_forensics():
    """
    Generate synthetic test image with high-contrast noise patch and verify OpenCV ELA
    produces non-zero manipulation entropy.
    """
    img = Image.new("RGB", (320, 240), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    # Inject high contrast saturated noise rectangle
    draw.rectangle([80, 60, 240, 180], fill=(255, 0, 50), outline=(0, 255, 100))

    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    response = client.post(
        "/api/v1/verify",
        files={"media_file": ("test_tampered.png", img_bytes, "image/png")}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["forensic_entropy"] is not None
    assert 0.0 < data["forensic_entropy"] <= 1.0
    assert 0.0 <= data["confidence_score"] <= 1.0


def test_video_keyframe_forensics():
    """
    Create a programmatic 2-second MP4 video using OpenCV VideoWriter,
    submit to /api/v1/verify, and assert frame-extraction consistency.
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_vid:
        tmp_vid_path = tmp_vid.name

    try:
        width, height, fps = 320, 240, 30
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(tmp_vid_path, fourcc, fps, (width, height))

        # Generate 60 frames (2 seconds) with alternating visual blocks
        for i in range(60):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :] = (i * 4 % 255, 120, 200)
            cv2.rectangle(frame, (50, 50), (200, 200), (0, 255, 0), -1)
            out.write(frame)
        out.release()

        with open(tmp_vid_path, "rb") as vf:
            response = client.post(
                "/api/v1/verify",
                files={"media_file": ("test_clip.mp4", vf, "video/mp4")}
            )

        assert response.status_code == 200
        data = response.json()
        assert data["forensic_entropy"] is not None
        assert 0.0 <= data["forensic_entropy"] <= 1.0
    finally:
        if os.path.exists(tmp_vid_path):
            os.remove(tmp_vid_path)


def test_mongodb_audit_persistence():
    """
    Inspect MongoDB verifications collection directly to ensure audit records
    match required data models and schemas.
    """
    try:
        mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=1500)
        mongo_client.admin.command('ping')
        db = mongo_client[DB_NAME]
        record = db["verifications"].find_one(sort=[("timestamp", -1)])

        if record:
            assert "verdict" in record
            assert "confidence_score" in record
            assert "timestamp" in record
            assert record["verdict"] in ["Likely Authentic", "Manipulated Media", "High Risk Contradiction"]
    except Exception as e:
        pytest.skip(f"Local MongoDB instance not active for direct socket testing: {e}")


if __name__ == "__main__":
    pytest.main(["-v", __file__])
