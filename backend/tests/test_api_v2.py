import pytest
from fastapi.testclient import TestClient
import io
from PIL import Image

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_get_available_models():
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert isinstance(data["models"], list)

def test_upload_invalid_file_extension():
    file_content = b"Not an image"
    files = {"files": ("test.txt", file_content, "text/plain")}
    response = client.post("/api/v1/analysis/upload", files=files, data={"model": "medical_vit"})
    assert response.status_code == 415

def test_upload_corrupted_image():
    file_content = b"Corrupted image content"
    files = {"files": ("test.jpg", file_content, "image/jpeg")}
    response = client.post("/api/v1/analysis/upload", files=files, data={"model": "medical_vit"})
    assert response.status_code == 400

def test_upload_valid_image():
    img = Image.new('RGB', (10, 10), color = 'red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    files = {"files": ("test_valid.jpg", img_byte_arr, "image/jpeg")}
    response = client.post("/api/v1/analysis/upload", files=files, data={"model": "medical_vit"})

    assert response.status_code == 200
    data = response.json()
    assert "analysis_id" in data
    assert data["status"] == "queued"
