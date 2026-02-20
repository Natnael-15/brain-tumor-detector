"""
Brain MRI Tumor Detector - Vercel Serverless API Entrypoint
Flask-based serverless function for Vercel deployment.
"""

from flask import Flask, jsonify
from datetime import datetime, timezone

app = Flask(__name__)


@app.route("/api/v1/health")
def health():
    """Health check endpoint."""
    return jsonify(
        {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "3.0.0",
            "environment": "vercel-serverless",
        }
    )


@app.route("/api/v1/models")
def models():
    """Return available detection models."""
    return jsonify(
        {
            "models": [
                {
                    "id": "ensemble",
                    "name": "Ensemble Model",
                    "type": "ensemble",
                    "description": "Multi-model ensemble with uncertainty quantification",
                },
                {
                    "id": "unet",
                    "name": "U-Net Segmentation",
                    "type": "segmentation",
                    "description": "3D U-Net for tumor segmentation",
                },
                {
                    "id": "resnet",
                    "name": "ResNet3D Classifier",
                    "type": "classification",
                    "description": "3D ResNet for tumor classification",
                },
            ],
            "total": 3,
            "message": "Full ML inference requires a dedicated backend deployment",
        }
    )


@app.route("/")
def index():
    """Root endpoint."""
    return jsonify(
        {
            "name": "Brain MRI Tumor Detector API",
            "version": "3.0.0",
            "docs": "/api/v1/health",
        }
    )
