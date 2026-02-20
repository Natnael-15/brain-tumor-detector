import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.model_service import ModelService


class SingleArgPredictor:
    def predict(self, file_path):
        return {
            "prediction": {"tumor_detected": False, "confidence": 0.9},
            "segmentation": {"dice_score": 0.0},
            "metrics": {"processing_time": 0.01},
        }


class AsyncTwoArgPredictor:
    async def predict(self, file_path, analysis_id):
        return {
            "prediction": {"tumor_detected": True, "confidence": 0.8},
            "segmentation": {"dice_score": 0.5},
            "metrics": {"processing_time": 0.02},
            "echo_analysis_id": analysis_id,
        }


def test_load_real_models_registers_medical_vit_without_keyerror():
    service = ModelService()

    service.models = {}
    asyncio.run(service._load_real_models())

    assert "medical_vit" in service.models
    assert service.models["medical_vit"]["config"]["name"] == "Medical Vision Transformer"


def test_predict_supports_single_arg_sync_predictor():
    service = ModelService()
    service._initialized = True
    service.models = {
        "unet3d": {
            "predictor": SingleArgPredictor(),
            "config": service.model_configs["unet3d"],
            "loaded": True,
            "type": "real",
        }
    }

    result = asyncio.run(service.predict("unet3d", "/tmp/fake.nii", "analysis-1"))

    assert result["analysis_id"] == "analysis-1"
    assert result["model_id"] == "unet3d"
    assert "prediction" in result


def test_predict_supports_async_two_arg_predictor():
    service = ModelService()
    service._initialized = True
    service.models = {
        "resnet3d": {
            "predictor": AsyncTwoArgPredictor(),
            "config": service.model_configs["resnet3d"],
            "loaded": True,
            "type": "mock",
        }
    }

    result = asyncio.run(service.predict("resnet3d", "/tmp/fake.nii", "analysis-2"))

    assert result["analysis_id"] == "analysis-2"
    assert result["echo_analysis_id"] == "analysis-2"
