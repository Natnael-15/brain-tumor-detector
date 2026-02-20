import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.model_service import ModelService


def test_load_real_models_registers_medical_vit_without_keyerror():
    service = ModelService()

    # Ensure a clean model map and run the real-model loading path directly.
    service.models = {}
    asyncio.run(service._load_real_models())

    assert "medical_vit" in service.models
    assert service.models["medical_vit"]["config"]["name"] == "Medical Vision Transformer"
