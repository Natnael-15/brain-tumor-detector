import asyncio

import importlib

model_service_module = importlib.import_module("backend.services.model_service")
ModelService = model_service_module.ModelService


class DummyPredictor:
    def __init__(self):
        self.calls = []

    def predict(self, file_path, analysis_id=None):
        self.calls.append((file_path, analysis_id))
        return {"ok": True}


def test_predict_handles_signature_introspection_failure(monkeypatch):
    service = ModelService()
    service._initialized = True

    predictor = DummyPredictor()
    service.models = {
        "dummy": {
            "predictor": predictor,
            "config": {"name": "Dummy", "type": "mock"},
            "loaded": True,
        }
    }

    def raise_signature_error(_):
        raise ValueError("no signature available")

    monkeypatch.setattr(model_service_module.inspect, "signature", raise_signature_error)

    result = asyncio.run(service.predict("dummy", "scan.nii.gz", "analysis-1"))

    assert result["ok"] is True
    assert result["analysis_id"] == "analysis-1"
    assert predictor.calls == [("scan.nii.gz", "analysis-1")]
