import os
import asyncio
import sys
from pathlib import Path
import logging

# Add the project root to the path so we can import the model service
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from backend.services.model_service import model_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_inference():
    logger.info("Initializing model service...")
    await model_service.initialize()
    
    dataset_path = Path("test_dataset/Testing")
    if not dataset_path.exists():
        logger.error(f"Dataset path {dataset_path} not found. Please ensure brain-tumor-mri-dataset.zip is extracted.")
        return

    # Test cases: one from each category
    test_categories = ["glioma", "meningioma", "notumor", "pituitary"]
    models_to_test = ["ensemble", "medical_vit", "yolov8", "nnunet"]

    for category in test_categories:
        cat_path = dataset_path / category
        images = list(cat_path.glob("*.jpg"))
        if not images:
            logger.warning(f"No images found in category {category}")
            continue
        
        test_image = images[0]
        logger.info(f"\n--- Testing category: {category.upper()} (Image: {test_image.name}) ---")
        
        for model_id in models_to_test:
            logger.info(f"Running inference with model: {model_id}")
            try:
                result = await model_service.predict(model_id, str(test_image), "test_analysis_id")
                
                pred = result["predictions"]
                logger.info(f"  Result: {'TUMOR DETECTED' if pred['tumor_detected'] else 'NO TUMOR'}")
                logger.info(f"  Type: {pred['tumor_type']}")
                logger.info(f"  Confidence: {pred['confidence']:.4f}")
                if "tumor_volume_ml" in pred:
                    logger.info(f"  Volume: {pred['tumor_volume_ml']} mL")
                if "location" in pred:
                    logger.info(f"  Location: {pred['location']}")
                
            except Exception as e:
                logger.error(f"  Error during inference with {model_id}: {e}")

if __name__ == "__main__":
    asyncio.run(test_inference())
