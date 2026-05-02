import os
import torch
import logging
from pathlib import Path
from transformers import ViTForImageClassification, ViTImageProcessor
from huggingface_hub import hf_hub_download
import monai
from monai.networks.nets import UNet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_models():
    models_dir = Path("models/saved")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("--- Pulling Real AI Models ---")
    
    # 1. Classification: ViT (Hugging Face)
    try:
        logger.info("Pulling Classification Model: Hemgg/brain-tumor-classification")
        model_id = "Hemgg/brain-tumor-classification"
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        AutoImageProcessor.from_pretrained(model_id)
        AutoModelForImageClassification.from_pretrained(model_id)
        logger.info("✅ Classification model ready (Transformers cache)")
    except Exception as e:
        logger.error(f"Failed to pull classification model: {e}")

    # 2. Segmentation: 3D U-Net (MONAI / BraTS)
    try:
        logger.info("Initializing Segmentation Model: 3D U-Net (MONAI)")
        # We can use MONAI's pre-trained weights for BraTS if available, 
        # otherwise we instantiate the architecture and look for weights.
        # For now, let's ensure we can at least load the architecture.
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=4,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
        # Try to download a generic pre-trained weight if possible, or use a known public one
        # Note: BraTS models are often large.
        logger.info("✅ Segmentation architecture initialized")
    except Exception as e:
        logger.error(f"Failed to initialize segmentation model: {e}")

    # 3. Detection & Segmentation: YOLOv8 (Ultralytics)
    try:
        from ultralytics import YOLO
        logger.info("Pulling Segmentation Model: YOLOv8-seg")
        # This will download the weights automatically
        model = YOLO("yolov8n-seg.pt") 
        logger.info("✅ Segmentation model ready (YOLOv8-seg)")
    except Exception as e:
        logger.error(f"Failed to pull segmentation model: {e}")

    logger.info("--- Model Pulling Complete ---")

if __name__ == "__main__":
    download_models()
