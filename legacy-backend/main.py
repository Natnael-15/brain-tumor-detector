"""
Main entry point for the Brain MRI Tumor Detector application.
"""

import argparse
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from data.preprocess import MRIPreprocessor
from training.train import ModelTrainer
from inference.predict import TumorPredictor
from visualization.viewer import BrainViewer
from reports.generator import ReportGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description="Brain MRI Tumor Detector")
    parser.add_argument("--mode", choices=["preprocess", "train", "predict", "visualize", "report", "web"], 
                       default="web", help="Operation mode")
    parser.add_argument("--input", type=str, help="Input file or directory path")
    parser.add_argument("--output", type=str, help="Output file or directory path")
    parser.add_argument("--model", type=str, help="Model file path")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Configuration file")
    
    args = parser.parse_args()
    
    try:
        if args.mode == "preprocess":
            logger.info("Starting MRI data preprocessing...")
            preprocessor = MRIPreprocessor(config_path=args.config)
            preprocessor.process_directory(args.input, args.output)
            
        elif args.mode == "train":
            logger.info("Starting model training...")
            trainer = ModelTrainer(config_path=args.config)
            trainer.train(args.input, args.output)
            
        elif args.mode == "predict":
            logger.info("Starting tumor prediction...")
            predictor = TumorPredictor(model_path=args.model)
            result = predictor.predict(args.input)
            logger.info(f"Prediction result: {result}")
            
        elif args.mode == "visualize":
            logger.info("Starting 3D visualization...")
            viewer = BrainViewer()
            viewer.display(args.input)
            
        elif args.mode == "report":
            logger.info("Generating medical report...")
            generator = ReportGenerator()
            generator.create_report(args.input, args.output)
            
        elif args.mode == "web":
            logger.info("Starting web interface...")
            launch_web_app()
            
    except Exception as e:
        logger.error(f"Error in {args.mode} mode: {str(e)}")
        sys.exit(1)


def launch_web_app():
    """Launch the Streamlit web application."""
    import subprocess
    import os
    
    app_path = Path(__file__).parent / "app.py"
    if app_path.exists():
        subprocess.run(["streamlit", "run", str(app_path)])
    else:
        logger.error("Web app file not found. Please ensure app.py exists.")


if __name__ == "__main__":
    main()