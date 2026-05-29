import time
import os
import sys
import psutil
from pathlib import Path
import numpy as np
from PIL import Image
import torch

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent / "backend"))

from transformers import AutoImageProcessor, AutoModelForImageClassification
from ultralytics import YOLO
import onnxruntime as ort

def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024) # in MB

def benchmark_vit(num_runs=30):
    print("\n=== Benchmarking ViT (Hemgg/brain-tumor-classification) ===")
    model_id = "Hemgg/brain-tumor-classification"
    
    # Preprocessor
    processor = AutoImageProcessor.from_pretrained(model_id)
    dummy_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    
    # 1. PyTorch Baseline
    mem_before_pt = get_process_memory()
    pt_model = AutoModelForImageClassification.from_pretrained(model_id)
    mem_after_pt = get_process_memory()
    pt_load_mem = mem_after_pt - mem_before_pt
    
    pt_inputs = processor(images=dummy_img, return_tensors="pt")
    
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = pt_model(**pt_inputs)
            
    # Benchmark PyTorch
    pt_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        with torch.no_grad():
            _ = pt_model(**pt_inputs)
        pt_times.append(time.perf_counter() - start)
        
    avg_pt = np.mean(pt_times) * 1000
    print(f"PyTorch: Average Inference Time = {avg_pt:.2f} ms | Load Memory = {pt_load_mem:.2f} MB")
    
    # Clean PyTorch model from memory
    del pt_model
    import gc
    gc.collect()
    
    # 2. ONNX optimized
    onnx_path = Path(__file__).parent.parent / "vit_model.onnx"
    if not onnx_path.exists():
        print(f"ONNX model not found at {onnx_path}, exporting...")
        
    mem_before_onnx = get_process_memory()
    onnx_session = ort.InferenceSession(str(onnx_path))
    mem_after_onnx = get_process_memory()
    onnx_load_mem = mem_after_onnx - mem_before_onnx
    
    onnx_inputs = {onnx_session.get_inputs()[0].name: processor(images=dummy_img, return_tensors="np")["pixel_values"]}
    
    # Warmup
    for _ in range(5):
        _ = onnx_session.run(None, onnx_inputs)
        
    # Benchmark ONNX
    onnx_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = onnx_session.run(None, onnx_inputs)
        onnx_times.append(time.perf_counter() - start)
        
    avg_onnx = np.mean(onnx_times) * 1000
    speedup = avg_pt / avg_onnx
    mem_reduction = pt_load_mem - onnx_load_mem
    print(f"ONNX:    Average Inference Time = {avg_onnx:.2f} ms | Load Memory = {onnx_load_mem:.2f} MB")
    print(f"ViT Results: Speedup = {speedup:.2f}x | Memory Saving = {mem_reduction:.2f} MB")
    
    return {
        "model": "ViT Classification",
        "pt_latency": avg_pt,
        "onnx_latency": avg_onnx,
        "pt_memory": pt_load_mem,
        "onnx_memory": onnx_load_mem,
        "speedup": speedup
    }

def benchmark_yolo(model_name="yolov8n", num_runs=30):
    print(f"\n=== Benchmarking YOLO ({model_name}) ===")
    pt_file = f"{model_name}.pt"
    onnx_file = f"{model_name}.onnx"
    
    # 1. PyTorch Baseline
    mem_before_pt = get_process_memory()
    pt_model = YOLO(pt_file)
    mem_after_pt = get_process_memory()
    pt_load_mem = mem_after_pt - mem_before_pt
    
    dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # Warmup
    for _ in range(5):
        _ = pt_model(dummy_img, verbose=False)
        
    # Benchmark PyTorch
    pt_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = pt_model(dummy_img, verbose=False)
        pt_times.append(time.perf_counter() - start)
        
    avg_pt = np.mean(pt_times) * 1000
    print(f"PyTorch: Average Inference Time = {avg_pt:.2f} ms | Load Memory = {pt_load_mem:.2f} MB")
    
    # Clean PyTorch model from memory
    del pt_model
    import gc
    gc.collect()
    
    # 2. ONNX optimized
    mem_before_onnx = get_process_memory()
    onnx_model = YOLO(onnx_file)
    mem_after_onnx = get_process_memory()
    onnx_load_mem = mem_after_onnx - mem_before_onnx
    
    # Warmup
    for _ in range(5):
        _ = onnx_model(dummy_img, verbose=False)
        
    # Benchmark ONNX
    onnx_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = onnx_model(dummy_img, verbose=False)
        onnx_times.append(time.perf_counter() - start)
        
    avg_onnx = np.mean(onnx_times) * 1000
    speedup = avg_pt / avg_onnx
    mem_reduction = pt_load_mem - onnx_load_mem
    print(f"ONNX:    Average Inference Time = {avg_onnx:.2f} ms | Load Memory = {onnx_load_mem:.2f} MB")
    print(f"YOLO {model_name} Results: Speedup = {speedup:.2f}x | Memory Saving = {mem_reduction:.2f} MB")
    
    return {
        "model": f"YOLO {model_name}",
        "pt_latency": avg_pt,
        "onnx_latency": avg_onnx,
        "pt_memory": pt_load_mem,
        "onnx_memory": onnx_load_mem,
        "speedup": speedup
    }

def main():
    print("=" * 80)
    print("      NEUROSCAN INFERENCE OPTIMIZATION: PYTORCH VS ONNX BENCHMARK")
    print("=" * 80)
    
    vit_res = benchmark_vit()
    yolo_res = benchmark_yolo("yolov8n")
    yolo_seg_res = benchmark_yolo("yolov8n-seg")
    
    print("\n" + "=" * 80)
    print("                           SUMMARY BENCHMARK REPORT")
    print("=" * 80)
    print(f"{'Model Name':<25} | {'PyTorch Latency':<16} | {'ONNX Latency':<14} | {'Speedup':<8} | {'Memory Savings':<14}")
    print("-" * 80)
    for res in [vit_res, yolo_res, yolo_seg_res]:
        mem_saving = res['pt_memory'] - res['onnx_memory']
        print(f"{res['model']:<25} | {res['pt_latency']:>13.2f} ms | {res['onnx_latency']:>11.2f} ms | {res['speedup']:>7.2f}x | {mem_saving:>10.2f} MB")
    print("=" * 80)

if __name__ == "__main__":
    main()
