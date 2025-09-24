#!/usr/bin/env python3
"""
YOLO Model Performance Configuration
==================================
Choose between speed and accuracy based on your needs
"""

# Performance profiles for different use cases
YOLO_PERFORMANCE_PROFILES = {
    "speed": {
        "description": "Fastest inference, good for real-time processing",
        "models": ["yolov8n.pt", "yolov10n.pt"],  # Nano models
        "confidence_threshold": 0.35,
        "max_objects": 8,
        "expected_fps_cpu": 20,
        "expected_fps_gpu": 120
    },
    
    "balanced": {
        "description": "Good balance of speed and accuracy",
        "models": ["yolov8m.pt", "yolov10m.pt"],  # Medium models
        "confidence_threshold": 0.3,
        "max_objects": 10,
        "expected_fps_cpu": 12,
        "expected_fps_gpu": 80
    },
    
    "accuracy": {
        "description": "Best accuracy, slower inference",
        "models": ["yolov8x.pt", "yolov10x.pt"],  # Extra large models
        "confidence_threshold": 0.25,
        "max_objects": 15,
        "expected_fps_cpu": 8,
        "expected_fps_gpu": 50
    },
    
    "maximum": {
        "description": "Maximum accuracy, slowest inference",
        "models": ["yolov10x.pt", "yolov8x.pt", "yolov8x-seg.pt"],
        "confidence_threshold": 0.2,
        "max_objects": 20,
        "expected_fps_cpu": 5,
        "expected_fps_gpu": 35
    }
}

# Default profile - you can change this
DEFAULT_PROFILE = "accuracy"

def get_profile_config(profile_name: str = None) -> dict:
    """Get configuration for a specific performance profile"""
    if profile_name is None:
        profile_name = DEFAULT_PROFILE
    
    return YOLO_PERFORMANCE_PROFILES.get(profile_name, YOLO_PERFORMANCE_PROFILES[DEFAULT_PROFILE])

def detect_optimal_profile():
    """Auto-detect optimal profile based on system capabilities"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            if "RTX 40" in gpu_name or "RTX 30" in gpu_name:
                return "maximum"  # High-end GPU
            elif "GTX" in gpu_name or "RTX 20" in gpu_name:
                return "accuracy"  # Mid-range GPU
            else:
                return "balanced"  # Entry-level GPU
        else:
            return "speed"  # CPU only
    except:
        return "balanced"  # Safe default

# Model size and performance comparison
YOLO_MODEL_SPECS = {
    "yolov8n.pt": {"size_mb": 6, "params_m": 3.2, "accuracy_map50": 37.3, "speed_factor": 1.0},
    "yolov8s.pt": {"size_mb": 22, "params_m": 11.2, "accuracy_map50": 44.9, "speed_factor": 0.7},
    "yolov8m.pt": {"size_mb": 52, "params_m": 25.9, "accuracy_map50": 50.2, "speed_factor": 0.5},
    "yolov8l.pt": {"size_mb": 87, "params_m": 43.7, "accuracy_map50": 52.9, "speed_factor": 0.3},
    "yolov8x.pt": {"size_mb": 131, "params_m": 68.2, "accuracy_map50": 53.9, "speed_factor": 0.2},
    
    "yolov10n.pt": {"size_mb": 5, "params_m": 2.3, "accuracy_map50": 38.5, "speed_factor": 1.1},
    "yolov10s.pt": {"size_mb": 16, "params_m": 7.2, "accuracy_map50": 46.3, "speed_factor": 0.8},
    "yolov10m.pt": {"size_mb": 34, "params_m": 15.4, "accuracy_map50": 51.1, "speed_factor": 0.6},
    "yolov10l.pt": {"size_mb": 58, "params_m": 24.4, "accuracy_map50": 53.2, "speed_factor": 0.4},
    "yolov10x.pt": {"size_mb": 91, "params_m": 29.5, "accuracy_map50": 54.4, "speed_factor": 0.3}
}

def print_performance_comparison():
    """Print performance comparison of different models"""
    print("\n🚀 YOLO Model Performance Comparison")
    print("=" * 70)
    print(f"{'Model':<12} {'Size':<8} {'Params':<8} {'mAP50':<8} {'Speed':<8}")
    print("-" * 70)
    
    for model, specs in YOLO_MODEL_SPECS.items():
        speed_desc = "Fast" if specs['speed_factor'] >= 0.8 else "Medium" if specs['speed_factor'] >= 0.4 else "Slow"
        print(f"{model:<12} {specs['size_mb']:<7}MB {specs['params_m']:<7}M {specs['accuracy_map50']:<7}% {speed_desc:<8}")
    
    print("\n📊 Recommendations:")
    print("• yolov10n/yolov8n: Real-time applications (webcams, live processing)")  
    print("• yolov10m/yolov8m: Balanced photo indexing (recommended)")
    print("• yolov10x/yolov8x: Maximum accuracy photo search (your current choice)")
    print("• yolov10x: Best overall choice (better accuracy than yolov8x)")

if __name__ == "__main__":
    print_performance_comparison()
    print(f"\n🎯 Recommended profile for your system: {detect_optimal_profile()}")
    print(f"🛠️ Current default profile: {DEFAULT_PROFILE}")