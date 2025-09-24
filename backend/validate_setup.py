#!/usr/bin/env python3
"""
Backend Setup Validation Script
===============================
Validates that all required dependencies are properly installed
and models can be loaded successfully.
"""

import sys
import importlib
import subprocess
from pathlib import Path
import pkg_resources
from typing import Dict, List, Tuple

def check_python_version() -> bool:
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major != 3 or version.minor < 8:
        print(f"❌ Python {version.major}.{version.minor} detected. Python 3.8+ required.")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
    return True

def check_required_packages() -> Tuple[bool, List[str]]:
    """Check if all required packages are installed"""
    
    # Critical packages that must be installed
    critical_packages = [
        'torch', 'torchvision', 'transformers', 'ultralytics',
        'fastapi', 'uvicorn', 'opencv-python', 'numpy', 'pillow'
    ]
    
    # Optional but recommended packages
    recommended_packages = [
        'insightface', 'onnxruntime', 'mtcnn', 'facenet-pytorch',
        'face-recognition', 'scikit-learn', 'networkx', 'dateparser'
    ]
    
    missing_critical = []
    missing_recommended = []
    
    print("\n🔍 Checking critical packages...")
    for package in critical_packages:
        try:
            importlib.import_module(package.replace('-', '_'))
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            missing_critical.append(package)
    
    print("\n🔍 Checking recommended packages...")
    for package in recommended_packages:
        try:
            importlib.import_module(package.replace('-', '_'))
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ⚠️  {package} (optional)")
            missing_recommended.append(package)
    
    return len(missing_critical) == 0, missing_critical + missing_recommended

def check_gpu_support() -> Dict[str, bool]:
    """Check GPU support for PyTorch and ONNX"""
    gpu_info = {
        'pytorch_cuda': False,
        'pytorch_mps': False,  # Apple Silicon
        'onnx_gpu': False
    }
    
    try:
        import torch
        gpu_info['pytorch_cuda'] = torch.cuda.is_available()
        if gpu_info['pytorch_cuda']:
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ CUDA GPU detected: {gpu_name}")
        
        # Check for Apple Silicon GPU support
        gpu_info['pytorch_mps'] = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        if gpu_info['pytorch_mps']:
            print("✅ Apple Silicon GPU (MPS) support available")
        
        if not gpu_info['pytorch_cuda'] and not gpu_info['pytorch_mps']:
            print("ℹ️  No GPU acceleration available - using CPU")
            
    except ImportError:
        print("❌ PyTorch not installed - cannot check GPU support")
    
    try:
        import onnxruntime
        providers = onnxruntime.get_available_providers()
        gpu_info['onnx_gpu'] = 'CUDAExecutionProvider' in providers or 'CoreMLExecutionProvider' in providers
        print(f"ℹ️  ONNX providers: {', '.join(providers)}")
    except ImportError:
        print("⚠️  ONNX Runtime not installed")
    
    return gpu_info

def test_model_loading() -> Dict[str, bool]:
    """Test loading of critical AI models"""
    model_status = {
        'clip': False,
        'yolo': False,
        'insightface': False
    }
    
    print("\n🤖 Testing model loading...")
    
    # Test CLIP model
    try:
        from transformers import CLIPModel, CLIPProcessor
        model = CLIPModel.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K")
        processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K")
        print("  ✅ CLIP model loaded successfully")
        model_status['clip'] = True
        del model, processor  # Free memory
    except Exception as e:
        print(f"  ❌ CLIP model failed: {e}")
    
    # Test YOLO model
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8x.pt')  # This will download if not present
        print("  ✅ YOLOv8x model loaded successfully")
        model_status['yolo'] = True
        del model  # Free memory
    except Exception as e:
        print(f"  ❌ YOLO model failed: {e}")
    
    # Test InsightFace
    try:
        import insightface
        app = insightface.app.FaceAnalysis(providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        print("  ✅ InsightFace model loaded successfully")
        model_status['insightface'] = True
        del app  # Free memory
    except Exception as e:
        print(f"  ⚠️  InsightFace model failed: {e}")
    
    return model_status

def test_api_components() -> bool:
    """Test that API components can be imported"""
    print("\n🌐 Testing API components...")
    
    try:
        from fastapi import FastAPI
        from uvicorn import Config
        from photo_database import PhotoDatabase
        from final_photo_search import UltimatePhotoSearcher
        print("  ✅ All API components imported successfully")
        return True
    except Exception as e:
        print(f"  ❌ API component import failed: {e}")
        return False

def generate_recommendations(missing_packages: List[str], gpu_info: Dict[str, bool], 
                           model_status: Dict[str, bool]) -> None:
    """Generate setup recommendations"""
    print("\n💡 Setup Recommendations:")
    
    if missing_packages:
        print(f"\n📦 Install missing packages:")
        print(f"   pip install {' '.join(missing_packages)}")
    
    if not gpu_info.get('pytorch_cuda', False) and not gpu_info.get('pytorch_mps', False):
        if sys.platform.startswith('darwin'):  # macOS
            print("\n🍎 For Apple Silicon GPU support:")
            print("   pip install torch torchvision torchaudio")
        else:
            print("\n🚀 For NVIDIA GPU support:")
            print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    
    if not model_status.get('insightface', False):
        print("\n👤 Face detection fallback options:")
        print("   - Install InsightFace: pip install insightface")
        print("   - Or use MTCNN: pip install mtcnn")
        print("   - Or use face_recognition: pip install face-recognition")
    
    if not model_status.get('clip', False):
        print("\n🔍 CLIP model issues:")
        print("   - Check internet connection for model download")
        print("   - Ensure sufficient disk space (~1GB)")
    
    if not model_status.get('yolo', False):
        print("\n🎯 YOLO model issues:")
        print("   - Check internet connection for model download")
        print("   - Ensure sufficient memory (8GB+ recommended)")

def main():
    """Run complete setup validation"""
    print("🚀 Backend Setup Validation")
    print("=" * 50)
    
    # Check Python version
    python_ok = check_python_version()
    if not python_ok:
        print("\n❌ Setup validation failed - upgrade Python first")
        return False
    
    # Check packages
    packages_ok, missing_packages = check_required_packages()
    
    # Check GPU support
    gpu_info = check_gpu_support()
    
    # Test model loading (only if basic packages are available)
    model_status = {}
    if packages_ok:
        model_status = test_model_loading()
    
    # Test API components
    api_ok = test_api_components()
    
    # Generate summary
    print("\n" + "=" * 50)
    print("📊 VALIDATION SUMMARY")
    print("=" * 50)
    
    total_score = 0
    max_score = 0
    
    # Core requirements
    if python_ok:
        print("✅ Python version: Compatible")
        total_score += 2
    else:
        print("❌ Python version: Incompatible")
    max_score += 2
    
    if packages_ok:
        print("✅ Required packages: All installed")
        total_score += 3
    else:
        print(f"⚠️  Required packages: {len(missing_packages)} missing")
        total_score += 1
    max_score += 3
    
    if api_ok:
        print("✅ API components: Working")
        total_score += 2
    else:
        print("❌ API components: Failed")
    max_score += 2
    
    # Model status
    models_loaded = sum(model_status.values())
    total_models = len(model_status)
    if total_models > 0:
        print(f"🤖 AI models: {models_loaded}/{total_models} loaded")
        total_score += models_loaded
        max_score += total_models
    
    # GPU acceleration
    if gpu_info.get('pytorch_cuda') or gpu_info.get('pytorch_mps'):
        print("🚀 GPU acceleration: Available")
        total_score += 1
    else:
        print("💻 GPU acceleration: Not available (CPU only)")
    max_score += 1
    
    print(f"\n📈 Overall Score: {total_score}/{max_score} ({total_score/max_score*100:.1f}%)")
    
    if total_score >= max_score * 0.8:
        print("🎉 Setup validation PASSED - Ready for deployment!")
        success = True
    elif total_score >= max_score * 0.6:
        print("⚠️  Setup validation PARTIAL - Some features may be limited")
        success = True
    else:
        print("❌ Setup validation FAILED - Critical issues need resolution")
        success = False
    
    # Generate recommendations
    if missing_packages or not all(model_status.values()):
        generate_recommendations(missing_packages, gpu_info, model_status)
    
    print(f"\n🔧 To start the backend server: python start_api.py")
    print(f"📚 For detailed setup guide: see SETUP.md")
    
    return success

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n🛑 Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Validation failed with error: {e}")
        sys.exit(1)