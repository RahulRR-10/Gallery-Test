#!/usr/bin/env python3
"""
System Requirements Checker
===========================
Verifies that all dependencies are installed correctly
"""

import sys

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} (Need 3.8+)")
        return False

def check_dependencies():
    """Check all required dependencies"""
    dependencies = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('ultralytics', 'YOLO'),
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('matplotlib', 'Matplotlib'),
        ('PIL', 'Pillow'),
        ('requests', 'Requests'),
        ('sqlite3', 'SQLite3'),
    ]
    
    optional_deps = [
        ('face_recognition', 'Face Recognition (optional)'),
    ]
    
    print("\n📦 Checking Core Dependencies:")
    all_good = True
    
    for module, name in dependencies:
        try:
            __import__(module)
            if module == 'torch':
                import torch
                device = "GPU" if torch.cuda.is_available() else "CPU"
                print(f"✅ {name} - {device} available")
            else:
                print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - Missing")
            all_good = False
    
    print("\n🔧 Checking Optional Dependencies:")
    for module, name in optional_deps:
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"⚠️ {name} - Not installed (will use fallback)")
    
    return all_good

def check_system_ready():
    """Check if system is ready to run"""
    print("\n🧪 Testing System Integration:")
    
    try:
        from final_photo_search import UltimatePhotoSearcher
        print("✅ Photo Search System")
    except Exception as e:
        print(f"❌ Photo Search System - {e}")
        return False
    
    try:
        from clip_model import CLIPEmbeddingExtractor
        print("✅ CLIP Model")
    except Exception as e:
        print(f"❌ CLIP Model - {e}")
        return False
    
    try:
        from photo_database import PhotoDatabase
        print("✅ Database System")
    except Exception as e:
        print(f"❌ Database System - {e}")
        return False
    
    return True

def main():
    """Main checker function"""
    print("🔍 Ultimate Photo Search System - Requirements Checker")
    print("=" * 60)
    
    # Check Python version
    python_ok = check_python_version()
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    # Check system integration
    system_ok = check_system_ready()
    
    print("\n" + "=" * 60)
    
    if python_ok and deps_ok and system_ok:
        print("🎉 System Ready!")
        print("✅ All requirements satisfied")
        print("🚀 You can now run: python demo_photo_search.py")
    else:
        print("⚠️ Setup Issues Detected")
        if not python_ok:
            print("📥 Please upgrade to Python 3.8+")
        if not deps_ok:
            print("📥 Please install missing dependencies:")
            print("   pip install -r requirements.txt")
        if not system_ok:
            print("📥 Please check system integration")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
