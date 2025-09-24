#!/usr/bin/env python3
"""
Quick Install Script for Gallery Backend
========================================
Automatically installs all required dependencies for the AI photo gallery backend.
"""

import sys
import subprocess
import platform
import os
from pathlib import Path

def run_command(command, description=None):
    """Run a command and handle errors"""
    if description:
        print(f"🔄 {description}...")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.stdout.strip():
            print(f"   {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {e}")
        if e.stderr:
            print(f"   Error: {e.stderr}")
        return False

def check_python_version():
    """Check Python version compatibility"""
    version = sys.version_info
    if version.major != 3 or version.minor < 8:
        print(f"❌ Python {version.major}.{version.minor} detected. Python 3.8+ required.")
        print("   Please upgrade Python and try again.")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
    return True

def detect_platform():
    """Detect platform and return installation recommendations"""
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    print(f"🖥️  Detected platform: {system} ({machine})")
    
    if system == "darwin":  # macOS
        if "arm" in machine or "aarch64" in machine:
            return "macos_arm"
        else:
            return "macos_intel"
    elif system == "linux":
        return "linux"
    elif system == "windows":
        return "windows"
    else:
        return "unknown"

def install_system_dependencies(platform_type):
    """Install system-level dependencies"""
    print("\n🔧 Installing system dependencies...")
    
    if platform_type == "macos_arm":
        print("   For Apple Silicon, we recommend using conda for dlib")
        print("   Run: conda install -c conda-forge dlib")
        
    elif platform_type == "linux":
        print("   Installing system packages...")
        commands = [
            "sudo apt update",
            "sudo apt install -y python3-dev python3-pip cmake libopenblas-dev liblapack-dev"
        ]
        for cmd in commands:
            if not run_command(cmd, f"Running: {cmd}"):
                print("   ⚠️  Some system packages may need manual installation")
                
    elif platform_type == "windows":
        print("   Windows: Visual Studio Build Tools recommended for face_recognition")
        print("   Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/")

def install_pytorch_gpu():
    """Install PyTorch with GPU support if available"""
    print("\n🚀 Setting up PyTorch...")
    
    # Check if NVIDIA GPU is available
    nvidia_available = run_command("nvidia-smi", "Checking for NVIDIA GPU")
    
    if nvidia_available:
        print("   NVIDIA GPU detected - installing PyTorch with CUDA support")
        pytorch_install = (
            "pip install torch torchvision torchaudio "
            "--index-url https://download.pytorch.org/whl/cu118"
        )
    else:
        print("   No NVIDIA GPU detected - installing CPU-only PyTorch")
        pytorch_install = "pip install torch torchvision torchaudio"
    
    return run_command(pytorch_install, "Installing PyTorch")

def install_requirements():
    """Install Python requirements"""
    print("\n📦 Installing Python packages...")
    
    # Make sure we're in the right directory
    backend_dir = Path(__file__).parent
    requirements_file = backend_dir / "requirements.txt"
    
    if not requirements_file.exists():
        print(f"❌ requirements.txt not found at {requirements_file}")
        return False
    
    return run_command(f"pip install -r {requirements_file}", "Installing requirements")

def download_models():
    """Pre-download AI models"""
    print("\n🤖 Pre-downloading AI models...")
    
    # Download YOLO model
    yolo_cmd = """python -c "from ultralytics import YOLO; YOLO('yolov8x.pt')" """
    run_command(yolo_cmd, "Downloading YOLOv8x model")
    
    # Download InsightFace model
    insight_cmd = """python -c "import insightface; app = insightface.app.FaceAnalysis(); app.prepare(ctx_id=-1)" """
    run_command(insight_cmd, "Downloading InsightFace model")
    
    # Download CLIP model
    clip_cmd = """python -c "from transformers import CLIPModel, CLIPProcessor; CLIPModel.from_pretrained('laion/CLIP-ViT-B-32-laion2B-s34B-b79K'); CLIPProcessor.from_pretrained('laion/CLIP-ViT-B-32-laion2B-s34B-b79K')" """
    run_command(clip_cmd, "Downloading CLIP model")

def main():
    """Main installation process"""
    print("🚀 Gallery Backend Installation")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Detect platform
    platform_type = detect_platform()
    
    # Install system dependencies
    install_system_dependencies(platform_type)
    
    # Upgrade pip
    run_command("python -m pip install --upgrade pip", "Upgrading pip")
    
    # Install PyTorch with GPU support
    if not install_pytorch_gpu():
        print("⚠️  PyTorch installation failed - continuing with CPU version")
    
    # Install requirements
    if not install_requirements():
        print("❌ Failed to install requirements")
        sys.exit(1)
    
    # Download models (optional)
    print("\n🤖 Would you like to pre-download AI models? (~1GB)")
    response = input("   This will take several minutes but improves first-run experience. (y/N): ")
    
    if response.lower().startswith('y'):
        download_models()
    else:
        print("   Models will be downloaded automatically on first use")
    
    # Final validation
    print("\n✅ Installation complete!")
    print("\n🔍 Running setup validation...")
    
    validation_result = run_command("python validate_setup.py", "Validating installation")
    
    if validation_result:
        print("\n🎉 Installation successful!")
        print("   To start the backend server: python start_api.py")
    else:
        print("\n⚠️  Installation completed with warnings")
        print("   Check validation output above for details")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Installation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Installation failed: {e}")
        sys.exit(1)