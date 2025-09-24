# Backend Setup Guide

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Start the API Server

```bash
python start_api.py
```

The API will be available at `http://localhost:8000`

## 📋 System Requirements

- **Python**: 3.8+ (3.9+ recommended)
- **Memory**: 8GB RAM minimum (16GB recommended for large photo collections)
- **Storage**: 2GB free space for models and dependencies
- **OS**: Windows 10+, macOS 10.15+, or Linux (Ubuntu 18.04+)

## 🤖 AI Models & Dependencies

### Core Models (Auto-downloaded on first use)

1. **YOLOv8x/YOLOv10x** (~136MB)
   - Object Detection: 80+ categories
   - High accuracy general-purpose detection
   - GPU acceleration supported

2. **InsightFace Buffalo_L** (~400MB)
   - State-of-the-art face detection and recognition
   - 512-dimensional face embeddings
   - Age and gender estimation

3. **LAION CLIP** (~350MB)
   - Semantic image-text understanding
   - Natural language photo search
   - Cross-modal similarity matching

**Total initial download**: ~1GB

### Model Storage Locations

- **YOLO models**: `./yolov8x.pt`, `./yolov10x.pt`
- **InsightFace**: `~/.insightface/models/`
- **CLIP**: `~/.cache/huggingface/transformers/`

## 🔧 Platform-Specific Setup

### Windows

```bash
# Install Visual Studio Build Tools for face_recognition
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Install Python dependencies
pip install -r requirements.txt
```

### macOS (Intel)

```bash
# Install dependencies
pip install -r requirements.txt
```

### macOS (Apple Silicon)

```bash
# Use conda for dlib (recommended)
conda install -c conda-forge dlib
pip install -r requirements.txt

# Or install from source (slower)
pip install cmake
pip install dlib
pip install -r requirements.txt
```

### Linux (Ubuntu/Debian)

```bash
# Install system dependencies
sudo apt update
sudo apt install python3-dev python3-pip cmake libopenblas-dev liblapack-dev

# Install Python dependencies
pip install -r requirements.txt
```

## 🚀 Performance Optimization

### GPU Acceleration (Recommended)

For NVIDIA GPUs, install PyTorch with CUDA support:

```bash
# CUDA 11.8 (most compatible)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1 (latest)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### YOLO Performance Profiles

The system automatically detects your hardware and selects optimal settings:

- **GPU (High-end)**: Maximum accuracy mode with YOLOv10x
- **GPU (Mid-range)**: Accuracy mode with YOLOv8x
- **CPU only**: Balanced mode with optimized settings

You can override this in `yolo_performance_config.py`.

## 🗃️ Database & Storage

### SQLite Database

- **Location**: `./photos.db`
- **Purpose**: Photo metadata, embeddings, and search indices
- **Backup**: Automatically creates `.backup` files

### Photo Processing

The system processes photos in the following order:
1. **EXIF extraction** (date, location, camera info)
2. **CLIP embeddings** (semantic understanding)
3. **Object detection** (YOLO - people, objects, scenes)
4. **Face detection** (InsightFace - faces, age, gender)
5. **Relationship mapping** (person clustering and relationships)

## 🔍 API Endpoints

### Core Endpoints

- `GET /photos` - List all photos with metadata
- `POST /search` - Smart search with natural language
- `POST /upload` - Upload and process new photos
- `GET /people` - List detected people/faces
- `POST /auto-index` - Start automatic photo monitoring

### Advanced Features

- `POST /search/people` - Search by person/face
- `POST /search/temporal` - Time-based search ("last Christmas", "2023")
- `GET /relationships` - Person relationship graph
- `POST /cluster-faces` - Re-cluster faces with new parameters

## 🛠️ Development & Testing

### Run Tests

```bash
# Unit tests
python -m pytest backend/test_*.py

# API integration tests
python backend/test_api_integration.py

# Performance benchmarks
python backend/test_performance.py
```

### Development Mode

```bash
# Start with auto-reload
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload

# Or use the development script
python start_api.py --dev
```

## 🚨 Troubleshooting

### Common Issues

#### ModuleNotFoundError: No module named 'dlib'

**Windows**:
```bash
# Install Visual Studio Build Tools first
pip install cmake
pip install dlib
```

**macOS**:
```bash
# Use conda (recommended)
conda install -c conda-forge dlib

# Or install with homebrew
brew install cmake
pip install dlib
```

#### CUDA Out of Memory

Reduce batch sizes in `yolo_performance_config.py`:
```python
# Change to speed profile for lower memory usage
DEFAULT_PROFILE = "speed"
```

#### InsightFace Model Download Fails

```bash
# Manual model download
python -c "import insightface; app = insightface.app.FaceAnalysis(); app.prepare(ctx_id=-1)"
```

#### Port 8000 Already in Use

```bash
# Find process using port 8000
# Windows
netstat -ano | findstr :8000

# macOS/Linux  
lsof -i :8000

# Start on different port
uvicorn api_server:app --host 0.0.0.0 --port 8001
```

### Performance Issues

#### Slow Face Detection

1. Ensure InsightFace is using the correct model:
   ```python
   # Check available providers
   import onnxruntime
   print(onnxruntime.get_available_providers())
   ```

2. For CPU-only systems, consider using MTCNN instead of InsightFace.

#### Slow Object Detection

1. Switch to faster YOLO profile:
   ```python
   # In yolo_performance_config.py
   DEFAULT_PROFILE = "speed"  # or "balanced"
   ```

2. For CPU-only systems, consider reducing `max_objects` parameter.

## 📚 Dependencies Overview

### Core AI/ML Stack
- **PyTorch** (2.0+): Deep learning framework
- **Transformers** (4.30+): CLIP model support  
- **Ultralytics** (8.0+): YOLOv8/v10 framework
- **InsightFace** (0.7.3+): Face analysis toolkit
- **OpenCV** (4.8+): Computer vision utilities

### Web Framework
- **FastAPI** (0.104+): Modern Python web framework
- **Uvicorn** (0.24+): ASGI server
- **Pydantic** (2.4+): Data validation

### Data Processing
- **NumPy** (1.24+): Numerical computing
- **Pandas** (2.0+): Data manipulation
- **Scikit-learn** (1.3+): Machine learning utilities
- **NetworkX** (3.1+): Graph analysis for relationships

## 🔒 Privacy & Security

### On-Device Processing
- All AI models run locally on your device
- Face embeddings and biometric data never leave your system
- No cloud processing or data transmission

### Data Protection
- Face recognition requires explicit user consent
- Biometric data can be deleted at any time
- Relationship mapping is opt-in only

### File Access
- Only accesses specified photo directories
- Respects file system permissions
- No modification of original photos

## 📈 Monitoring & Logging

### Log Files
- **Application logs**: `./logs/app.log`
- **Error logs**: `./logs/error.log`
- **Performance logs**: `./logs/performance.log`

### Metrics
- Processing speed (photos/second)
- Model inference times
- Memory usage statistics
- API response times

---

For more information, see the main [README.md](../README.md) or visit the project documentation.