# 🌟 Ultimate AI Photo Search System with FastAPI Backend

A complete AI-powered photo search system with FastAPI REST API backend, designed for mobile app development. Features relationship intelligence, semantic search, and face detection - all running 100% locally on your device.

## 🏗️ Project Architecture

### **📱 Mobile-First Design**
- **FastAPI Backend**: Complete REST API for mobile app integration
- **React Native Ready**: API designed specifically for mobile consumption
- **Background Processing**: Async tasks for indexing and AI processing
- **CORS Enabled**: Cross-origin support for web and mobile clients

### **🧠 AI Intelligence Core**
- **Semantic Search**: CLIP embeddings for natural language understanding
- **Object Detection**: YOLO model detects 80+ object categories  
- **Advanced Face Detection**: InsightFace with age/gender analysis
- **Relationship Intelligence**: AI-powered family/friend detection
- **Temporal Search**: Time-based queries ("last month", "2025")

### **🔗 Advanced Features**
- **Face Clustering**: Automatic person grouping with confidence scoring
- **Group Management**: Organize people into custom groups (family, friends)
- **Relationship Mapping**: Smart inference of connections between people
- **Event Intelligence**: Context-aware photo clustering
- **Multi-Person Search**: Find photos with specific people combinations

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the API Server
```bash
# Start FastAPI server
python start_api.py

# Or with custom settings
python start_api.py --host 0.0.0.0 --port 8000 --reload
```

### 3. Index Your Photos
```bash
# Use the CLI for initial indexing
python final_photo_search.py --directory path/to/photos
```

### 4. Access the API
```bash
# API Documentation
http://localhost:8000/docs

# Basic connection test
curl http://localhost:8000/api/status
```

### 5. Test the API
```bash
# Run comprehensive API tests
python test_api.py

# Test specific endpoints
python test_api.py --test connection
python test_api.py --test search
```

## 📱 API Endpoints

### **Core Endpoints**
- `GET /api/status` - API health and system status
- `GET /api/stats` - Database statistics and metrics
- `POST /api/search` - Semantic photo search with filters
- `GET /api/photos/{photo_id}` - Get specific photo details

### **Face & People Management**
- `GET /api/faces/clusters` - List all face clusters (people)
- `POST /api/faces/clusters/{cluster_id}/label` - Label a person
- `GET /api/groups` - List people groups (family, friends)
- `POST /api/groups` - Create new people groups

### **Relationship Intelligence**
- `GET /api/relationships` - List discovered relationships
- `POST /api/relationships/infer` - Run relationship inference
- `GET /api/relationships/{cluster_id}` - Get relationships for a person

### **Background Tasks**
- `POST /api/index` - Start background photo indexing
- `POST /api/faces/cluster` - Start background face clustering
- `POST /api/relationships/build` - Build relationship mappings

## 🛠️ CLI Commands (For Setup & Management)

### **Initial Setup**
```bash
# Index photos from directory
python final_photo_search.py --directory path/to/photos

# Set up face recognition
python final_photo_search.py --cluster-faces
python final_photo_search.py --list-clusters
python final_photo_search.py --label-person cluster_1 "Alice"

# Build relationships
python final_photo_search.py --build-relationships
python final_photo_search.py --infer-relationships

# Create groups
python final_photo_search.py --create-group "family" cluster_1 cluster_2
```

### **Search & Query**
```bash
# Basic search
python final_photo_search.py --search "vacation beach"
python final_photo_search.py --person "Alice" --search "birthday"
python final_photo_search.py --group "family" --time "last year"
```

### **Advanced Management**
```bash
# Face clustering options
python final_photo_search.py --cluster-faces --cluster-eps 0.35
python final_photo_search.py --backfill-faces
python final_photo_search.py --assign-new-faces

# Relationship building
python final_photo_search.py --build-relationships
python final_photo_search.py --enhanced-relationships
python final_photo_search.py --list-relationship-types

# Export and visualization
python final_photo_search.py --export-relationships
python final_photo_search.py --show-relationship-stats
```

## 🏗️ Project Structure

```
📦 Ultimate AI Photo Search
├── 🌐 FastAPI Backend
│   ├── api_server.py           # Main FastAPI application
│   ├── api_helpers.py          # Database integration helpers
│   ├── start_api.py           # API server startup script
│   └── test_api.py            # Comprehensive API testing
├── 🧠 Core AI Engine
│   ├── final_photo_search.py   # Main search system & CLI
│   ├── clip_model.py          # CLIP embeddings
│   ├── advanced_face_detection.py # Face detection
│   └── temporal_search.py     # Time parsing
├── 🔗 Intelligence Systems
│   ├── relationship_mapping.py # Relationship inference
│   └── photo_database.py     # Database management
├── 📊 Data & Config
│   ├── photos.db             # SQLite database
│   ├── sample_photos/        # Photo directory
│   └── requirements.txt      # Dependencies
└── 📚 Documentation
    ├── README.md             # This file
    ├── API_README.md         # API documentation
    └── prompt.md             # Development context
```

## 🔒 Privacy & Performance

- **100% Local Processing**: All AI runs on your device
- **No Cloud Dependencies**: Complete offline functionality
- **Efficient Storage**: SQLite with optimized embeddings
- **Background Processing**: Non-blocking API operations
- **CORS Enabled**: Mobile and web app ready

## 📱 Mobile Development Ready

The FastAPI backend is specifically designed for mobile app integration:

- **React Native Compatible**: RESTful API design
- **Async Operations**: Background tasks for heavy processing  
- **Proper Error Handling**: Mobile-friendly error responses
- **CORS Configured**: Cross-origin requests supported
- **Comprehensive Testing**: Full API test coverage

## 🚀 Next Steps

1. **Stage 1 ✅**: FastAPI Backend (Complete)
2. **Stage 2**: React Native Mobile App
3. **Stage 3**: Advanced Mobile UI Features
4. **Stage 4**: Real-time Sync & Optimization

## 🤝 Contributing

This project maintains the principle of keeping all AI logic intact and editable while providing modern API interfaces for mobile development.

## 📄 License

Open source - feel free to use and modify for your photo management needs.
