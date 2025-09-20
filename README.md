# 📱 AI Photo Gallery - Ultimate On-Device Photo Search System

A comprehensive photo gallery application with advanced AI-powered features including face recognition, object detection, relationship mapping, and intelligent search capabilities. Built with React Native frontend and FastAPI backend.

## 🌟 Features

### 🔍 **Intelligent Search**
- **Natural Language Search**: Search photos using descriptive text
- **Object Detection**: Find photos containing specific objects (dogs, cars, etc.)
- **Face Recognition**: Search by person using advanced face clustering
- **Temporal Search**: Find photos from specific time periods

### 👥 **Face Recognition & People**
- **Advanced Face Detection**: SOTA face detection using InsightFace
- **Face Clustering**: Automatically group similar faces
- **Person Labeling**: Label face clusters with names
- **Relationship Mapping**: Discover relationships between people based on photo co-occurrence

### 📊 **Smart Organization**
- **Auto-Indexing**: Automatically process and index photos
- **Metadata Extraction**: Extract EXIF data, timestamps, and location info
- **Event Detection**: Group photos into events based on time proximity
- **Dynamic Galleries**: Browse photos by people, objects, or relationships

### 🔒 **Privacy-First Design**
- **100% On-Device Processing**: All AI runs locally, no cloud uploads
- **Biometric Data Protection**: Face embeddings stored locally only
- **User Consent**: Explicit opt-in for face processing
- **Data Control**: Easy deletion of person data and clusters

## 🏗️ Architecture

```
┌─────────────────┐    HTTP/REST API    ┌──────────────────┐
│  React Native   │ ◄─────────────────► │   FastAPI        │
│  Frontend       │                     │   Backend        │
│                 │                     │                  │
│ • Gallery UI    │                     │ • Photo API      │
│ • Search Screen │                     │ • Face Detection │
│ • Photo Viewer  │                     │ • AI Processing  │
│ • People Screen │                     │ • Relationships  │
└─────────────────┘                     └──────────────────┘
                                                   │
                                                   ▼
                                        ┌──────────────────┐
                                        │   SQLite DB      │
                                        │                  │
                                        │ • Photos         │
                                        │ • Faces          │
                                        │ • Clusters       │
                                        │ • Relationships  │
                                        └──────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- **Node.js** 16+ 
- **Python** 3.8+
- **React Native** development environment
- **Android Studio** (for Android) or **Xcode** (for iOS)

### 1. Clone Repository
```bash
git clone https://github.com/RahulRR-10/Gallery-Test.git
cd Gallery-Test
```

### 2. Backend Setup
```bash
cd backend

# Install Python dependencies
pip install -r requirements.txt

# Start the API server
python -m uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Frontend Setup
```bash
cd frontend

# Install Node.js dependencies
npm install

# Start Metro bundler
npm run start

# Run on Android (in a new terminal)
npm run android

# Or run on iOS (in a new terminal)
npm run ios
```

### 4. Configuration
Update the IP address in `frontend/src/config/environment.ts`:
```typescript
const Config = {
  development: {
    localIP: 'YOUR_COMPUTER_IP_ADDRESS', // Update this!
    localPort: 8000,
  }
};
```

## 📁 Project Structure

```
Gallery-Test/
├── backend/                          # FastAPI Backend
│   ├── api_server.py                 # Main API server
│   ├── photo_database.py             # Database operations
│   ├── api_helpers.py                # API helper functions
│   ├── advanced_face_detection.py    # Face detection & clustering
│   ├── final_photo_search.py         # AI-powered search
│   ├── relationship_mapping.py       # Relationship inference
│   ├── requirements.txt              # Python dependencies
│   ├── sample_photos/                # Sample photo directory
│   └── photos.db                     # SQLite database
│
├── frontend/                         # React Native Frontend
│   ├── src/
│   │   ├── components/               # Reusable UI components
│   │   │   ├── AppHeader.tsx
│   │   │   ├── PhotoGrid.tsx
│   │   │   ├── IndexingManager.tsx
│   │   │   └── ...
│   │   ├── screens/                  # Main application screens
│   │   │   ├── GalleryScreen.tsx     # Main photo gallery
│   │   │   ├── SearchScreen.tsx      # Search interface
│   │   │   ├── PeopleScreen.tsx      # Face clusters view
│   │   │   ├── PhotoViewer.tsx       # Individual photo view
│   │   │   ├── PersonScreen.tsx      # Person's photos
│   │   │   └── RelationshipsScreen.tsx
│   │   ├── services/
│   │   │   └── api.ts                # API communication
│   │   ├── utils/
│   │   │   └── photoUtils.ts         # Photo path utilities
│   │   └── config/
│   │       └── environment.ts        # Environment configuration
│   ├── android/                      # Android-specific files
│   ├── ios/                          # iOS-specific files
│   ├── package.json
│   └── App.tsx                       # Root component
│
└── README.md                         # This file
```

## 🔧 API Endpoints

### Photos
- `GET /api/photos` - Get all photos
- `GET /api/photos/{id}` - Get specific photo details
- `POST /api/search` - Search photos with various criteria

### Indexing
- `POST /api/index` - Start photo indexing process
- `GET /api/tasks/{task_id}` - Check indexing progress

### Face Recognition
- `GET /api/faces/clusters` - Get face clusters (people)
- `POST /api/faces/cluster` - Start face clustering
- `POST /api/faces/label` - Label a person

### Relationships
- `GET /api/relationships` - Get discovered relationships
- `POST /api/relationships/build` - Build relationship graph

### System
- `GET /api/status` - API health check
- `GET /api/stats` - System statistics

## 🤖 AI Components

### Face Detection (InsightFace)
- **Detection Model**: Buffalo_L for high accuracy
- **Recognition Model**: W600K_R50 for face embeddings
- **Age/Gender**: Optional demographic analysis
- **Clustering**: DBSCAN algorithm for grouping faces

### Object Detection (YOLOv8)
- **Model**: YOLOv8x for comprehensive object detection
- **Classes**: 80+ object categories (COCO dataset)
- **Performance**: Optimized for on-device inference

### Search System
- **CLIP Model**: For natural language to image search
- **Vector Similarity**: Cosine similarity for semantic matching
- **Temporal Parsing**: Natural language time expressions
- **Multi-modal**: Combines text, visual, and metadata search

## 🗄️ Database Schema

### Photos Table
```sql
CREATE TABLE photos (
    id TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    timestamp INTEGER,
    exif_timestamp INTEGER,
    objects TEXT,
    clip_embedding BLOB
);
```

### Face Clusters Table
```sql
CREATE TABLE face_clusters (
    cluster_id TEXT PRIMARY KEY,
    label TEXT,
    num_faces INTEGER,
    created_at TEXT
);
```

### Relationships Table
```sql
CREATE TABLE relationships (
    cluster_id_a TEXT,
    cluster_id_b TEXT,
    count INTEGER,
    weight REAL,
    PRIMARY KEY (cluster_id_a, cluster_id_b)
);
```

## 🎯 Usage Examples

### Basic Photo Browsing
1. **Index Photos**: Place photos in `backend/sample_photos/`
2. **Start Indexing**: Use the "Gallery Setup" in the app
3. **Browse**: Photos appear in the main gallery

### Face Recognition
1. **Cluster Faces**: Go to "People" tab → "Cluster Faces"
2. **Label People**: Tap on face clusters to add names
3. **Search by Person**: Use the search to find specific people

### Relationship Discovery
1. **Build Relationships**: Go to "Relationships" → "Build Relationships"
2. **View Results**: See inferred relationships between people
3. **Relationship Types**: family, friend, acquaintance based on co-occurrence

### Advanced Search
```
Search Examples:
• "dogs playing in the park"
• "birthday party with cake"
• "photos with John and Mary"
• "pictures from last summer"
• "cars on the street"
```

## 🔨 Development

### Backend Development
```bash
cd backend

# Install development dependencies
pip install -r requirements.txt

# Run with auto-reload
python -m uvicorn api_server:app --reload --host 0.0.0.0 --port 8000

# Run database migrations (if any)
python photo_database.py

# Test API endpoints
curl http://localhost:8000/api/status
```

### Frontend Development
```bash
cd frontend

# Install dependencies
npm install

# Start Metro bundler
npm start

# Run on device/emulator
npm run android  # or npm run ios

# Build release version
npm run build

# Clear cache if needed
npm start -- --reset-cache
```

### Testing Face Recognition
```bash
cd backend

# Test face detection on sample photos
python advanced_face_detection.py --test

# Test relationship building
python relationship_mapping.py
```

## 🐛 Troubleshooting

### Common Issues

#### Backend Not Starting
```bash
# Check Python version
python --version  # Should be 3.8+

# Install missing dependencies
pip install fastapi uvicorn opencv-python insightface

# Check port availability
netstat -an | grep 8000
```

#### Frontend Connection Issues
```typescript
// Update IP address in frontend/src/config/environment.ts
const Config = {
  development: {
    localIP: '192.168.1.100', // Your actual IP
    localPort: 8000,
  }
};
```

#### Face Detection Not Working
```bash
# Install InsightFace models (first run)
# Models will be downloaded automatically to ~/.insightface/

# Check if models are downloaded
ls ~/.insightface/models/buffalo_l/
```

#### Photos Not Loading
1. Check photo paths in `sample_photos/` directory
2. Ensure backend is serving static files at `/images`
3. Verify photo permissions and file formats (JPG, PNG)

### Performance Optimization

#### Backend
- Use GPU acceleration for face detection (if available)
- Adjust batch sizes for large photo collections
- Monitor memory usage during indexing

#### Frontend
- Enable Hermes engine for better performance
- Use image caching for faster loading
- Implement pagination for large galleries

## 🚀 Deployment

### Production Backend
```bash
# Install production dependencies
pip install gunicorn

# Run with Gunicorn
gunicorn api_server:app -w 4 -k uvicorn.workers.UvicornWorker

# Or use Docker
docker build -t photo-gallery-backend .
docker run -p 8000:8000 photo-gallery-backend
```

### Production Frontend
```bash
# Build release APK (Android)
cd android
./gradlew assembleRelease

# Build for iOS
cd ios
xcodebuild -workspace frontend.xcworkspace -scheme frontend archive
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- **InsightFace**: State-of-the-art face analysis toolkit
- **YOLOv8**: Object detection framework
- **OpenAI CLIP**: Vision-language model for semantic search
- **FastAPI**: Modern Python web framework
- **React Native**: Cross-platform mobile development

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the API documentation

## 🔮 Roadmap

### Upcoming Features
- [ ] **Advanced Relationship Types**: Detect family relationships (parent/child, siblings)
- [ ] **Event Timeline**: Automatic event detection and timeline view
- [ ] **Location Clustering**: Group photos by location using GPS data
- [ ] **Duplicate Detection**: Find and manage duplicate photos
- [ ] **Cloud Sync**: Optional cloud backup with end-to-end encryption
- [ ] **Video Support**: Extend AI features to video files
- [ ] **Collaborative Albums**: Share albums with family/friends
- [ ] **Advanced Filters**: More sophisticated search filters

### Technical Improvements
- [ ] **Performance**: GPU acceleration for mobile devices
- [ ] **Offline AI**: Improved on-device model optimization
- [ ] **Real-time Processing**: Live photo analysis during capture
- [ ] **Cross-platform**: Web version using React
- [ ] **API v2**: GraphQL API for better frontend integration

---

**Built with ❤️ for privacy-conscious photo management**