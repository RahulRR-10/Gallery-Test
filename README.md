# 📱 Smart Photo Gallery - AI-Powered On-Device Photo Management

A modern React Native photo gallery application with advanced AI capabilities for face recognition, object detection, and intelligent search. All processing happens locally on your device for complete privacy.

## 🌟 Key Features

### � **Native Gallery Experience**

- **Timeline View**: Native-style photo grid with date sections
- **High-Performance Scrolling**: Optimized FlatList with image caching
- **Full-Screen Photo Viewer**: Smooth photo viewing experience
- **Auto-Refresh**: Real-time updates when new photos are added

### 🤖 **AI-Powered Intelligence**

- **Face Detection & Clustering**: Automatically group photos by people using InsightFace
- **Object Recognition**: Detect 80+ object types with YOLOv8
- **Semantic Search**: Natural language photo search using OpenAI CLIP
- **Auto-Indexing**: Automatically process new photos with Watchdog monitoring

### 👥 **People Management**

- **Face Clustering**: Group similar faces automatically with DBSCAN
- **Person Labeling**: Name and organize people in your photos
- **Relationship Discovery**: Find connections between people based on co-occurrence
- **Group Management**: Create custom groups (family, friends, etc.)

### 🔒 **Privacy-First Design**

- **100% Local Processing**: All AI runs on-device, no cloud uploads
- **SQLite Database**: Local storage for all photo metadata and embeddings
- **No External Dependencies**: Complete offline functionality

## 🏗️ Architecture

```
┌─────────────────┐    HTTP/REST API    ┌──────────────────┐
│  React Native   │ ◄─────────────────► │   FastAPI        │
│  Frontend       │                     │   Backend        │
│                 │                     │                  │
│ • Gallery View  │                     │ • Photo API      │
│ • Search UI     │                     │ • Face Detection │
│ • People Screen │                     │ • Auto-Indexing │
│ • Photo Viewer  │                     │ • AI Processing  │
└─────────────────┘                     └──────────────────┘
                                                   │
                                                   ▼
                                        ┌──────────────────┐
                                        │   SQLite DB      │
                                        │                  │
                                        │ • Photos         │
                                        │ • Faces          │
                                        │ • Clusters       │
                                        │ • Embeddings     │
                                        └──────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- **Node.js** 18+ and **npm**
- **Python** 3.8+
- **React Native** development environment
- **Android Studio** (for Android) or **Xcode** (for iOS)

### 1. Clone and Setup

```bash
git clone <your-repo-url>
cd test_samsung
```

### 2. Backend Setup

```bash
cd backend

# Install Python dependencies
pip install -r requirements.txt

# Start the API server
python -m uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```

The backend will be available at `http://localhost:8000`

### 3. Frontend Setup

```bash
cd frontend

# Install Node.js dependencies
npm install

# Start Metro bundler
npm start

# Run on Android (new terminal)
npm run android

# Or run on iOS (new terminal)
npm run ios
```

### 4. Configuration

Update the backend IP address in `frontend/src/config/environment.ts`:

```typescript
const Config = {
  development: {
    localIP: "YOUR_COMPUTER_IP_ADDRESS", // e.g., "192.168.1.100"
    localPort: 8000,
  },
};
```

## 📁 Project Structure

```
test_samsung/
├── backend/                          # FastAPI Backend
│   ├── api_server.py                 # Main API server
│   ├── photo_database.py             # SQLite database operations
│   ├── auto_photo_indexer.py         # Watchdog auto-indexing service
│   ├── final_photo_search.py         # Core photo search functionality
│   ├── advanced_face_detection.py    # InsightFace integration
│   ├── fast_clustering.py            # Optimized face clustering
│   ├── clip_model.py                 # CLIP semantic embeddings
│   ├── temporal_search.py            # Time-based search parsing
│   ├── relationship_mapping.py       # People relationship discovery
│   ├── requirements.txt              # Python dependencies
│   ├── sample_photos/                # Photo storage directory
│   └── photos.db                     # SQLite database
│
├── frontend/                         # React Native Frontend
│   ├── src/
│   │   ├── screens/                  # Application screens
│   │   │   ├── GalleryScreen.tsx     # Main photo timeline
│   │   │   ├── SearchScreen.tsx      # Search interface
│   │   │   ├── PeopleScreen.tsx      # Face clusters management
│   │   │   ├── PhotoViewer.tsx       # Full-screen photo view
│   │   │   ├── PersonScreen.tsx      # Individual person's photos
│   │   │   ├── GroupsScreen.tsx      # People groups management
│   │   │   ├── RelationshipsScreen.tsx # Relationship visualization
│   │   │   └── SettingsScreen.tsx    # App settings
│   │   ├── components/               # Reusable UI components
│   │   │   ├── AppHeader.tsx         # Common header component
│   │   │   ├── PhotoGrid.tsx         # Grid photo display
│   │   │   ├── IndexingManager.tsx   # Indexing progress UI
│   │   │   ├── TaskStatus.tsx        # Background task status
│   │   │   └── SearchBar.tsx         # Search input component
│   │   ├── services/
│   │   │   └── api.ts                # Backend API communication
│   │   ├── utils/
│   │   │   └── photoUtils.ts         # Photo path utilities
│   │   └── config/
│   │       └── environment.ts        # Environment configuration
│   ├── android/                      # Android build files
│   ├── ios/                          # iOS build files
│   ├── package.json                  # Node.js dependencies
│   └── App.tsx                       # Root application component
└── README.md                         # This file
```

## 🔧 API Endpoints

### Core Photo Operations

- `GET /api/photos` - Get all indexed photos
- `GET /api/photos/{id}` - Get specific photo details
- `POST /api/search` - Search photos with natural language queries
- `GET /api/stats` - Get database statistics

### Indexing & Processing

- `POST /api/index` - Start background photo indexing
- `GET /api/tasks/{task_id}` - Check background task status
- `GET /api/auto-index/status` - Auto-indexing service status
- `POST /api/auto-index/start` - Start auto-indexing
- `POST /api/auto-index/stop` - Stop auto-indexing

### Face Recognition

- `GET /api/faces/clusters` - Get all face clusters (people)
- `POST /api/faces/cluster` - Start face clustering process
- `POST /api/faces/clusters/{id}/label` - Label a person

### Groups & Relationships

- `GET /api/groups` - Get people groups
- `POST /api/groups` - Create new people group
- `GET /api/relationships` - Get discovered relationships
- `POST /api/relationships/build` - Build relationship mappings

### System

- `GET /api/status` - API health check
- `GET /images/{filename}` - Serve photo files

## 🤖 AI Technology Stack

### Face Recognition (InsightFace)

- **Detection Model**: Buffalo_L for high-accuracy face detection
- **Embedding Model**: Face recognition with 512-dimensional embeddings
- **Clustering**: DBSCAN algorithm for automatic face grouping
- **Performance**: On-device processing with optimized inference

### Object Detection (YOLOv8x)

- **Model**: YOLOv8x for comprehensive object recognition
- **Classes**: 80+ object categories from COCO dataset
- **Capabilities**: Real-time object detection and classification
- **Integration**: Automatic object tagging during photo indexing

### Semantic Search (OpenAI CLIP)

- **Model**: LAION CLIP for natural language understanding
- **Features**: Text-to-image semantic similarity matching
- **Queries**: Support for descriptive search terms
- **Performance**: Fast vector similarity search with embeddings

### Auto-Indexing (Watchdog)

- **Monitoring**: Real-time file system watching
- **Processing**: Automatic photo detection and indexing
- **Efficiency**: Debounced processing to handle file operations
- **Coverage**: Recursive directory monitoring with file stability checks

## 🗄️ Database Schema

### Photos Table

```sql
CREATE TABLE photos (
    id TEXT PRIMARY KEY,           -- SHA-256 hash of file
    path TEXT NOT NULL,            -- File system path
    timestamp INTEGER,             -- File creation timestamp
    exif_timestamp INTEGER,        -- EXIF timestamp if available
    objects TEXT,                  -- JSON array of detected objects
    clip_embedding BLOB            -- CLIP semantic embedding
);
```

### Faces Table

```sql
CREATE TABLE faces (
    face_id TEXT PRIMARY KEY,      -- Unique face identifier
    photo_id TEXT,                 -- Reference to photos table
    bbox TEXT,                     -- Face bounding box coordinates
    embedding BLOB,                -- Face recognition embedding
    cluster_id TEXT,               -- Assigned cluster/person ID
    detection_method TEXT          -- Detection algorithm used
);
```

### Face Clusters Table

```sql
CREATE TABLE face_clusters (
    cluster_id TEXT PRIMARY KEY,   -- Unique cluster identifier
    label TEXT,                    -- Person name/label
    num_faces INTEGER,             -- Number of faces in cluster
    created_at TEXT                -- Cluster creation timestamp
);
```

## 🎯 How to Use

### First-Time Setup

1. **Start the Application**

   - Launch backend: `python -m uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload`
   - Launch frontend: `npm run android` or `npm run ios`

2. **Add Photos**

   - Place photos in `backend/sample_photos/` directory
   - Auto-indexing will automatically detect and process new photos
   - Or use the "Gallery Setup" button in the app to manually index

3. **Enable Face Recognition**
   - Go to "People" tab in the app
   - Tap the clustering button (face icon) to start face detection
   - Wait for processing to complete (progress shown in real-time)

### Daily Usage

#### Browse Photos

- **Gallery Tab**: Scroll through your photos in timeline view
- **Full-Screen Viewing**: Tap any photo to view in detail
- **Auto-Refresh**: New photos appear automatically without restart

#### Search Photos

- **Search Tab**: Enter natural language queries
- Examples:
  - "dogs playing in the park"
  - "birthday party with cake"
  - "red flowers"
  - "people smiling"

#### Manage People

- **People Tab**: View face clusters automatically created
- **Label People**: Tap on face clusters to add names
- **Person View**: See all photos of a specific person

#### Organize Groups

- **Groups Tab**: Create custom groups (family, friends, work)
- **Add Members**: Select people to include in groups
- **Group Photos**: View photos containing group members

#### Discover Relationships

- **Relationships Tab**: See automatically discovered connections
- **Co-occurrence**: People who appear together frequently
- **Relationship Strength**: Based on number of shared photos

## � Development

### Backend Development

```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Run with auto-reload for development
python -m uvicorn api_server:app --reload --host 0.0.0.0 --port 8000

# Test individual components
python final_photo_search.py --index sample_photos  # Index photos
python final_photo_search.py --stats                # Show database stats
python final_photo_search.py --search "red flower"  # Test search
```

### Frontend Development

```bash
cd frontend

# Install dependencies
npm install

# Start Metro bundler with cache reset
npm start -- --reset-cache

# Development builds
npm run android  # Android
npm run ios      # iOS

# Debug mode
npm run android -- --mode debug
```

### Testing the API

```bash
# Health check
curl http://localhost:8000/api/status

# Get photos
curl http://localhost:8000/api/photos

# Search photos
curl -X POST http://localhost:8000/api/search \
     -H "Content-Type: application/json" \
     -d '{"query": "dog", "limit": 5}'
```

## 🐛 Troubleshooting

### Common Issues

#### Backend Won't Start

```bash
# Check Python version (3.8+ required)
python --version

# Install missing dependencies
pip install fastapi uvicorn

# Check if port 8000 is already in use
netstat -an | findstr :8000  # Windows
lsof -i :8000                # macOS/Linux
```

#### Frontend Connection Issues

- **Update IP Address**: Edit `frontend/src/config/environment.ts` with your computer's IP
- **Check Network**: Ensure device and computer are on same network
- **Firewall**: Make sure port 8000 is not blocked by firewall

#### Face Detection Not Working

```bash
# Check if InsightFace models are downloaded
# Models download automatically on first use to ~/.insightface/

# Verify face detection manually
cd backend
python -c "from advanced_face_detection import AdvancedFaceDetector; detector = AdvancedFaceDetector(); print('Face detection working')"
```

#### Photos Not Loading

- **Check Photo Directory**: Ensure photos are in `backend/sample_photos/`
- **File Permissions**: Make sure backend can read photo files
- **Supported Formats**: JPG, PNG, BMP, TIFF, WEBP are supported
- **File Paths**: Avoid special characters in file names

#### Auto-Indexing Issues

```bash
# Check auto-indexing status
curl http://localhost:8000/api/auto-index/status

# Restart auto-indexing
curl -X POST http://localhost:8000/api/auto-index/stop
curl -X POST http://localhost:8000/api/auto-index/start
```

#### React Native Build Issues

```bash
# Clear React Native caches
cd frontend
npm start -- --reset-cache

# Clean Android build
cd android
./gradlew clean

# Clean iOS build
cd ios
rm -rf build/
```

### Performance Tips

#### Backend Optimization

- **GPU Acceleration**: Install GPU versions of PyTorch if available
- **Batch Processing**: Reduce batch sizes if running out of memory
- **Database**: Use SSD storage for better SQLite performance

#### Frontend Optimization

- **Image Caching**: Images are cached automatically by React Native
- **Memory Management**: Large photo collections may require pagination
- **Network**: Use WiFi for faster image loading

## 🚀 Deployment

### Production Backend

```bash
# Install production ASGI server
pip install gunicorn

# Run with Gunicorn
gunicorn api_server:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# Or run with Docker
docker build -t smart-photo-gallery-backend .
docker run -p 8000:8000 smart-photo-gallery-backend
```

### Production Frontend

```bash
# Build Android APK
cd frontend/android
./gradlew assembleRelease

# Build iOS app
cd frontend/ios
xcodebuild -workspace frontend.xcworkspace -scheme frontend archive
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[InsightFace](https://github.com/deepinsight/insightface)**: State-of-the-art face analysis toolkit
- **[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)**: Advanced object detection framework
- **[OpenAI CLIP](https://github.com/openai/CLIP)**: Vision-language model for semantic search
- **[FastAPI](https://fastapi.tiangolo.com/)**: Modern Python web framework
- **[React Native](https://reactnative.dev/)**: Cross-platform mobile development
- **[Watchdog](https://github.com/gorakhargosh/watchdog)**: Python file system monitoring

## 🔮 Future Enhancements

### Planned Features

- [ ] **Enhanced Relationship Detection**: Parent/child, sibling relationships using advanced ML
- [ ] **Event Timeline**: Automatic event detection and timeline visualization
- [ ] **Location Intelligence**: GPS-based photo clustering and location search
- [ ] **Duplicate Detection**: Find and manage duplicate or similar photos
- [ ] **Video Support**: Extend AI features to video files with frame analysis
- [ ] **Advanced Search Filters**: Date ranges, location filters, object combinations

### Technical Improvements

- [ ] **Performance**: GPU acceleration for mobile AI inference
- [ ] **Offline AI**: Optimized on-device models for better performance
- [ ] **Cloud Sync**: Optional encrypted cloud backup with local-first approach
- [ ] **Web Interface**: Browser-based photo management using React
- [ ] **Real-time Processing**: Live photo analysis during camera capture

---

**Built with ❤️ for privacy-conscious photo management**
