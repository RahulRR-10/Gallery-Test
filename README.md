# 📱 Samsung Galaxy Style Photo Gallery - Complete AI-Powered System

A fully-featured React Native photo gallery application with Samsung Galaxy UI styling and comprehensive AI capabilities. Features face recognition, object detection, semantic search, and automatic photo processing - all running locally for complete privacy.

## 🌟 Complete Feature Set

### 📱 **Samsung Galaxy UI Experience**
- **Native Gallery Design**: Dark theme matching Samsung Gallery aesthetic
- **Bottom Tab Navigation**: Pictures, Albums, People, Menu tabs
- **3-Column Photo Grid**: Mobile-optimized layout with proper spacing
- **Full-Screen Photo Viewer**: Smooth photo viewing with navigation
- **Refresh Control**: Pull-to-refresh functionality throughout
- **Auto-Indexing UI**: Real-time indexing status and manual controls

### 🤖 **Advanced AI Processing**
- **Face Detection**: InsightFace for high-accuracy face recognition
- **Object Detection**: YOLOv8x/YOLOv10x for 80+ object categories
- **Semantic Search**: OpenAI CLIP for natural language photo queries
- **Fast Face Clustering**: Optimized DBSCAN clustering for performance
- **Background Processing**: Non-blocking AI processing with task queue
- **Auto-Indexing**: Watchdog file monitoring for automatic processing

### 👥 **People & Relationship Management**
- **People Screen**: Browse all detected people with photo counts
- **Person Profiles**: Individual person pages with their photos
- **Face Clustering**: Automatic grouping of similar faces
- **Relationship Discovery**: Co-occurrence analysis between people
- **Groups Management**: Custom group creation and management
- **Manual Labeling**: Add names and organize people

### 🔍 **Comprehensive Search System**
- **Search Screen**: Dedicated search interface with multiple modes
- **Text Search**: Search by photo metadata and descriptions
- **Semantic Search**: Natural language queries ("sunset on beach")
- **People Search**: Find photos by detected people
- **Object Search**: Search by detected objects and categories
- **Date Range Search**: Filter photos by time periods

### ⚙️ **Settings & Configuration**
- **Settings Screen**: Complete configuration interface
- **Processing Options**: Control AI processing preferences
- **Storage Management**: Monitor database and cache usage
- **Auto-Indexing Control**: Enable/disable automatic processing
- **Performance Tuning**: Adjust processing parameters

## 🏗️ Complete System Architecture

```
┌─────────────────────────────┐    REST API     ┌─────────────────────────────┐
│     React Native App       │ ◄─────────────► │      FastAPI Backend       │
│                             │                 │                             │
│ Screens:                    │                 │ Core Services:              │
│ • PicturesScreen (Gallery)  │                 │ • Photo API Endpoints       │
│ • AlbumsScreen             │                 │ • Face Detection Service    │
│ • PeopleScreen             │                 │ • Object Detection Service  │
│ • SearchScreen             │                 │ • Semantic Search Engine    │
│ • SettingsScreen           │                 │ • Auto-Indexing System     │
│ • MenuScreen               │                 │ • Background Task Queue     │
│ • PhotoViewer              │                 │ • Static File Serving      │
│ • PersonScreen             │                 │                             │
│ • GroupsScreen             │                 │ AI Models:                  │
│ • RelationshipsScreen      │                 │ • InsightFace (Faces)       │
│                             │                 │ • YOLOv8x/v10x (Objects)   │
│ Components:                 │                 │ • OpenAI CLIP (Semantic)    │
│ • PhotoGrid               │                 │ • DBSCAN (Clustering)      │
│ • SearchBar               │                 │                             │
│ • IndexingManager         │                 │ Database Layer:             │
│ • TaskStatus              │                 │ • PhotoDatabase Class       │
└─────────────────────────────┘                 │ • SQLite with Embeddings   │
                                                │ • Face/Object Metadata     │
                                                └─────────────────────────────┘
                                                              │
                                                              ▼
                                                ┌─────────────────────────────┐
                                                │        SQLite Database      │
                                                │                             │
                                                │ Tables:                     │
                                                │ • photos (metadata, paths)  │
                                                │ • faces (embeddings, bbox)  │
                                                │ • face_clusters (groups)    │
                                                │ • objects (detections)      │
                                                │ • semantic_search (CLIP)    │
                                                └─────────────────────────────┘
```

## 📱 App Screens Overview

### Core Navigation Tabs
1. **Pictures Tab**: Main gallery with 3-column grid, timeline view, auto-indexing status
2. **Albums Tab**: Organized photo collections and smart albums
3. **People Tab**: Face detection results, person profiles, clustering management
4. **Menu Tab**: Settings, tools, and additional features

### Additional Screens
- **PhotoViewer**: Full-screen photo viewing with metadata
- **PersonScreen**: Individual person's photos and information
- **SearchScreen**: Multi-modal search interface
- **SettingsScreen**: App configuration and preferences
- **GroupsScreen**: People group management
- **RelationshipsScreen**: People relationship analysis
- **StoriesScreen**: Photo story creation (placeholder)

## 🚀 Quick Start Guide

### Prerequisites
- **Node.js** 18+ and **npm**
- **Python** 3.9+
- **React Native** development environment
- **Android Studio** (Android) or **Xcode** (iOS)

### 1. Project Setup
```bash
git clone <your-repo-url>
cd test_samsung
```

### 2. Backend Setup
```bash
cd backend

# Install Python dependencies (includes YOLOv10 support)
pip install -r requirements.txt

# Create sample photos directory
mkdir -p sample_photos

# Start the FastAPI server with auto-indexing
python -m uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```

The backend will be available at `http://localhost:8000` with:
- Auto-indexing enabled on startup
- Static file serving for photos
- Background task processing
- API documentation at `/docs`

### 3. Frontend Setup
```bash
cd frontend

# Install Node.js dependencies
npm install

# Update API configuration in src/config/api.ts
# Set BASE_URL to your backend server (default: http://10.0.2.2:8000 for Android)

# Start Metro bundler
npm start

# Run on Android (new terminal)
npm run android

# Or run on iOS
npm run ios
```

### 4. Add Photos for Processing
1. Place photos in `backend/sample_photos/` directory
2. The auto-indexing system will automatically detect and process new photos
3. Processing includes:
   - Face detection and embedding generation
   - Object detection and classification
   - CLIP semantic embedding creation
   - Metadata extraction (EXIF, location, date)

## 🔧 API Endpoints

### Photo Management
- `GET /api/photos` - Get all photos (limit: 1000)
- `GET /api/photos/{photo_id}` - Get specific photo details
- `POST /api/photos/search` - Search photos with text/semantic queries
- `GET /api/photos/{photo_id}/faces` - Get faces detected in photo

### Face & People Management
- `GET /api/faces` - Get all detected faces
- `GET /api/faces/clusters` - Get face clustering results
- `POST /api/faces/cluster` - Trigger face clustering
- `GET /api/people` - Get people (face clusters) with photo counts
- `PUT /api/faces/clusters/{cluster_id}` - Update person information

### Auto-Indexing System
- `GET /api/auto-indexing/status` - Check auto-indexing status
- `POST /api/auto-indexing/start` - Start auto-indexing service
- `POST /api/auto-indexing/stop` - Stop auto-indexing service
- `POST /api/index-photos` - Manually trigger photo indexing

### Background Tasks
- `GET /api/tasks` - Get background task status
- `POST /api/cluster-faces-background` - Start background face clustering

## 🤖 AI Model Configuration

### Face Detection (InsightFace)
- **Model**: Buffalo_l for high accuracy
- **Features**: Face detection, embedding generation, age/gender estimation
- **Performance**: ~100ms per face on CPU

### Object Detection (YOLO)
- **Models**: YOLOv8x and YOLOv10x support
- **Classes**: 80 COCO dataset categories
- **Confidence**: 0.5 threshold for detections
- **Performance**: ~500ms per image on CPU

### Semantic Search (CLIP)
- **Model**: OpenAI CLIP (ViT-B/32)
- **Capability**: Natural language photo queries
- **Languages**: English text queries
- **Performance**: ~200ms per image on CPU

## 🗂️ Project Structure

```
test_samsung/
├── backend/                 # FastAPI Python backend
│   ├── api_server.py       # Main FastAPI application
│   ├── photo_database.py   # SQLite database operations
│   ├── final_photo_search.py # Search implementation
│   ├── fast_clustering.py  # Optimized face clustering
│   ├── requirements.txt    # Python dependencies
│   ├── sample_photos/      # Photo storage directory
│   └── photos.db          # SQLite database
├── frontend/               # React Native app
│   ├── src/
│   │   ├── screens/       # All app screens
│   │   │   ├── PicturesScreen.tsx
│   │   │   ├── PeopleScreen.tsx
│   │   │   ├── SearchScreen.tsx
│   │   │   ├── SettingsScreen.tsx
│   │   │   └── ... (other screens)
│   │   ├── components/    # Reusable UI components
│   │   ├── services/      # API client and utilities
│   │   └── utils/         # Helper functions
│   ├── App.tsx           # Main app component
│   └── package.json      # Node.js dependencies
└── README.md            # This file
```

## 🎯 Key Features in Detail

### Auto-Indexing System
- **File Monitoring**: Watchdog monitors `sample_photos/` directory
- **Automatic Processing**: New photos are automatically processed on addition
- **Background Tasks**: Non-blocking processing with task queue
- **Status Monitoring**: Real-time indexing status in the app
- **Manual Control**: Start/stop auto-indexing from settings

### Face Clustering Algorithm
- **Detection**: InsightFace detects faces with bounding boxes
- **Embedding**: 512-dimensional face embeddings generated
- **Clustering**: DBSCAN algorithm groups similar faces
- **Fast Processing**: Optimized clustering for large photo collections
- **Manual Verification**: Review and correct clustering results

### Search Capabilities
1. **Text Search**: Search photo filenames and metadata
2. **Semantic Search**: Natural language queries using CLIP
3. **Face Search**: Find photos containing specific people
4. **Object Search**: Search by detected objects and categories
5. **Combined Search**: Mix multiple search modes

### Performance Optimizations
- **React Query**: Caching and automatic refetching
- **Image Caching**: Efficient image loading and memory management
- **Virtualized Lists**: Smooth scrolling for large photo collections
- **Background Processing**: Non-blocking AI operations
- **SQLite Indexing**: Optimized database queries

## 🛠️ Development & Customization

### Adding New AI Models
1. Update `requirements.txt` with new model dependencies
2. Implement model loading in appropriate service file
3. Add API endpoints in `api_server.py`
4. Update frontend to use new endpoints

### Customizing UI Theme
- Modify theme colors in `frontend/App.tsx`
- Update component styles in individual screen files
- Customize icons and navigation in `App.tsx`

### Database Schema Extensions
- Modify table schemas in `photo_database.py`
- Add migration logic for existing databases
- Update API endpoints to handle new fields

## 🔒 Privacy & Security

- **100% Local Processing**: All AI operations run on-device
- **No Cloud Uploads**: Photos never leave your device
- **SQLite Storage**: Local database with no external connections
- **No Telemetry**: No usage data collection or tracking
- **Offline Capable**: Full functionality without internet

## 📋 System Requirements

### Minimum Requirements
- **RAM**: 4GB (8GB+ recommended for large photo collections)
- **Storage**: 2GB free space for models and database
- **CPU**: Multi-core processor (ARM64 recommended)
- **OS**: Android 7.0+ or iOS 12.0+

### Development Requirements
- **Node.js**: 18.0+
- **Python**: 3.9+
- **React Native CLI**: Latest version
- **Android SDK**: API Level 21+
- **Xcode**: 12.0+ (iOS development)

## 🐛 Troubleshooting

### Common Issues
1. **Photos not loading**: Check `sample_photos/` directory permissions
2. **AI processing slow**: Reduce batch sizes in processing functions
3. **Database errors**: Delete `photos.db` to reset (will lose metadata)
4. **Face clustering issues**: Adjust DBSCAN parameters in clustering code
5. **API connection failed**: Verify backend server is running and accessible

### Performance Tuning
- Adjust photo processing batch sizes
- Modify face detection confidence thresholds
- Update clustering algorithm parameters
- Optimize image resizing and caching

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly on both platforms
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **InsightFace** for face recognition technology
- **Ultralytics YOLO** for object detection
- **OpenAI CLIP** for semantic search capabilities
- **React Native** for cross-platform mobile development
- **FastAPI** for high-performance backend API

---

