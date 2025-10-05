# 🤖 Pixie - Your AI-Powered Photo Gallery

**Next-generation photo management with intelligent search, face recognition, and natural language understanding - completely local and privacy-first.**

## ✨ Key Features

- 🔍 **Natural Language Search** - Find photos using queries like "birthday party with John and Sarah"
- 👥 **Smart Face Recognition** - Automatic face detection, clustering, and relationship mapping
- 🎯 **Object Detection** - AI-powered recognition of 80+ object categories using YOLOv10
- 🧠 **Semantic Understanding** - CLIP-powered contextual search and scene understanding
- � **Samsung Galaxy UI** - Pixel-perfect mobile interface with dark theme optimization
- 🛡️ **Privacy-First** - 100% local processing, no cloud dependencies
- ⚡ **Real-Time Performance** - Auto-indexing with background processing
- 🔗 **Relationship Discovery** - Automatic social connection mapping from photo co-occurrence

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![React Native](https://img.shields.io/badge/React_Native-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
![SQLite](https://img.shields.io/badge/SQLite-07405E?style=for-the-badge&logo=sqlite&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

**AI Models:**

- 🎭 **InsightFace** - Face recognition and embedding generation
- 🎯 **YOLOv8x/v10x** - Object detection and classification
- 🧠 **OpenAI CLIP** - Natural language to image understanding
- 🎲 **DBSCAN** - Intelligent face clustering algorithm

**Architecture:**

- **Backend:** FastAPI with SQLite database
- **Frontend:** React Native with TypeScript
- **AI Pipeline:** Local inference with optimized model loading
- **Storage:** Local file system with metadata indexing

## 🔨 Setup and Installation

### Prerequisites

Ensure you have the following installed:

- **Node.js** 18+ and **npm**
- **Python** 3.9+
- **Git** for version control
- **Android Studio** (for Android) or **Xcode** (for iOS)
- **React Native CLI** globally installed

### Step 1: Clone the Repository

```bash
git clone https://github.com/RahulRR-10/Kiddocare_RVCE_4_B.AmishaPai.git
cd Kiddocare_RVCE_4_B.AmishaPai
```

### Step 2: Backend Setup

```bash
# Navigate to backend directory
cd backend

# Install Python dependencies from requirements.txt
pip install -r requirements.txt

# Create photos directory
mkdir sample_photos

# Initialize database (automatic on first run)
# The database will be created when you start the server
```

### Step 3: Frontend Setup

```bash
# Navigate to frontend directory
cd ../frontend

# Install Node.js dependencies
npm install

# For iOS, install CocoaPods dependencies
cd ios && pod install && cd ..
```

Add your local IP address in `.env` file (create from `.envExample`).
Optionally also add it into frontend/src/config/environment.ts

### Step 4: Download AI Models

The AI models will be automatically downloaded on first run:

- **YOLOv8x/v10x** models (~140MB each)
- **InsightFace** models (~100MB)
- **OpenAI CLIP** models (~600MB)

_Note: First startup may take longer due to model downloads_

## 🚀 Running the Project

### Start the Backend Server

```bash
cd backend
python -m uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```

The backend will be available at:

- **API Server:** `http://localhost:8000`
- **API Documentation:** `http://localhost:8000/docs`

### Start the Frontend App

```bash
# In a new terminal
cd frontend

# Start Metro bundler
npm start

# In another terminal, run the app
npm run android    # For Android
# or
npm run ios        # For iOS
```

### Add Photos for Processing

1. Place your photos in the `backend/sample_photos/` directory (ensure they contain EXIF data)
2. The auto-indexing system will automatically:
   - Detect new photos
   - Extract faces and generate embeddings
   - Perform object detection
   - Create semantic embeddings
   - Update the database

## 📖 API Documentation

Once the backend is running, visit `http://localhost:8000/docs` for:

- Interactive API testing interface
- Complete endpoint documentation
- Request/response schemas
- Authentication details (if applicable)

## 🎯 Submissions

### Demo Materials

- **Demo Video:** [https://youtu.be/5mEDDtHzC_w](https://youtu.be/5mEDDtHzC_w)
- **GitHub Repository:** [https://github.com/RahulRR-10/Kiddocare_RVCE_4_B.AmishaPai.git](https://github.com/RahulRR-10/Kiddocare_RVCE_4_B.AmishaPai.git)
- **Supplementary Document:** [https://docs.google.com/document/d/1nonxLqfItLUBOJSJdzQjC6OZu2-Dbvk209_MkjsXBbc/edit?usp=sharing](https://docs.google.com/document/d/1nonxLqfItLUBOJSJdzQjC6OZu2-Dbvk209_MkjsXBbc/edit?usp=sharing)
- **Presentation Slides:** [https://www.canva.com/design/DAGyphpqxc8/eQnkCYDO6u_apMiAT-2wnQ/edit?utm_content=DAGyphpqxc8&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton](https://www.canva.com/design/DAGyphpqxc8/eQnkCYDO6u_apMiAT-2wnQ/edit?utm_content=DAGyphpqxc8&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

### Project Resources

- **Technical Architecture:** See `backend/API_README.md`
- **Setup Guide:** See `backend/SETUP.md`

## � Project Structure

```
test_samsung/
├── backend/                          # FastAPI Python backend
│   ├── api_server.py                 # Main FastAPI application
│   ├── photo_database.py             # SQLite database operations
│   ├── final_photo_search.py         # Complete search system
│   ├── fast_object_search.py         # Fast object detection search
│   ├── intelligent_query_parser.py   # Natural language query parsing
│   ├── lightweight_person_search.py  # Optimized person search
│   ├── advanced_face_detection.py    # Face detection and recognition
│   ├── contextual_search.py          # Scenario-based search engine
│   ├── fast_clustering.py            # Face clustering algorithms
│   ├── temporal_search.py            # Time-based search utilities
│   ├── relationship_mapping.py       # Social relationship analysis
│   ├── auto_photo_indexer.py         # Background auto-indexing
│   ├── api_helpers.py                # Database helper functions
│   ├── clip_model.py                 # CLIP semantic search
│   ├── detection_config.py           # AI model configurations
│   ├── requirements.txt              # Python dependencies
│   ├── sample_photos/                # Photo storage directory
│   ├── photos.db                     # SQLite database
│   ├── yolov8x.pt                    # YOLO object detection model
│   ├── yolov10x.pt                   # Advanced YOLO model
│   └── relationship_graph.json       # Relationship data
├── frontend/                         # React Native app
│   ├── src/
│   │   ├── screens/                  # All app screens
│   │   │   ├── PicturesScreen.tsx    # Main gallery view
│   │   │   ├── AlbumsScreen.tsx      # Albums management
│   │   │   ├── PeopleScreen.tsx      # People and faces
│   │   │   ├── MenuScreen.tsx        # Settings and menu
│   │   │   ├── SearchScreen.tsx      # Search interface
│   │   │   ├── PhotoViewer.tsx       # Full-screen photo view
│   │   │   ├── PersonScreen.tsx      # Individual person view
│   │   │   ├── SettingsScreen.tsx    # App configuration
│   │   │   ├── GroupsScreen.tsx      # People groups
│   │   │   └── RelationshipsScreen.tsx # Relationship analysis
│   │   ├── components/               # Reusable UI components
│   │   │   ├── PhotoGrid.tsx         # Photo grid component
│   │   │   ├── SearchBar.tsx         # Search input
│   │   │   ├── IndexingManager.tsx   # Indexing status
│   │   │   └── TaskStatus.tsx        # Background task monitoring
│   │   ├── services/                 # API and utilities
│   │   │   ├── api.ts                # API client
│   │   │   ├── photoService.ts       # Photo operations
│   │   │   └── searchService.ts      # Search functionality
│   │   └── utils/                    # Helper functions
│   ├── App.tsx                       # Main app component
│   ├── package.json                  # Node.js dependencies
│   └── android/                      # Android-specific files
└── README.md                         # Project documentation
```

## 🔧 Troubleshooting

### Common Issues

**Photos not appearing:**

- Ensure photos are in `backend/sample_photos/` directory
- Check file permissions on the photos directory
- Verify supported image formats (JPG, PNG, JPEG)

**Backend startup errors:**

- Verify Python 3.9+ is installed: `python --version`
- Install missing dependencies: `pip install -r requirements.txt`
- Check port 8000 is not in use: `netstat -an | findstr :8000`

**Frontend connection issues:**

- Ensure backend is running on `http://localhost:8000`
- Check API_BASE_URL in frontend configuration
- Verify network connectivity between frontend and backend

**AI processing slow:**

- Reduce batch sizes in processing configuration
- Check available system memory (minimum 4GB recommended)
- Monitor CPU usage during AI inference

**Face clustering not working:**

- Clear existing clusters and re-run clustering
- Adjust DBSCAN parameters (eps, min_samples) in clustering code
- Ensure sufficient face samples for clustering

**Database errors:**

- Delete `photos.db` to reset database (will lose all metadata)
- Check SQLite database permissions
- Verify database schema matches current version

### Performance Optimization

- **Reduce image processing batch sizes** for lower memory usage
- **Adjust AI model confidence thresholds** to balance accuracy vs speed
- **Enable model caching** to avoid reloading models between requests
- **Use SSD storage** for faster database and file access
- **Allocate more RAM** for better AI model performance

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **[InsightFace](https://github.com/deepinsight/insightface)** - Face recognition technology
- **[Ultralytics](https://github.com/ultralytics/ultralytics)** - YOLO object detection models
- **[OpenAI CLIP](https://github.com/openai/CLIP)** - Vision-language understanding
- **[React Native](https://reactnative.dev/)** - Cross-platform mobile framework
- **[FastAPI](https://fastapi.tiangolo.com/)** - High-performance API framework

---

