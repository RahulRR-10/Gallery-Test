# 🌟 Ultimate On-Device AI Photo Search System

A complete AI-powered photo search system that runs 100% locally on your device. Search your photos using natural language queries like "red flower", "person with tie", or "motorcycle racing".

## 🚀 Features

- **🧠 Semantic Search**: CLIP embeddings for natural language understanding
- **🎯 Object Detection**: YOLO model detects 80+ object categories
- **👤 Face Detection**: OpenCV + face_recognition for people identification
- **🔒 Privacy-First**: All processing happens locally (no cloud required)
- **⚡ Real-Time**: Instant search results with similarity scoring
- **🤖 Auto-Indexing**: Automatically detects and indexes new photos
- **📊 Smart Database**: SQLite with efficient embedding storage

## 📁 Project Structure

```
├── final_photo_search.py    # Main CLI search system
├── auto_photo_search.py     # Auto-indexing with monitoring
├── demo_photo_search.py     # Interactive demonstration
├── project_summary.py       # Project overview and stats
├── clip_model.py           # CLIP embedding extractor
├── photo_database.py       # SQLite database manager
├── photos.db              # Photo database (auto-created)
├── yolov8n.pt            # YOLO model weights
├── sample_photos/        # Your photo collection
└── README.md            # This file
```

## 🎮 Quick Start

### 1. Search Photos

```bash
# Search with natural language
python final_photo_search.py --search "red flower"
python final_photo_search.py --search "person wearing tie"
python final_photo_search.py --search "motorcycle racing"

# Limit results
python final_photo_search.py --search "beautiful nature" --limit 3

# Show database stats
python final_photo_search.py --stats
```

### 2. Index New Photos

```bash
# Index photos from a directory
python final_photo_search.py --index sample_photos

# Auto-indexing with monitoring
python auto_photo_search.py --watch
```

### 3. Interactive Demo

```bash
# Full system demonstration
python demo_photo_search.py

# Interactive search mode only
python auto_photo_search.py
```

## 🔧 How It Works

### AI Models Used

- **CLIP**: OpenAI ViT-B/32 (151M parameters) for image-text embeddings
- **YOLO**: YOLOv8n for real-time object detection (80 categories)
- **Face Detection**: OpenCV Haar cascades + face_recognition

### Search Process

1. **Indexing**: Extract 512-dimensional CLIP embeddings for each photo
2. **Object Detection**: Identify objects using YOLO (person, car, etc.)
3. **Face Detection**: Find and encode faces in photos
4. **Storage**: Save embeddings and metadata in SQLite database
5. **Search**: Convert text query to embedding, find similar photos
6. **Ranking**: Sort results by cosine similarity score

## 📊 Current Database Stats

- **📸 Total Photos**: 64 indexed
- **🧠 CLIP Model**: LAION ViT-H/14 (986M params) - High Accuracy
- **🎯 Object Detection**: YOLOv8x (68M params) - Professional Grade
- **👤 Face Detection**: InsightFace Buffalo_L - State-of-the-Art
- **🕒 Temporal Intelligence**: EXIF + Filename parsing
- **🧠 Embedding Dimension**: 1024 (upgraded from 512)
- **💾 Database Size**: ~2MB with full AI analysis

## 🎯 Search Examples

| Query                         | Best Results                           |
| ----------------------------- | -------------------------------------- |
| `"motorcycle racing sport"`   | motorbike_0223.jpg, motorbike_0221.jpg |
| `"person wearing tie formal"` | person_0808.jpg, person_0807.jpg       |
| `"beautiful flower garden"`   | flower_0716.jpg, flower_0715.jpg       |
| `"delicious food dessert"`    | fruit_0006.jpg (cake detected)         |
| `"colorful nature outdoor"`   | flower_0716.jpg (potted plant)         |

## 🛠️ System Requirements

- **Python 3.8+**
- **PyTorch** (CPU or GPU supported)
- **transformers** (Hugging Face)
- **ultralytics** (YOLOv8x)
- **opencv-python**
- **insightface** (Advanced face detection)
- **onnxruntime** (Model inference)
- **dateparser** (Temporal intelligence)
- **RAM**: 4GB+ recommended (8GB+ for optimal performance)
- **Storage**: 2GB+ for models and cache
- **Network**: Initial download of ~1GB models

## 🌟 What Makes This Special

- **🔒 Privacy-First**: No data leaves your device
- **🧠 Multi-Modal AI**: Understands both images and text
- **⚡ Real-Time**: Instant search results
- **🎯 Accurate**: State-of-the-art AI models
- **📱 Easy to Use**: Simple command-line interface
- **🔧 Extensible**: Modular design for easy enhancement
- **💡 Smart**: Automatic duplicate detection
- **🌍 Universal**: Works with any image collection

## 📝 Usage Tips

1. **Add Photos**: Drop images into `sample_photos/` folder
2. **Index First**: Run indexing before searching new photos
3. **Natural Language**: Use descriptive queries like "happy person"
4. **Be Specific**: "red sports car" works better than just "car"
5. **Check Objects**: Use `--stats` to see detected object categories

## 🚀 Phase 2 Roadmap

### Stage 1: Temporal Intelligence

**Temporal Intelligence** adds time-based filtering to your photo searches using EXIF metadata timestamps. This allows you to search photos not just by content, but by when they were taken.

**Features:**
- Parse EXIF timestamps from photo metadata
- Map human time expressions to date ranges using dateparser
- Filter search results by time periods
- Support natural language time queries

**Examples:**
```bash
# Search photos from a specific year
python final_photo_search.py --search "flowers" --time "2023"

# Find photos from last Christmas
python final_photo_search.py --search "celebration" --time "last Christmas"

# Pictures from college years
python final_photo_search.py --search "friends" --time "2018-2022"

# Recent vacation photos
python final_photo_search.py --search "travel" --time "last month"
```

**Implementation:**
- EXIF timestamp extraction using Pillow
- Natural language parsing with dateparser library
- Extended database schema for temporal metadata
- CLI integration with `--time` parameter

### Stage 2: Face Recognition + Relationship Mapping

**Face Recognition + Relationship Mapping** builds upon the advanced face detection to identify and cluster people across your photo collection.

**Features:**
- Face embedding clustering using InsightFace models
- User labeling system for person identification
- Co-occurrence analysis for relationship mapping
- Social graph construction from photo metadata
- Advanced person search capabilities

**Examples:**
```bash
# Search photos with a specific person (after labeling)
python final_photo_search.py --search "photos with Sarah"

# Find family photos
python final_photo_search.py --search "family photos"

# Pictures with coworkers
python final_photo_search.py --search "pictures with my coworkers"

# Photos of multiple people together
python final_photo_search.py --search "group photos"
```

**Implementation:**
- InsightFace face embeddings for person identification
- DBSCAN clustering for face grouping
- User labeling interface for person names
- Co-occurrence matrix for relationship detection
- Extended database schema for face clusters and relationships

**Status:** 🚧 Coming in Stage 2
