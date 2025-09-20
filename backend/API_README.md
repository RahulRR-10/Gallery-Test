# 🚀 FastAPI Backend Documentation

## Overview

The FastAPI backend provides a REST API wrapper around the existing AI photo search CLI system. This maintains the principle of keeping all AI logic intact and editable while providing a mobile-friendly API interface.

## 🏃‍♂️ Quick Start

### 1. Start the API Server

```bash
# Simple start (recommended)
python start_api.py

# Or manually with uvicorn
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload

# For local development only
python start_api.py --local
```

### 2. Access the API

- **API Base URL**: `http://localhost:8000`
- **Interactive Docs**: `http://localhost:8000/docs`
- **API Schema**: `http://localhost:8000/redoc`
- **Health Check**: `http://localhost:8000/api/status`

### 3. Test the API

```bash
# Run all tests
python test_api.py

# Test specific functionality
python test_api.py --test search --query "flower"
python test_api.py --test stats
```

## 📡 API Endpoints

### **Core Endpoints**

| Method | Endpoint      | Description                                      |
| ------ | ------------- | ------------------------------------------------ |
| `GET`  | `/`           | API information and status                       |
| `GET`  | `/api/status` | Health check and system status                   |
| `GET`  | `/api/stats`  | System statistics (photos, faces, relationships) |

### **Search Endpoints**

| Method | Endpoint             | Description                                           |
| ------ | -------------------- | ----------------------------------------------------- |
| `POST` | `/api/search`        | Search photos (semantic, person, group, relationship) |
| `GET`  | `/api/photos/{id}`   | Get detailed photo information                        |
| `GET`  | `/images/{filename}` | Serve image files                                     |

**Search Request Example:**

```json
{
  "query": "beautiful flower",
  "person": "Alice",
  "group": "family",
  "relationship": "family",
  "time_filter": "last month",
  "limit": 10
}
```

### **Photo Management**

| Method | Endpoint               | Description                 |
| ------ | ---------------------- | --------------------------- |
| `POST` | `/api/index`           | Index photos from directory |
| `GET`  | `/api/tasks/{task_id}` | Get background task status  |

**Index Request Example:**

```json
{
  "directory": "sample_photos",
  "recursive": true
}
```

### **Face Management**

| Method | Endpoint              | Description                             |
| ------ | --------------------- | --------------------------------------- |
| `POST` | `/api/faces/cluster`  | Start face clustering (background task) |
| `GET`  | `/api/faces/clusters` | List all face clusters                  |
| `POST` | `/api/faces/label`    | Label a face cluster                    |

**Label Request Example:**

```json
{
  "cluster_id": "cluster_1",
  "name": "Alice"
}
```

### **Relationship & Group Management**

| Method | Endpoint                   | Description                                |
| ------ | -------------------------- | ------------------------------------------ |
| `POST` | `/api/relationships/build` | Build relationship graph (background task) |
| `GET`  | `/api/relationships`       | List all relationships                     |
| `POST` | `/api/groups/create`       | Create person group                        |
| `GET`  | `/api/groups`              | List all groups                            |

**Create Group Example:**

```json
{
  "group_name": "family",
  "cluster_ids": ["cluster_1", "cluster_2", "cluster_3"]
}
```

## 🔧 Configuration

### **Environment Variables**

- `API_HOST`: Host to bind to (default: `0.0.0.0`)
- `API_PORT`: Port to bind to (default: `8000`)
- `DB_PATH`: Database file path (default: `photos.db`)

### **CORS Configuration**

The API is configured to allow all origins for development. For production, update the CORS settings in `api_server.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React Native dev server
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

## 📱 Mobile App Integration

### **Network Setup**

1. **Find your computer's IP address:**

   ```bash
   # Windows
   ipconfig

   # Linux/Mac
   ifconfig
   ```

2. **Start API server:**

   ```bash
   python start_api.py --host 0.0.0.0 --port 8000
   ```

3. **Connect mobile app to:**
   ```
   http://192.168.1.XXX:8000
   ```

### **API Response Format**

All API responses follow this structure:

```json
{
  "results": [...],
  "total": 10,
  "query": "search terms",
  "search_method": "semantic"
}
```

**Photo Response:**

```json
{
  "id": 1,
  "filename": "photo.jpg",
  "path": "/path/to/photo.jpg",
  "similarity_score": 0.95,
  "objects": ["person", "flower", "outdoor"],
  "faces": [
    {
      "cluster_id": "cluster_1",
      "bbox": [100, 100, 200, 200],
      "confidence": 0.98
    }
  ],
  "relationships": [
    {
      "person1": "cluster_1",
      "person2": "cluster_2",
      "type": "family",
      "confidence": 0.9
    }
  ]
}
```

## 🔧 Background Tasks

Long-running operations (indexing, clustering, relationship building) run as background tasks:

1. **Start Task**: POST to endpoint returns `task_id`
2. **Monitor Progress**: GET `/api/tasks/{task_id}`
3. **Task Status**: `running`, `completed`, `failed`

**Task Status Response:**

```json
{
  "status": "running",
  "progress": 75,
  "total": 100,
  "error": null
}
```

## 🛠️ Development

### **Adding New Endpoints**

1. Add endpoint function to `api_server.py`
2. Update Pydantic models if needed
3. Add helper functions to `api_helpers.py`
4. Add tests to `test_api.py`

### **Error Handling**

All endpoints use consistent error handling:

```json
{
  "detail": "Error message",
  "status_code": 500
}
```

### **Logging**

Logs are written to console with different levels:

- `INFO`: Normal operations
- `WARNING`: Non-critical issues
- `ERROR`: Critical errors

## 📊 Performance Notes

### **Image Serving**

- Images served via FastAPI StaticFiles
- Thumbnails should be generated for mobile optimization
- Consider image compression for mobile networks

### **Database Queries**

- SQLite connections are created per request
- Consider connection pooling for high load
- API helpers cache frequently accessed data

### **Memory Usage**

- CLIP and YOLO models loaded once at startup
- Background tasks may use additional memory
- Monitor memory usage with large photo collections

## 🔐 Security Considerations

### **Production Deployment**

- Update CORS origins for production
- Add authentication/authorization if needed
- Use HTTPS for mobile app connections
- Validate all input parameters
- Rate limiting for API endpoints

### **File Access**

- Images served only from designated directories
- Path traversal protection in place
- File type validation for uploads

## 🧪 Testing

### **API Tests**

```bash
# Test all endpoints
python test_api.py

# Test specific functionality
python test_api.py --test search
python test_api.py --test faces
python test_api.py --test relationships

# Custom search test
python test_api.py --test search --query "motorcycle"
```

### **Manual Testing**

Use the interactive API docs at `http://localhost:8000/docs` to:

- Test individual endpoints
- View request/response schemas
- Execute API calls directly

## 🚀 Next Steps

1. **Stage 1 Complete**: ✅ FastAPI backend with core endpoints
2. **Stage 2**: Create React Native mobile app
3. **Stage 3**: Add face management UI
4. **Stage 4**: Implement relationship visualization
5. **Stage 5**: Performance optimization and polish

The FastAPI backend is now ready for mobile app development! 🎉
