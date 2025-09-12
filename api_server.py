"""
FastAPI Backend for AI Photo Search System

This module provides a REST API wrapper around the existing CLI photo search functionality.
All AI logic, database handling, and CLI functions remain intact and editable.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Query, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
import asyncio
import json
import traceback
from pathlib import Path
import logging
from pydantic import BaseModel
from typing import List
# Import existing modules (keep all AI logic intact)
from photo_database import PhotoDatabase
from api_helpers import APIHelpers
from final_photo_search import UltimatePhotoSearcher
import relationship_mapping
from temporal_search import TemporalParser
import datetime
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Photo Search API",
    description="REST API for the Ultimate On-Device AI Photo Search System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for mobile app connectivity
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
db = None
api_helpers = APIHelpers()
photo_searcher = None
temporal_parser = TemporalParser()

def get_database():
    """Get database instance with lazy initialization"""
    global db
    if db is None:
        db = PhotoDatabase()
    return db

def get_photo_searcher():
    """Get photo searcher instance with lazy initialization"""
    global photo_searcher
    if photo_searcher is None:
        photo_searcher = UltimatePhotoSearcher()
    return photo_searcher

# Pydantic models for API requests/responses
class SearchRequest(BaseModel):
    query: Optional[str] = Field(None, description="Search query text")
    person: Optional[str] = Field(None, description="Person name to search for")
    group: Optional[str] = Field(None, description="Group name to search for")
    relationship: Optional[str] = Field(None, description="Relationship type to search for")
    time_filter: Optional[str] = Field(None, description="Time expression (e.g., 'last month')")
    limit: int = Field(10, description="Maximum number of results")

class IndexRequest(BaseModel):
    directory: str = Field(..., description="Directory path to index")
    recursive: bool = Field(True, description="Index subdirectories recursively")

class LabelPersonRequest(BaseModel):
    cluster_id: str = Field(..., description="Face cluster ID")
    name: str = Field(..., description="Person name")

class CreateGroupRequest(BaseModel):
    group_name: str = Field(..., description="Group name")
    cluster_ids: List[str] = Field(..., description="List of cluster IDs to include")

class PhotoResponse(BaseModel):
    id: int
    filename: str
    path: str
    similarity_score: Optional[float] = None
    objects: List[str] = []
    faces: List[Dict[str, Any]] = []
    relationships: List[Dict[str, Any]] = []
    timestamp: Optional[str] = None

class StatsResponse(BaseModel):
    total_photos: int
    total_faces: int
    total_clusters: int
    total_relationships: int
    total_groups: int
    database_size_mb: float

class ClusterResponse(BaseModel):
    cluster_id: str
    label: Optional[str]
    photo_count: int
    sample_photos: List[str]

class LabelRequest(BaseModel):
    cluster_id: str
    name: str

class GroupRequest(BaseModel):
    group_name: str
    cluster_ids: List[str]

# Background task tracking
background_tasks_status = {}

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    logger.info("Starting AI Photo Search API...")
    
    # Mount static files for image serving
    if os.path.exists("sample_photos"):
        app.mount("/images", StaticFiles(directory="sample_photos"), name="images")
    
    logger.info("API started successfully")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "AI Photo Search API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running"
    }

@app.get("/api/status")
async def get_status():
    """Get API status and health check"""
    try:
        db = get_database()
        # Test database connection
        stats = api_helpers.get_stats()
        return {
            "status": "healthy",
            "database": "connected",
            "photos_indexed": stats.get("photos", 0)
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "unhealthy", "error": str(e)}
        )

@app.post("/api/index")
async def index_photos(request: IndexRequest, background_tasks: BackgroundTasks):
    """Index photos from a directory"""
    try:
        if not os.path.exists(request.directory):
            raise HTTPException(status_code=404, detail="Directory not found")
        
        # Start background indexing task
        task_id = f"index_{len(background_tasks_status)}"
        background_tasks_status[task_id] = {"status": "running", "progress": 0}
        
        # Add background task
        background_tasks.add_task(
            index_photos_background, 
            task_id, 
            request.directory, 
            request.recursive
        )
        
        return {
            "message": "Indexing started",
            "task_id": task_id,
            "directory": request.directory
        }
    except Exception as e:
        logger.error(f"Index error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def index_photos_background(task_id: str, directory: str, recursive: bool):
    """Background task for photo indexing"""
    try:
        db = get_database()
        
        # Get list of files to process
        photo_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
        photo_files = []
        
        if recursive:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if Path(file).suffix.lower() in photo_extensions:
                        photo_files.append(os.path.join(root, file))
        else:
            photo_files = [
                os.path.join(directory, f) for f in os.listdir(directory)
                if Path(f).suffix.lower() in photo_extensions
            ]
        
        total_files = len(photo_files)
        background_tasks_status[task_id]["total"] = total_files
        
        # Index each photo using the loaded searcher instance
        for i, photo_path in enumerate(photo_files):
            try:
                # Get the searcher instance (already loaded with models)
                searcher = get_photo_searcher()
                
                if searcher:
                    # Use the searcher's _process_single_image method directly
                    logger.info(f"Indexing photo {i+1}/{total_files}: {photo_path}")
                    
                    # Process the photo using the loaded models
                    result = searcher._process_single_image(photo_path)
                    
                    if result == "processed":
                        logger.info(f"Successfully indexed: {photo_path}")
                    elif result == "skipped":
                        logger.info(f"Already indexed: {photo_path}")
                    else:
                        logger.warning(f"Unexpected result for {photo_path}: {result}")
                else:
                    logger.error("Searcher instance not available")
                
                progress = int((i + 1) / total_files * 100)
                background_tasks_status[task_id]["progress"] = progress
                
                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.2)
                
            except Exception as e:
                logger.warning(f"Failed to index {photo_path}: {e}")
        
        background_tasks_status[task_id]["status"] = "completed"
        logger.info(f"Indexing completed: {total_files} photos processed")
        
    except Exception as e:
        background_tasks_status[task_id]["status"] = "failed"
        background_tasks_status[task_id]["error"] = str(e)
        logger.error(f"Background indexing failed: {e}")

@app.get("/api/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Get background task status"""
    if task_id not in background_tasks_status:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return background_tasks_status[task_id]

@app.post("/api/search")
async def search_photos(request: SearchRequest):
    """Search photos using various methods"""
    try:
        searcher = get_photo_searcher()
        results = []
        
        # Parse time filter if provided
        time_filter = None
        if request.time_filter:
            time_filter = temporal_parser.parse_time_expression(request.time_filter)
        
        # For now, use the main search method from the searcher
        # TODO: Implement specific search methods for person, group, relationship
        
        if request.query:
            # Use the main search_photos method
            results = searcher.search_photos(
                query=request.query,
                limit=request.limit,
                show_results=False,  # Don't show visual results in API
                time_filter=request.time_filter
            )
        else:
            # Browse recent photos
            results = api_helpers.get_recent_photos(limit=request.limit)
        
        # Convert results to API format
        photo_responses = []
        for result in results:
            # Parse objects if it's a string
            objects = result.get("objects", [])
            if isinstance(objects, str):
                objects = [obj.strip() for obj in objects.split(",") if obj.strip()]
            
            # Parse faces if it's a string
            faces = result.get("faces", [])
            if isinstance(faces, str):
                try:
                    import json
                    faces = json.loads(faces)
                except:
                    faces = []
            
            photo_response = PhotoResponse(
                id=result.get("id", 0),
                filename=os.path.basename(result.get("path", "")),
                path=result.get("path", ""),
                similarity_score=result.get("similarity", 0.0),
                objects=objects,
                faces=faces,
                relationships=result.get("relationships", []),
                timestamp=result.get("timestamp")
            )
            photo_responses.append(photo_response)
        
        return {
            "results": photo_responses,
            "total": len(photo_responses),
            "query": request.query,
            "search_method": _get_search_method(request)
        }
        
    except Exception as e:
        logger.error(f"Search error: {e}\n{traceback.format_exc()}")
        # Fallback to simple search
        try:
            if request.query:
                results = api_helpers.search_photos_simple(
                    query=request.query,
                    limit=request.limit
                )
                photo_responses = [
                    PhotoResponse(
                        id=result.get("id", 0),
                        filename=os.path.basename(result.get("path", "")),
                        path=result.get("path", ""),
                        similarity_score=result.get("similarity", 0.0),
                        objects=result.get("objects", []),
                        faces=[],
                        relationships=[],
                        timestamp=result.get("timestamp")
                    )
                    for result in results
                ]
                return {
                    "results": photo_responses,
                    "total": len(photo_responses),
                    "query": request.query,
                    "search_method": "simple_fallback"
                }
            else:
                return {
                    "results": [],
                    "total": 0,
                    "query": request.query,
                    "search_method": "none"
                }
        except Exception as fallback_error:
            logger.error(f"Fallback search also failed: {fallback_error}")
            raise HTTPException(status_code=500, detail=str(e))

def _get_search_method(request: SearchRequest) -> str:
    """Determine the search method used"""
    if request.relationship:
        return "relationship"
    elif request.group:
        return "group"
    elif request.person:
        return "person"
    elif request.query:
        return "semantic"
    else:
        return "browse"

@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    try:
        stats = api_helpers.get_stats()
        
        # Calculate database size
        db_path = "photos.db"
        db_size_mb = 0.0
        if os.path.exists(db_path):
            db_size_mb = os.path.getsize(db_path) / (1024 * 1024)
        
        return StatsResponse(
            total_photos=stats.get("photos", 0),
            total_faces=stats.get("faces", 0),
            total_clusters=stats.get("clusters", 0),
            total_relationships=stats.get("relationships", 0),
            total_groups=stats.get("groups", 0),
            database_size_mb=round(db_size_mb, 2)
        )
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/photos/{photo_id}")
@app.get("/api/photos/{photo_id}")
async def get_photo_details(photo_id: int):
    """Get detailed information about a specific photo"""
    try:
        photo = api_helpers.get_photo_by_id(photo_id)

        if not photo:
            raise HTTPException(status_code=404, detail="Photo not found")

        return PhotoResponse(
            id=photo["id"],
            filename=os.path.basename(photo["path"]),
            path=photo["path"],
            objects=photo.get("objects", []),
            faces=photo.get("faces", []),
            relationships=photo.get("relationships", []),
            timestamp=photo.get("timestamp")
        )

    except HTTPException as e:
        raise e  # ✅ let FastAPI handle 404
    except Exception as e:
        logger.error(f"Photo details error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post("/api/faces/cluster")
async def cluster_faces(background_tasks: BackgroundTasks):
    """Start face clustering process"""
    try:
        task_id = f"cluster_{len(background_tasks_status)}"
        background_tasks_status[task_id] = {"status": "running", "progress": 0}
        
        background_tasks.add_task(cluster_faces_background, task_id)
        
        return {
            "message": "Face clustering started",
            "task_id": task_id
        }
        
    except Exception as e:
        logger.error(f"Cluster faces error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def cluster_faces_background(task_id: str):
    """Background task for face clustering"""
    try:
        # Import and run clustering (keep existing logic)
        import subprocess
        result = subprocess.run(
            ["python", "final_photo_search.py", "--cluster-faces"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            background_tasks_status[task_id]["status"] = "completed"
        else:
            background_tasks_status[task_id]["status"] = "failed"
            background_tasks_status[task_id]["error"] = result.stderr
            
    except Exception as e:
        background_tasks_status[task_id]["status"] = "failed"
        background_tasks_status[task_id]["error"] = str(e)

@app.get("/api/faces/clusters")
async def list_face_clusters():
    """List all face clusters"""
    try:
        clusters = api_helpers.get_face_clusters()
        
        cluster_responses = []
        for cluster in clusters:
            # Extract sample photos - handle both dict and string formats
            sample_photos = cluster.get("sample_photos", [])
            sample_paths = []
            
            if sample_photos:
                # If it's a list of dicts, extract paths
                if isinstance(sample_photos[0], dict):
                    sample_paths = [photo.get("path", "") for photo in sample_photos[:5]]
                else:
                    # If it's already a list of strings, use as is
                    sample_paths = sample_photos[:5]
            
            cluster_response = ClusterResponse(
                cluster_id=cluster["cluster_id"],
                label=cluster.get("label"),
                photo_count=cluster.get("photo_count", 0),
                sample_photos=sample_paths
            )
            cluster_responses.append(cluster_response)
        
        return {"clusters": cluster_responses}
        
    except Exception as e:
        logger.error(f"List clusters error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/faces/label")
async def label_person(request: LabelPersonRequest):
    """Label a face cluster with a person name"""
    try:
        db = get_database()
        
        # Use existing CLI functionality
        import subprocess
        result = subprocess.run([
            "python", "final_photo_search.py", 
            "--label-person", request.cluster_id, request.name
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            return {
                "message": f"Labeled cluster {request.cluster_id} as '{request.name}'",
                "cluster_id": request.cluster_id,
                "name": request.name
            }
        else:
            raise HTTPException(status_code=400, detail=result.stderr)
            
    except Exception as e:
        logger.error(f"Label person error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/groups/create")
# ...existing code...

def create_group(self, group_name: str, cluster_ids: list) -> bool:
        """Create a new group with given cluster IDs"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                created_at = datetime.datetime.utcnow().isoformat()
                cursor.execute("""
                    INSERT INTO groups (group_name, cluster_ids, created_at)
                    VALUES (?, ?, ?)
                """, (group_name, json.dumps(cluster_ids), created_at))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error creating group {group_name}: {e}")
            return False

# ...existing code...

@app.post("/api/relationships/build")
async def build_relationships(background_tasks: BackgroundTasks):
    """Build relationship graph"""
    try:
        task_id = f"relationships_{len(background_tasks_status)}"
        background_tasks_status[task_id] = {"status": "running", "progress": 0}
        
        background_tasks.add_task(build_relationships_background, task_id)
        
        return {
            "message": "Building relationships started",
            "task_id": task_id
        }
        
    except Exception as e:
        logger.error(f"Build relationships error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def build_relationships_background(task_id: str):
    """Background task for building relationships"""
    try:
        import subprocess
        
        # Build relationships
        result1 = subprocess.run([
            "python", "final_photo_search.py", "--build-relationships"
        ], capture_output=True, text=True)
        
        if result1.returncode != 0:
            raise Exception(result1.stderr)
        
        # Infer relationships
        result2 = subprocess.run([
            "python", "final_photo_search.py", "--infer-relationships"
        ], capture_output=True, text=True)
        
        if result2.returncode == 0:
            background_tasks_status[task_id]["status"] = "completed"
        else:
            background_tasks_status[task_id]["status"] = "failed"
            background_tasks_status[task_id]["error"] = result2.stderr
            
    except Exception as e:
        background_tasks_status[task_id]["status"] = "failed"
        background_tasks_status[task_id]["error"] = str(e)

@app.get("/api/groups")
async def list_groups():
    """List all groups"""
    try:
        groups = api_helpers.get_groups()
        return {"groups": groups}
    except Exception as e:
        logger.error(f"List groups error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/relationships")
async def list_relationships():
    """List all relationships"""
    try:
        relationships = api_helpers.get_relationships()
        return {"relationships": relationships}
    except Exception as e:
        logger.error(f"List relationships error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/images/{filename}")
async def serve_image(filename: str):
    """Serve image files"""
    image_path = os.path.join("sample_photos", filename)
    
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    
    return FileResponse(image_path)

# ...existing code...

@app.get("/api/faces/clusters")
async def list_face_clusters():
    """List all face clusters (people)"""
    # Call synchronous helper directly
    return {"clusters": api_helpers.get_face_clusters()}

@app.post("/api/faces/clusters/{cluster_id}/label")
async def label_face_cluster(cluster_id: str, request: LabelRequest):
    """Label a person (face cluster)"""
    # Update cluster label in database
    api_helpers.label_face_cluster(cluster_id, request.name)
    return {"success": True}

@app.get("/api/groups")
async def list_groups():
    """List people groups (family, friends)"""
    return {"groups": api_helpers.get_groups()}

@app.post("/api/groups")
async def create_group(request: GroupRequest):
    """Create new people group"""
    api_helpers.create_group(request.group_name, request.cluster_ids)
    return {"success": True}

@app.get("/api/relationships")
async def list_relationships():
    """List discovered relationships"""
    return {"relationships": api_helpers.get_relationships()}

@app.get("/api/relationships/{cluster_id}")
async def get_relationships_for_person(cluster_id: str):
    """Get relationships for a person"""
    return {"relationships": api_helpers.get_relationships_for_person(cluster_id)}

# ...existing code...


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
