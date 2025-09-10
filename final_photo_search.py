#!/usr/bin/env python3
"""
🔍 Ultimate On-Device Photo Search System
==========================================
A complete AI-powered photo search system that combines:
- CLIP embeddings for semantic search
- YOLO object detection
- Face detection and recognition
- Natural language queries
- Visual results display

Usage:
  python final_photo_search.py --index [directory]    # Index photos
  python final_photo_search.py --search "query"       # Search photos
  python final_photo_search.py --stats                # Show database stats
"""

import os
import sys
import argparse
import sqlite3
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')
import hashlib
import json
from sklearn.cluster import DBSCAN

# Import our components
from clip_model import CLIPEmbeddingExtractor
from photo_database import PhotoDatabase
from temporal_search import TemporalParser
import numpy as np

# For visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from matplotlib.patches import Rectangle
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️ matplotlib not available - results will be text-only")

# For YOLO object detection
try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False
    print("⚠️ YOLO not available - object detection disabled")

# For face detection - Advanced models
try:
    from advanced_face_detection import AdvancedFaceDetector
    HAS_ADVANCED_FACE = True
except ImportError:
    HAS_ADVANCED_FACE = False
    
# Fallback face detection
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    print("⚠️ OpenCV not available - face detection disabled")

try:
    import face_recognition  # type: ignore
    HAS_FACE_RECOGNITION = True
except ImportError:
    HAS_FACE_RECOGNITION = False

class UltimatePhotoSearcher:
    """
    The ultimate photo search system combining all AI features
    """
    
    def __init__(self, db_path: str = "photos.db"):
        """Initialize the complete photo search system"""
        print("🚀 Initializing Ultimate Photo Search System...")
        
        # Initialize database with all enhancements
        self.db = PhotoDatabase(db_path)
        self._add_enhanced_columns()
        
        # Initialize CLIP model
        print("🔄 Loading CLIP model...")
        self.clip_extractor = CLIPEmbeddingExtractor()
        
        # Initialize temporal parser
        print("🕒 Initializing temporal intelligence...")
        self.temporal_parser = TemporalParser()
        
        # Initialize YOLO if available
        self.yolo_model = None
        if HAS_YOLO:
            try:
                print("🎯 Loading YOLO model...")
                # Upgrade to YOLOv8x for much better accuracy (68M params vs 3M)
                self.yolo_model = YOLO('yolov8x.pt')
                print("✅ YOLO object detection ready! (YOLOv8x - High Accuracy)")
            except Exception as e:
                print(f"⚠️ YOLO loading failed: {e}")
                self.yolo_model = None
        
        # Initialize face detection - Use advanced models
        self.face_detector = None
        self.face_cascade = None  # Keep for backward compatibility
        
        if HAS_ADVANCED_FACE:
            try:
                print("🎯 Loading advanced face detection...")
                self.face_detector = AdvancedFaceDetector()
                print("✅ Advanced face detection ready!")
            except Exception as e:
                print(f"⚠️ Advanced face detection failed: {e}")
                self._init_fallback_face_detection()
        else:
            self._init_fallback_face_detection()
        
        print("✅ Ultimate Photo Search System ready!")
    
    def _init_fallback_face_detection(self):
        """Initialize fallback face detection"""
        if HAS_OPENCV:
            try:
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'  # type: ignore
                if os.path.exists(cascade_path):
                    self.face_cascade = cv2.CascadeClassifier(cascade_path)
                    print("✅ Fallback face detection ready!")
                else:
                    print("⚠️ Face cascade file not found")
            except Exception as e:
                print(f"⚠️ Face detection setup failed: {e}")
        else:
            print("⚠️ No face detection available")

    # -------- Orientation helpers --------
    def _load_image_upright(self, path: str):
        """Load image honoring EXIF orientation and return (np_array, orientation_code)."""
        try:
            from PIL import Image, ImageOps, ExifTags
            img = Image.open(path)
            exif = img.getexif()
            orientation = 1
            if exif:
                for k, v in exif.items():
                    tag = ExifTags.TAGS.get(k, k)
                    if tag == 'Orientation':
                        orientation = v
                        break
            img_upright = ImageOps.exif_transpose(img)
            import numpy as _np
            return _np.array(img_upright), orientation
        except Exception:
            # Fallback to matplotlib loader (no orientation)
            return mpimg.imread(path), 1

    def _transform_bbox(self, bbox, img_w: int, img_h: int):
        """Given bbox which may be [x1,y1,x2,y2] or [x,y,w,h], normalize to [x, y, w, h]."""
        if bbox is None:
            return None
        if len(bbox) != 4:
            return None
        x1, y1, x2y, y2h = bbox
        # Heuristic: if third value > first, assume [x1,y1,x2,y2]
        if x2y > x1 and y2h > y1:
            x2, y2 = x2y, y2h
            return [x1, y1, x2 - x1, y2 - y1]
        else:
            # [x, y, w, h]
            return [x1, y1, x2y, y2h]

    def _apply_orientation_to_bbox(self, bbox_xywh, img_w: int, img_h: int, orientation: int):
        """Adjust [x,y,w,h] bbox from original image coordinates to upright image coords.
        Orientation codes: 1=normal, 3=180, 6=90 CW, 8=90 CCW.
        """
        x, y, w, h = bbox_xywh
        if orientation == 3:
            # rotate 180
            nx = img_w - (x + w)
            ny = img_h - (y + h)
            return [nx, ny, w, h]
        elif orientation == 6:
            # rotate 90 CW
            nx = img_h - (y + h)
            ny = x
            return [nx, ny, h, w]
        elif orientation == 8:
            # rotate 90 CCW
            nx = y
            ny = img_w - (x + w)
            return [nx, ny, h, w]
        else:
            return [x, y, w, h]
    
    def _add_enhanced_columns(self):
        """Add enhanced columns to the database"""
        import sqlite3
        
        # Add objects column for YOLO detection
        try:
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()
            cursor.execute("ALTER TABLE photos ADD COLUMN objects TEXT")
            conn.commit()
            conn.close()
            print("✅ Added 'objects' column")
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        # Add faces column for face detection
        try:
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()
            cursor.execute("ALTER TABLE photos ADD COLUMN faces TEXT")
            conn.commit()
            conn.close()
            print("✅ Added 'faces' column")
        except sqlite3.OperationalError:
            pass  # Column already exists
    
    def index_photos(self, directory: str, batch_size: int = 10) -> Dict[str, int]:
        """
        Index all photos in a directory with complete AI analysis
        """
        print(f"\n📁 Indexing photos from: {directory}")
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = []
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if Path(file).suffix.lower() in image_extensions:
                    image_files.append(os.path.join(root, file))
        
        print(f"📸 Found {len(image_files)} images to index")
        
        if not image_files:
            print("❌ No images found!")
            return {"total": 0, "processed": 0, "skipped": 0, "errors": 0}
        
        stats = {"total": len(image_files), "processed": 0, "skipped": 0, "errors": 0}
        
        # Process in batches
        for i in range(0, len(image_files), batch_size):
            batch = image_files[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(image_files) + batch_size - 1) // batch_size
            
            print(f"\n🔄 Processing batch {batch_num}/{total_batches} ({len(batch)} images)...")
            
            for image_path in batch:
                try:
                    result = self._process_single_image(image_path)
                    if result == "skipped":
                        stats["skipped"] += 1
                    else:
                        stats["processed"] += 1
                    
                    # Progress indicator
                    progress = (stats["processed"] + stats["skipped"] + stats["errors"]) / stats["total"] * 100
                    print(f"  ✅ {os.path.basename(image_path)} - {progress:.1f}% complete")
                    
                except Exception as e:
                    print(f"  ❌ Error processing {os.path.basename(image_path)}: {e}")
                    stats["errors"] += 1
        
        print(f"\n📊 Indexing complete!")
        print(f"   Total: {stats['total']}")
        print(f"   Processed: {stats['processed']}")
        print(f"   Skipped: {stats['skipped']}")
        print(f"   Errors: {stats['errors']}")
        
        return stats
    
    def _process_single_image(self, image_path: str) -> str:
        """Process a single image with all AI enhancements"""
        # Generate photo ID
        photo_id = hashlib.md5(image_path.encode()).hexdigest()[:16]
        
        # Check if already indexed
        existing = self.db.get_photo_by_id(photo_id)
        if existing:
            return "skipped"
        
        # Extract CLIP embedding
        clip_embedding = self.clip_extractor.get_clip_image_embedding(image_path)
        
        # Get file info
        file_stats = os.stat(image_path)
        timestamp = int(file_stats.st_mtime)
        file_size = file_stats.st_size
        
        # Insert basic photo info
        self.db.insert_photo(
            photo_id=photo_id,
            path=image_path,
            timestamp=timestamp,
            embedding=clip_embedding,
            file_size=file_size
        )
        
        # Detect objects with YOLO
        if self.yolo_model:
            objects_data = self._detect_objects(image_path)
            if objects_data:
                object_names = [obj['class'] for obj in objects_data]
                self._update_photo_objects(photo_id, object_names)
        
        # Detect faces (advanced or fallback)
        if self.face_detector or self.face_cascade:
            faces_data = self._detect_faces(image_path)
            if faces_data:
                self._update_photo_faces(photo_id, faces_data)
                # Persist faces into Stage 2 normalized table
                for idx, face in enumerate(faces_data):
                    try:
                        face_id = hashlib.md5(f"{photo_id}:{idx}:{face.get('bbox')}".encode()).hexdigest()[:16]
                        bbox = face.get('bbox')
                        # bbox can be [x1,y1,x2,y2] or [x,y,w,h]; store as JSON string
                        bbox_str = json.dumps(bbox) if bbox is not None else json.dumps([])
                        embedding = None
                        enc = face.get('encoding')
                        if isinstance(enc, list):
                            embedding = np.array(enc, dtype=np.float32)
                        elif isinstance(enc, np.ndarray):
                            embedding = enc.astype(np.float32)
                        method = face.get('method')
                        self.db.insert_face(face_id, photo_id, bbox_str, embedding, method)
                    except Exception as e:
                        print(f"⚠️ Failed to persist face for {image_path}: {e}")
        
        return "processed"
    
    def _update_photo_objects(self, photo_id: str, objects: List[str]):
        """Update objects for a photo"""
        import sqlite3
        objects_str = ','.join(objects)
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        cursor.execute("UPDATE photos SET objects = ? WHERE id = ?", (objects_str, photo_id))
        conn.commit()
        conn.close()
    
    def _update_photo_faces(self, photo_id: str, faces_data: List[Dict]):
        """Update faces for a photo"""
        import sqlite3
        import json
        faces_str = json.dumps(faces_data)
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        cursor.execute("UPDATE photos SET faces = ? WHERE id = ?", (faces_str, photo_id))
        conn.commit()
        conn.close()
    
    def _detect_objects(self, image_path: str) -> List[Dict]:
        """Detect objects using YOLO"""
        if not self.yolo_model:
            return []
        
        try:
            results = self.yolo_model(image_path, verbose=False)
            objects = []
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        confidence = float(box.conf.cpu().numpy()[0])
                        if confidence > 0.5:  # Confidence threshold
                            class_id = int(box.cls.cpu().numpy()[0])
                            class_name = self.yolo_model.names[class_id]
                            bbox = box.xyxy.cpu().numpy()[0].tolist()  # [x1, y1, x2, y2]
                            
                            objects.append({
                                'class': class_name,
                                'confidence': confidence,
                                'bbox': bbox
                            })
            
            return objects
        except Exception as e:
            print(f"⚠️ Object detection failed for {image_path}: {e}")
            return []
    
    def _detect_faces(self, image_path: str) -> List[Dict]:
        """Detect faces using advanced face detection or fallback methods"""
        # Try advanced face detection first
        if self.face_detector:
            try:
                faces = self.face_detector.detect_faces(image_path)
                
                # Convert to our format
                face_data = []
                for face in faces:
                    face_info = {
                        'bbox': face['bbox'],
                        'confidence': face['confidence'],
                        'method': 'advanced',
                        'encoding': face.get('embedding'),
                        'age': face.get('age'),
                        'gender': face.get('gender')
                    }
                    face_data.append(face_info)
                
                if face_data:
                    print(f"👤 Advanced face detection: {len(face_data)} faces")
                    return face_data
                    
            except Exception as e:
                print(f"⚠️ Advanced face detection failed: {e}")
        
        # Fallback to basic face detection
        if not self.face_cascade:
            return []
        
        try:
            # Load image
            if HAS_OPENCV:
                image = cv2.imread(image_path)
                if image is None:
                    return []
                
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                # Detect faces with OpenCV
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                
                face_data = []
                for (x, y, w, h) in faces:
                    face_info = {
                        'bbox': [int(x), int(y), int(w), int(h)],
                        'method': 'opencv',
                        'confidence': 0.8  # OpenCV doesn't provide confidence
                    }
                    
                    # Try to get face encoding if face_recognition is available
                    if HAS_FACE_RECOGNITION:
                        try:
                            # Convert BGR to RGB for face_recognition
                            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            face_locations = [(y, x+w, y+h, x)]  # face_recognition format
                            encodings = face_recognition.face_encodings(rgb_image, face_locations)
                            if encodings:
                                face_info['encoding'] = encodings[0].tolist()
                                face_info['method'] = 'face_recognition'
                        except Exception:
                            pass  # Fall back to OpenCV detection only
                    
                    face_data.append(face_info)
                
                return face_data
            
        except Exception as e:
            print(f"⚠️ Face detection failed for {image_path}: {e}")
            return []
        
        return []
    
    def search_photos(self, query: str, limit: int = 10, show_results: bool = True, 
                     time_filter: Optional[str] = None) -> List[Dict]:
        """
        Search photos using natural language queries with optional time filtering
        
        Args:
            query: Natural language search query
            limit: Maximum number of results to return
            show_results: Whether to display visual results
            time_filter: Optional time expression like "2020", "last year", "last Christmas"
        """
        print(f"\n🔍 Searching for: '{query}'")
        
        # Parse time filter if provided
        time_range_str = "No time filter"
        if time_filter:
            start_ts, end_ts = self.temporal_parser.parse_time_expression(time_filter)
            time_range_str = self.temporal_parser.format_timestamp_range(start_ts, end_ts)
            print(f"🕒 Time filter: {time_range_str}")
        else:
            start_ts, end_ts = None, None
        
        # Get text embedding for query
        query_embedding = self.clip_extractor.get_clip_text_embedding(query)
        
        # Search database with temporal filtering
        results = self._search_by_similarity(query_embedding, limit, start_ts, end_ts)
        
        if not results:
            print("❌ No matching photos found")
            if time_filter:
                print(f"   Try expanding your time range or removing the time filter")
            return []
        
        print(f"✅ Found {len(results)} matching photos ({time_range_str}):")
        
        # Display results
        for i, result in enumerate(results, 1):
            print(f"\n{i}. 📸 {os.path.basename(result['path'])}")
            print(f"   🎯 Similarity: {result['similarity']:.3f}")
            
            # Show photo date if available
            if result.get('exif_date'):
                print(f"   📅 Photo date: {result['exif_date'][:10]}")  # Show just the date part
            elif result.get('created_date'):
                print(f"   📅 File date: {result['created_date'][:10]}")
            
            # Show detected objects
            if result.get('objects'):
                try:
                    objects_str = result['objects']
                    if objects_str and objects_str != 'None':
                        # Handle different object format possibilities
                        if objects_str.startswith('['):
                            # List format from YOLO detection  
                            objects = eval(objects_str)
                            if isinstance(objects, list) and objects:
                                if isinstance(objects[0], dict):
                                    obj_names = [obj['class'] for obj in objects if 'class' in obj]
                                else:
                                    obj_names = objects
                                if obj_names:
                                    print(f"   🎯 Objects: {', '.join(set(obj_names))}")
                        else:
                            # String format
                            obj_names = objects_str.split(',')
                            if obj_names and obj_names[0]:
                                print(f"   🎯 Objects: {', '.join([obj.strip() for obj in obj_names])}")
                except Exception as e:
                    print(f"   🎯 Objects: (error parsing: {e})")
            
            # Show detected faces
            if result.get('faces'):
                try:
                    faces_str = result['faces']
                    if faces_str and faces_str != 'None':
                        if faces_str.startswith('['):
                            faces = eval(faces_str)
                            if isinstance(faces, list):
                                print(f"   👤 Faces: {len(faces)} detected")
                        else:
                            print(f"   👤 Faces: detected")
                except Exception as e:
                    print(f"   👤 Faces: (error parsing: {e})")
        
        # Show visual results if possible
        if show_results and HAS_MATPLOTLIB and len(results) > 0:
            self._display_results(results, query)
        
        return results
    
    def _search_by_similarity(self, query_embedding: np.ndarray, limit: int, 
                             start_timestamp: Optional[int] = None, 
                             end_timestamp: Optional[int] = None) -> List[Dict]:
        """
        Search photos by embedding similarity with optional temporal filtering
        
        Args:
            query_embedding: CLIP embedding of the search query
            limit: Maximum number of results to return
            start_timestamp: Start of time range filter (Unix timestamp)
            end_timestamp: End of time range filter (Unix timestamp)
        """
        # Get photos in time range if temporal filter is applied
        if start_timestamp is not None or end_timestamp is not None:
            # Get photos filtered by time
            time_filtered_photos = self.db.search_photos_by_time(start_timestamp, end_timestamp, use_exif=True)
            if not time_filtered_photos:
                # Try with file timestamps if no EXIF results
                time_filtered_photos = self.db.search_photos_by_time(start_timestamp, end_timestamp, use_exif=False)
            
            if not time_filtered_photos:
                return []
            
            # Get embeddings only for time-filtered photos
            photo_ids_in_range = {photo_id for photo_id, _, _, _ in time_filtered_photos}
            all_embeddings = self.db.get_all_embeddings()
            filtered_embeddings = [(pid, path, emb) for pid, path, emb in all_embeddings if pid in photo_ids_in_range]
        else:
            # Get all embeddings from database
            filtered_embeddings = self.db.get_all_embeddings()
        
        if not filtered_embeddings:
            return []
        
        # Calculate similarities
        similarities = []
        for photo_id, path, photo_embedding in filtered_embeddings:
            # Cosine similarity
            similarity = np.dot(query_embedding, photo_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(photo_embedding)
            )
            
            # Get additional info
            photo_info = self.db.get_photo_by_id(photo_id)
            objects = None
            faces = None
            
            if photo_info:
                # Try to get objects and faces from the enhanced database
                try:
                    import sqlite3
                    conn = sqlite3.connect(self.db.db_path)
                    cursor = conn.cursor()
                    cursor.execute("SELECT objects, faces FROM photos WHERE id = ?", (photo_id,))
                    result = cursor.fetchone()
                    if result:
                        objects, faces = result
                    conn.close()
                except:
                    pass
            
            similarities.append({
                'path': path,
                'similarity': float(similarity),
                'objects': objects,
                'faces': faces,
                'exif_date': photo_info.get('exif_date') if photo_info else None,
                'created_date': photo_info.get('created_date') if photo_info else None
            })
        
        # Sort by similarity
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similarities[:limit]
    
    def _display_results(self, results: List[Dict], query: str):
        """Display search results visually using matplotlib"""
        if not HAS_MATPLOTLIB:
            return
        
        n_results = min(len(results),10)  # Max 6 images
        cols = min(3, n_results)
        rows = (n_results + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        fig.suptitle(f'Search Results for: "{query}"', fontsize=16, fontweight='bold')
        
        for i in range(n_results):
            result = results[i]
            
            try:
                # Load and display image honoring EXIF orientation
                img, orientation = self._load_image_upright(result['path'])
                img_h, img_w = img.shape[0], img.shape[1]
                axes[i].imshow(img)
                axes[i].axis('off')
                
                # Add title with similarity score
                title = f"{os.path.basename(result['path'])}\nSimilarity: {result['similarity']:.3f}"
                axes[i].set_title(title, fontsize=10)
                
                # Add object detection boxes if available
                if result.get('objects'):
                    try:
                        objects = eval(result['objects']) if isinstance(result['objects'], str) else result['objects']
                        for obj in objects:
                            if 'bbox' in obj:
                                raw = obj['bbox']
                                bb = self._transform_bbox(raw, img_w, img_h)
                                if bb is None:
                                    continue
                                x, y, w, h = self._apply_orientation_to_bbox(bb, img_w, img_h, orientation)
                                rect = Rectangle((x, y), w, h, 
                                               linewidth=2, edgecolor='red', facecolor='none')
                                axes[i].add_patch(rect)
                                
                                # Add label
                                axes[i].text(x, y - 5, f"{obj['class']}", 
                                           fontsize=8, color='red', weight='bold',
                                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
                    except:
                        pass
                
                # Add person-targeted face boxes if provided (Stage 2 person search)
                if result.get('target_faces'):
                    try:
                        # Define colors for different people
                        person_colors = ['lime', 'cyan', 'magenta', 'yellow', 'orange', 'red']
                        person_labels_seen = {}
                        color_index = 0
                        
                        for face in result['target_faces']:
                            raw = face.get('bbox')
                            person_label = face.get('person_label', 'Unknown')
                            
                            if raw:
                                bb = self._transform_bbox(raw, img_w, img_h)
                                if bb is None:
                                    continue
                                    
                                # Assign color to person if not already assigned
                                if person_label not in person_labels_seen:
                                    person_labels_seen[person_label] = person_colors[color_index % len(person_colors)]
                                    color_index += 1
                                
                                color = person_labels_seen[person_label]
                                x, y, w, h = self._apply_orientation_to_bbox(bb, img_w, img_h, orientation)
                                rect = Rectangle((x, y), w, h, 
                                               linewidth=3, edgecolor=color, facecolor='none')
                                axes[i].add_patch(rect)
                                axes[i].text(x, y - 5, person_label, 
                                           fontsize=9, color=color, weight='bold',
                                           bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
                    except Exception:
                        pass

                # Add generic face detection boxes if available
                if result.get('faces'):
                    try:
                        faces = eval(result['faces']) if isinstance(result['faces'], str) else result['faces']
                        for face in faces:
                            if 'bbox' in face:
                                raw = face['bbox']
                                bb = self._transform_bbox(raw, img_w, img_h)
                                if bb is None:
                                    continue
                                x, y, w, h = self._apply_orientation_to_bbox(bb, img_w, img_h, orientation)
                                rect = Rectangle((x, y), w, h, 
                                               linewidth=2, edgecolor='blue', facecolor='none')
                                axes[i].add_patch(rect)
                                
                                # Add face label
                                axes[i].text(x, y - 5, "Face", 
                                           fontsize=8, color='blue', weight='bold',
                                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
                    except:
                        pass
                        
            except Exception as e:
                axes[i].text(0.5, 0.5, f"Error loading\n{os.path.basename(result['path'])}", 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].axis('off')
        
        # Hide extra subplots
        for i in range(n_results, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def show_stats(self):
        """Display database statistics"""
        print("\n📊 Photo Database Statistics")
        print("=" * 50)
        # Stage 2: people stats
        try:
            import sqlite3
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM faces")
            total_faces = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM faces WHERE embedding IS NOT NULL")
            faces_with_emb = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM face_clusters")
            num_clusters = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM face_clusters WHERE label IS NOT NULL AND label != ''")
            labeled_clusters = cursor.fetchone()[0]
            print(f"👥 Total faces: {total_faces}")
            print(f"🧠 Faces with embeddings: {faces_with_emb}")
            print(f"🔗 Clusters: {num_clusters} (labeled: {labeled_clusters})")
            # Top labels
            cursor.execute("SELECT label, num_faces FROM face_clusters WHERE label IS NOT NULL AND label != '' ORDER BY num_faces DESC LIMIT 5")
            rows = cursor.fetchall()
            if rows:
                print("🏷️ Top people:")
                for label, cnt in rows:
                    print(f"   {label}: {cnt}")
            conn.close()
        except Exception:
            pass
        
        # Use the database's built-in stats method if available
        try:
            stats = self.db.get_database_stats()
            print(f"📸 Total photos: {stats.get('total_photos', 0)}")
            print(f"🧠 Photos with CLIP embeddings: {stats.get('total_photos', 0)}")
        except:
            # Fallback to manual counting
            import sqlite3
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()
            
            # Total photos
            cursor.execute("SELECT COUNT(*) FROM photos")
            total_photos = cursor.fetchone()[0]
            print(f"📸 Total photos: {total_photos}")
            
            # Photos with embeddings
            cursor.execute("SELECT COUNT(*) FROM photos WHERE clip_embedding IS NOT NULL")
            with_embeddings = cursor.fetchone()[0]
            print(f"🧠 Photos with CLIP embeddings: {with_embeddings}")
            
            # Check if enhanced columns exist
            try:
                cursor.execute("SELECT COUNT(*) FROM photos WHERE objects IS NOT NULL AND objects != ''")
                with_objects = cursor.fetchone()[0]
                print(f"🎯 Photos with object detection: {with_objects}")
            except:
                print(f"🎯 Photos with object detection: N/A")
            
            try:
                cursor.execute("SELECT COUNT(*) FROM photos WHERE faces IS NOT NULL AND faces != ''")
                with_faces = cursor.fetchone()[0]
                print(f"👤 Photos with face detection: {with_faces}")
            except:
                print(f"👤 Photos with face detection: N/A")
            
            # Most common objects
            try:
                cursor.execute("SELECT objects FROM photos WHERE objects IS NOT NULL AND objects != ''")
                all_objects = cursor.fetchall()
                
                if all_objects:
                    object_counts = {}
                    for (objects_str,) in all_objects:
                        try:
                            # Handle both string and list formats
                            if objects_str.startswith('['):
                                objects = eval(objects_str)
                            else:
                                objects = objects_str.split(',')
                            
                            for obj in objects:
                                obj_name = obj.strip().strip("'\"")
                                if obj_name:
                                    object_counts[obj_name] = object_counts.get(obj_name, 0) + 1
                        except:
                            continue
                    
                    if object_counts:
                        print(f"\n🏆 Most detected objects:")
                        sorted_objects = sorted(object_counts.items(), key=lambda x: x[1], reverse=True)
                        for obj, count in sorted_objects[:5]:
                            print(f"   {obj}: {count}")
            except:
                pass
            
            # Database size
            try:
                cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
                db_size = cursor.fetchone()[0]
                print(f"\n💾 Database size: {db_size / 1024 / 1024:.2f} MB")
            except:
                print(f"\n💾 Database size: Unknown")
            
            conn.close()
        
        print("=" * 50)

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="🔍 Ultimate On-Device Photo Search System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python final_photo_search.py --index ./photos           # Index all photos
  python final_photo_search.py --search "red flower"      # Search for red flowers
  python final_photo_search.py --search "person smiling"  # Search for people
  python final_photo_search.py --person "Alice"           # Photos with Alice
  python final_photo_search.py --person "Alice" --person "Bob"  # Photos with BOTH Alice and Bob
  python final_photo_search.py --person "Alice" --search "beach"  # Alice at the beach
  python final_photo_search.py --stats                    # Show database stats
        """
    )
    
    parser.add_argument('--index', type=str, metavar='DIRECTORY',
                       help='Index photos from the specified directory')
    parser.add_argument('--check-photo', type=str, metavar='PATH',
                       help='Debug: check if a photo is in DB and show its record')
    parser.add_argument('--list-photos', action='store_true',
                       help='Debug: list all photos in database with their paths')
    parser.add_argument('--search', type=str, metavar='QUERY',
                       help='Search photos using natural language query')
    parser.add_argument('--time', type=str, metavar='TIME_EXPR',
                       help='Time filter for search (e.g., "2020", "last year", "last Christmas")')
    parser.add_argument('--stats', action='store_true',
                       help='Show database statistics')
    parser.add_argument('--limit', type=int, default=10, metavar='N',
                       help='Maximum number of search results (default: 5)')
    parser.add_argument('--db', type=str, default='photos.db', metavar='PATH',
                       help='Database file path (default: photos.db)')
    # Stage 2 CLI options
    parser.add_argument('--cluster-faces', action='store_true',
                       help='Run DBSCAN clustering on face embeddings')
    parser.add_argument('--cluster-eps', type=float, default=0.4, metavar='EPS',
                       help='DBSCAN eps for face clustering (default: 0.4, cosine)')
    parser.add_argument('--cluster-min-samples', type=int, default=3, metavar='N',
                       help='DBSCAN min_samples for face clustering (default: 3)')
    parser.add_argument('--list-clusters', action='store_true',
                       help='List face clusters (with labels if any)')
    parser.add_argument('--label-person', nargs=2, metavar=('CLUSTER_ID','LABEL'),
                       help='Assign a label to a cluster id')
    parser.add_argument('--rebuild-relationships', action='store_true',
                       help='Recompute co-occurrence relationships between clusters')
    parser.add_argument('--person', type=str, metavar='NAME', action='append',
                       help='Filter search to photos containing labeled person(s). Can be used multiple times for multiple people.')
    parser.add_argument('--backfill-faces', action='store_true',
                       help='Detect and store faces for already-indexed photos')
    parser.add_argument('--assign-new-faces', action='store_true',
                       help='Incrementally assign only new/unclustered faces to nearest existing clusters')
    parser.add_argument('--assign-threshold', type=float, default=0.45, metavar='COSINE',
                       help='Minimum cosine similarity to assign a face to an existing cluster (default: 0.45)')
    parser.add_argument('--assign-photo', type=str, metavar='PATH',
                       help='Only assign faces from a specific photo path')
    
    args = parser.parse_args()
    
    # Check if no meaningful arguments provided (include Stage 2 flags)
    if not any([
        args.index,
        args.check_photo,
        args.list_photos,
        args.search,
        args.stats,
        args.cluster_faces,
        args.list_clusters,
        args.label_person,
        args.rebuild_relationships,
        args.person,
        args.backfill_faces,
        args.assign_new_faces
    ]):
        parser.print_help()
        return
    
    # Initialize the search system
    try:
        searcher = UltimatePhotoSearcher(args.db)
    except Exception as e:
        print(f"❌ Failed to initialize search system: {e}")
        return
    
    # Execute requested action
    try:
        if args.index:
            if not os.path.exists(args.index):
                print(f"❌ Directory not found: {args.index}")
                return
            searcher.index_photos(args.index)
        elif args.check_photo:
            import os as _os
            rec = searcher.db.get_photo_by_path(args.check_photo)
            if rec:
                print("✅ Photo found in DB:")
                print(rec)
            else:
                print("❌ Photo not found in DB")
        elif args.list_photos:
            import sqlite3
            conn = sqlite3.connect(args.db)
            cursor = conn.cursor()
            cursor.execute('SELECT id, path FROM photos ORDER BY path')
            photos = cursor.fetchall()
            conn.close()
            print(f"📸 Found {len(photos)} photos in database:")
            for i, (photo_id, path) in enumerate(photos, 1):
                print(f"  {i:2d}. {photo_id}: {path}")
        
        elif args.backfill_faces:
            backfill_faces(searcher)
        elif args.assign_new_faces:
            assign_new_faces(searcher, threshold=args.assign_threshold, only_photo_path=args.assign_photo)
        elif args.cluster_faces:
            cluster_faces(args.db, eps=args.cluster_eps, min_samples=args.cluster_min_samples)
        elif args.list_clusters:
            list_clusters(args.db)
        elif args.label_person:
            cid, label = args.label_person
            label_cluster(args.db, cid, label)
        elif args.rebuild_relationships:
            rebuild_relationships(args.db)
        elif args.search or args.person:
            # Validate time filter with search
            if args.time and not args.search:
                print("ℹ️ Using time filter without text query; will filter by time only")
            person_labels = args.person
            if person_labels:
                # Narrow search to photos containing these labeled people
                results = search_with_multiple_people(searcher, person_labels, args.search, args.limit, args.time)
                if results is None:
                    print(f"❌ One or more people not found: {person_labels}")
            else:
                searcher.search_photos(args.search or "", limit=args.limit, time_filter=args.time)
        
        elif args.stats:
            searcher.show_stats()
            
    except KeyboardInterrupt:
        print("\n⚠️ Operation cancelled by user")
    except Exception as e:
        print(f"❌ Error: {e}")

def cluster_faces(db_path: str, eps: float = 0.4, min_samples: int = 3):
    try:
        db = PhotoDatabase(db_path)
        face_embeddings = db.get_all_face_embeddings()
        if not face_embeddings:
            print("⚠️ No face embeddings found to cluster")
            return
        face_ids = [fid for fid, _ in face_embeddings]
        X = np.stack([emb / (np.linalg.norm(emb) + 1e-8) for _, emb in face_embeddings]).astype(np.float32)
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        labels = clustering.fit_predict(X)
        clusters = {}
        for fid, lbl in zip(face_ids, labels):
            if lbl < 0:
                continue
            clusters.setdefault(f"cluster_{lbl}", []).append(fid)
        if not clusters:
            print("⚠️ No clusters found (all noise)")
            return
        for cid, members in clusters.items():
            db.assign_cluster_to_faces(cid, members)
            db.upsert_cluster(cid)
        db.build_relationships_from_photos()
        print(f"✅ Clustering complete. Created/updated {len(clusters)} clusters")
    except Exception as e:
        print(f"❌ Clustering failed: {e}")

def list_clusters(db_path: str):
    db = PhotoDatabase(db_path)
    clusters = db.get_clusters()
    if not clusters:
        print("⚠️ No clusters found")
        return
    print("\n👥 Face Clusters")
    print("-" * 40)
    for c in clusters:
        print(f"{c['cluster_id']}  label={c['label'] or '-'}  faces={c['num_faces']}")

def label_cluster(db_path: str, cluster_id: str, label: str):
    db = PhotoDatabase(db_path)
    db.label_cluster(cluster_id, label)
    print(f"✅ Labeled {cluster_id} as '{label}'")

def rebuild_relationships(db_path: str):
    db = PhotoDatabase(db_path)
    db.build_relationships_from_photos()
    print("✅ Rebuilt relationships")

def search_with_person(searcher: UltimatePhotoSearcher, person_label: str, query: Optional[str], limit: int, time_filter: Optional[str]):
    db = searcher.db
    cluster = db.get_cluster_by_label(person_label)
    if not cluster:
        return None
    photo_ids = db.get_photos_with_clusters([cluster['cluster_id']])
    print(f"🔍 Found {len(photo_ids)} photos with cluster {cluster['cluster_id']} (label: {person_label})")
    if not photo_ids:
        print(f"⚠️ No photos found for '{person_label}'")
        return []
    # Apply time filter using existing temporal parser mechanics
    if time_filter:
        start_ts, end_ts = searcher.temporal_parser.parse_time_expression(time_filter)
        time_filtered = db.search_photos_by_time(start_ts, end_ts, use_exif=True) or db.search_photos_by_time(start_ts, end_ts, use_exif=False)
        allowed_ids = {pid for pid, _, _, _ in time_filtered}
        photo_ids = [pid for pid in photo_ids if pid in allowed_ids]
        if not photo_ids:
            print(f"⚠️ No photos for '{person_label}' within time range")
            return []
    # If no query, just print the files
    if not query:
        # Retrieve paths
        all_embs = db.get_all_embeddings()
        lookup = {pid: path for pid, path, _ in all_embs}
        results = []
        for pid in photo_ids:
            path = lookup.get(pid, '')
            if not path:
                continue
            # collect target faces for this photo belonging to the cluster
            target_faces = []
            faces = db.get_faces_by_photo(pid)
            cluster_faces = [f for f in faces if f.get('cluster_id') == cluster['cluster_id']]
            print(f"   📸 {os.path.basename(path)}: {len(cluster_faces)} faces from cluster {cluster['cluster_id']}")
            for f in cluster_faces:
                try:
                    bbox = f.get('bbox')
                    if isinstance(bbox, str):
                        import json as _json
                        bbox = _json.loads(bbox)
                    target_faces.append({'bbox': bbox})
                except Exception:
                    continue
            results.append({'path': path, 'similarity': 1.0, 'target_faces': target_faces})
        # show console list and plot
        for i, r in enumerate(results[:limit], 1):
            print(f"\n{i}. 📸 {os.path.basename(r['path'])}")
        if HAS_MATPLOTLIB and results:
            searcher._display_results(results[:limit], f"Person: {person_label}")
        return results[:limit]
    # With a query, restrict similarity search to these photo_ids
    query_emb = searcher.clip_extractor.get_clip_text_embedding(query)
    all_embs = db.get_all_embeddings()
    filtered = [(pid, path, emb) for pid, path, emb in all_embs if pid in set(photo_ids)]
    similarities = []
    for pid, path, emb in filtered:
        sim = float(np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb)))
        info = db.get_photo_by_id(pid)
        similarities.append({
            'path': path,
            'similarity': sim,
            'exif_date': info.get('exif_date') if info else None,
            'created_date': info.get('created_date') if info else None
        })
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    # Attach target faces for visualization
    top_results = similarities[:limit]
    for r in top_results:
        pid = None
        # find photo id by matching path
        for ppid, ppath, _ in db.get_all_embeddings():
            if ppath == r['path']:
                pid = ppid
                break
        target_faces = []
        if pid:
            for f in db.get_faces_by_photo(pid):
                if f.get('cluster_id') == cluster['cluster_id']:
                    try:
                        bbox = f.get('bbox')
                        if isinstance(bbox, str):
                            import json as _json
                            bbox = _json.loads(bbox)
                        target_faces.append({'bbox': bbox})
                    except Exception:
                        continue
        r['target_faces'] = target_faces
    for i, r in enumerate(top_results, 1):
        print(f"\n{i}. 📸 {os.path.basename(r['path'])}")
        print(f"   🎯 Similarity: {r['similarity']:.3f}")
    if HAS_MATPLOTLIB and top_results:
        searcher._display_results(top_results, f"Person: {person_label} | {query or ''}")
    return top_results

def search_with_multiple_people(searcher: UltimatePhotoSearcher, person_labels: List[str], query: Optional[str], limit: int, time_filter: Optional[str]):
    """Search for photos containing multiple specific people (intersection of all people)."""
    db = searcher.db
    
    # Get clusters for all requested people
    clusters = []
    missing_people = []
    
    for person_label in person_labels:
        cluster = db.get_cluster_by_label(person_label)
        if not cluster:
            missing_people.append(person_label)
        else:
            clusters.append(cluster)
    
    if missing_people:
        print(f"❌ People not found: {', '.join(missing_people)}")
        return None
    
    if not clusters:
        print("❌ No valid people found")
        return None
    
    cluster_ids = [c['cluster_id'] for c in clusters]
    print(f"🔍 Searching for photos containing ALL of: {', '.join(person_labels)}")
    print(f"📊 Using clusters: {cluster_ids}")
    
    # Find photos that contain ALL of the specified people
    # Get photos for each cluster
    all_photo_sets = []
    for cluster_id in cluster_ids:
        photo_ids = db.get_photos_with_clusters([cluster_id])
        all_photo_sets.append(set(photo_ids))
        print(f"   Cluster {cluster_id}: {len(photo_ids)} photos")
    
    # Find intersection - photos that contain ALL people
    if not all_photo_sets:
        print("⚠️ No photos found for any person")
        return []
    
    common_photo_ids = set.intersection(*all_photo_sets)
    print(f"🎯 Found {len(common_photo_ids)} photos containing ALL {len(person_labels)} people")
    
    if not common_photo_ids:
        print("⚠️ No photos found containing all specified people together")
        return []
    
    # Apply time filter if specified
    if time_filter:
        start_ts, end_ts = searcher.temporal_parser.parse_time_expression(time_filter)
        time_filtered = db.search_photos_by_time(start_ts, end_ts, use_exif=True) or db.search_photos_by_time(start_ts, end_ts, use_exif=False)
        allowed_ids = {pid for pid, _, _, _ in time_filtered}
        common_photo_ids = common_photo_ids.intersection(allowed_ids)
        print(f"🕒 After time filter: {len(common_photo_ids)} photos")
        
        if not common_photo_ids:
            print(f"⚠️ No photos with all people within time range")
            return []
    
    # Convert to list for consistent ordering
    photo_ids = list(common_photo_ids)
    
    # If no query, just display the files
    if not query:
        all_embs = db.get_all_embeddings()
        lookup = {pid: path for pid, path, _ in all_embs}
        results = []
        
        for i, pid in enumerate(photo_ids[:limit], 1):
            path = lookup.get(pid, '')
            if not path:
                continue
            
            print(f"\n{i}. 📸 {os.path.basename(path)}")
            
            # Show face count for each person and collect target faces
            all_target_faces = []
            for person_label, cluster_id in zip(person_labels, cluster_ids):
                faces = db.get_faces_by_photo(pid)
                cluster_faces = [f for f in faces if f.get('cluster_id') == cluster_id]
                print(f"   👤 {person_label}: {len(cluster_faces)} faces")
                
                # Add cluster faces to target faces for visual display
                for face in cluster_faces:
                    face_copy = face.copy()
                    face_copy['person_label'] = person_label
                    all_target_faces.append(face_copy)
            
            result_dict = {
                'path': path, 
                'similarity': 1.0,  # No similarity when no query
                'target_faces': all_target_faces,
                'photo_id': pid
            }
            results.append(result_dict)
        
        # Show visual results if matplotlib is available
        if HAS_MATPLOTLIB and results:
            searcher._display_results(results, f"People: {', '.join(person_labels)} | No text query")
        
        return results
    
    # With query: perform semantic search within the filtered photos
    print(f"🔍 Performing semantic search for '{query}' within {len(photo_ids)} photos")
    
    # Get embeddings for the filtered photos
    all_embs = db.get_all_embeddings()
    filtered_embs = [(pid, path, emb) for pid, path, emb in all_embs if pid in common_photo_ids]
    
    if not filtered_embs:
        print("⚠️ No embeddings found for filtered photos")
        return []
    
    # Extract query embedding
    query_embedding = searcher.clip_extractor.get_clip_text_embedding(query)
    print(f"📝 Extracted text embedding: '{query}' -> ({len(query_embedding)},)")
    
    # Calculate similarities
    similarities = []
    for pid, path, photo_emb in filtered_embs:
        sim = np.dot(query_embedding, photo_emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(photo_emb))
        
        # Get photo metadata
        info = db.get_photo_by_id(pid)
        similarities.append({
            'path': path,
            'similarity': sim,
            'exif_date': info.get('exif_date') if info else None,
            'created_date': info.get('created_date') if info else None,
            'photo_id': pid
        })
    
    # Sort by similarity
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    top_results = similarities[:limit]
    
    print(f"✅ Found {len(top_results)} matching photos with all people:")
    
    # Display results
    for i, r in enumerate(top_results, 1):
        print(f"\n{i}. 📸 {os.path.basename(r['path'])}")
        print(f"   🎯 Similarity: {r['similarity']:.3f}")
        if r['exif_date']:
            print(f"   📅 Photo date: {r['exif_date']}")
        
        # Show face details for each person
        pid = r['photo_id']
        all_target_faces = []  # Collect all target faces for visual display
        
        for person_label, cluster_id in zip(person_labels, cluster_ids):
            faces = db.get_faces_by_photo(pid)
            cluster_faces = [f for f in faces if f.get('cluster_id') == cluster_id]
            print(f"   👤 {person_label}: {len(cluster_faces)} faces")
            
            # Add cluster faces to target faces for visual display
            for face in cluster_faces:
                face_copy = face.copy()
                face_copy['person_label'] = person_label
                all_target_faces.append(face_copy)
        
        # Add target faces to result for visual display
        r['target_faces'] = all_target_faces
    
    # Show visual results if matplotlib is available
    if HAS_MATPLOTLIB and top_results:
        searcher._display_results(top_results, f"People: {', '.join(person_labels)} | {query or 'No text query'}")
    
    return top_results

def backfill_faces(searcher: UltimatePhotoSearcher, batch_size: int = 50):
    """Run face detection for photos already in DB that lack face records."""
    db = searcher.db
    all_rows = db.get_all_embeddings()
    total = len(all_rows)
    if total == 0:
        print("⚠️ No photos in database to backfill")
        return
    print(f"🔄 Backfilling faces for {total} photos...")
    processed = 0
    skipped = 0
    for i, (photo_id, path, _) in enumerate(all_rows, 1):
        try:
            existing_faces = db.get_faces_by_photo(photo_id)
            if existing_faces:
                skipped += 1
                continue
            if not os.path.exists(path):
                skipped += 1
                continue
            faces_data = searcher._detect_faces(path)
            if faces_data:
                searcher._update_photo_faces(photo_id, faces_data)
                for idx, face in enumerate(faces_data):
                    try:
                        face_id = hashlib.md5(f"{photo_id}:{idx}:{face.get('bbox')}".encode()).hexdigest()[:16]
                        bbox = face.get('bbox')
                        bbox_str = json.dumps(bbox) if bbox is not None else json.dumps([])
                        embedding = None
                        enc = face.get('encoding')
                        if isinstance(enc, list):
                            embedding = np.array(enc, dtype=np.float32)
                        elif isinstance(enc, np.ndarray):
                            embedding = enc.astype(np.float32)
                        method = face.get('method')
                        db.insert_face(face_id, photo_id, bbox_str, embedding, method)
                    except Exception as e:
                        print(f"   ⚠️ Failed to persist face for {os.path.basename(path)}: {e}")
                processed += 1
            else:
                skipped += 1
            if i % 20 == 0 or i == total:
                print(f"   Progress: {i}/{total} (processed {processed}, skipped {skipped})")
        except Exception as e:
            print(f"   ❌ Error on {os.path.basename(path)}: {e}")
    print(f"✅ Backfill complete. Processed {processed}, skipped {skipped}")

def assign_new_faces(searcher: UltimatePhotoSearcher, threshold: float = 0.45, only_photo_path: Optional[str] = None):
    """Assign only unclustered faces to nearest existing clusters; keep current clusters unchanged."""
    db = searcher.db
    unclustered = db.get_unclustered_faces_with_embeddings()
    if only_photo_path:
        # Map path -> photo_id
        pid_lookup = {pid: path for pid, path, _ in db.get_all_embeddings()}
        target_pids = [pid for pid, p in pid_lookup.items() if os.path.abspath(p) == os.path.abspath(only_photo_path)]
        if not target_pids:
            print(f"❌ Photo not found in DB: {only_photo_path}")
            return
        unclustered = [(fid, pid, emb) for fid, pid, emb in unclustered if pid in set(target_pids)]
    if not unclustered:
        print("ℹ️ No unclustered faces found")
        return
    centroids = db.get_cluster_centroids()
    if not centroids:
        print("ℹ️ No existing clusters to assign to. Run --cluster-faces first.")
        return
    import numpy as _np
    # Prepare centroid matrix
    cid_list = [cid for cid, _, _ in centroids]
    C = _np.stack([c for _, c, _ in centroids]).astype(_np.float32)
    assigned = 0
    for fid, pid, emb in unclustered:
        e = emb / (_np.linalg.norm(emb) + 1e-8)
        sims = (C @ e)
        j = int(sims.argmax())
        max_sim = float(sims[j])
        print(f"   Face {fid}: max similarity = {max_sim:.3f} to cluster {cid_list[j]}")
        if max_sim >= threshold:
            db.assign_cluster_to_faces(cid_list[j], [fid])
            assigned += 1
            print(f"   ✅ Assigned to cluster {cid_list[j]}")
        else:
            print(f"   ❌ Below threshold {threshold}")
    print(f"✅ Assigned {assigned}/{len(unclustered)} new faces to existing clusters (threshold={threshold})")

if __name__ == "__main__":
    print("🚀 Ultimate On-Device Photo Search System")
    print("=" * 50)
    main()
