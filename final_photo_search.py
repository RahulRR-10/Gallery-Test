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

# Import our components
from clip_model import CLIPEmbeddingExtractor
from photo_database import PhotoDatabase

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

# For face detection
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
        
        # Initialize YOLO if available
        self.yolo_model = None
        if HAS_YOLO:
            try:
                print("🎯 Loading YOLO model...")
                self.yolo_model = YOLO('yolov8n.pt')
                print("✅ YOLO object detection ready!")
            except Exception as e:
                print(f"⚠️ YOLO loading failed: {e}")
                self.yolo_model = None
        
        # Initialize face detection
        self.face_cascade = None
        if HAS_OPENCV:
            try:
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'  # type: ignore
                if os.path.exists(cascade_path):
                    self.face_cascade = cv2.CascadeClassifier(cascade_path)
                    print("✅ Face detection ready!")
                else:
                    print("⚠️ Face cascade file not found")
            except Exception as e:
                print(f"⚠️ Face detection setup failed: {e}")
        
        print("✅ Ultimate Photo Search System ready!")
    
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
        import hashlib
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
        
        # Detect faces
        if self.face_cascade:
            faces_data = self._detect_faces(image_path)
            if faces_data:
                self._update_photo_faces(photo_id, faces_data)
        
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
        """Detect faces using OpenCV or face_recognition"""
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
                        'method': 'opencv'
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
    
    def search_photos(self, query: str, limit: int = 5, show_results: bool = True) -> List[Dict]:
        """
        Search photos using natural language queries
        """
        print(f"\n🔍 Searching for: '{query}'")
        
        # Get text embedding for query
        query_embedding = self.clip_extractor.get_clip_text_embedding(query)
        
        # Search database
        results = self._search_by_similarity(query_embedding, limit)
        
        if not results:
            print("❌ No matching photos found")
            return []
        
        print(f"✅ Found {len(results)} matching photos:")
        
        # Display results
        for i, result in enumerate(results, 1):
            print(f"\n{i}. 📸 {os.path.basename(result['path'])}")
            print(f"   🎯 Similarity: {result['similarity']:.3f}")
            
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
    
    def _search_by_similarity(self, query_embedding: np.ndarray, limit: int) -> List[Dict]:
        """Search photos by embedding similarity"""
        # Get all embeddings from database
        all_embeddings = self.db.get_all_embeddings()
        
        if not all_embeddings:
            return []
        
        # Calculate similarities
        similarities = []
        for photo_id, path, photo_embedding in all_embeddings:
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
                'faces': faces
            })
        
        # Sort by similarity
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similarities[:limit]
    
    def _display_results(self, results: List[Dict], query: str):
        """Display search results visually using matplotlib"""
        if not HAS_MATPLOTLIB:
            return
        
        n_results = min(len(results), 6)  # Max 6 images
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
                # Load and display image
                img = mpimg.imread(result['path'])
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
                                bbox = obj['bbox']
                                # bbox format: [x1, y1, x2, y2]
                                x1, y1, x2, y2 = bbox
                                width = x2 - x1
                                height = y2 - y1
                                rect = Rectangle((x1, y1), width, height, 
                                               linewidth=2, edgecolor='red', facecolor='none')
                                axes[i].add_patch(rect)
                                
                                # Add label
                                axes[i].text(x1, y1 - 5, f"{obj['class']}", 
                                           fontsize=8, color='red', weight='bold',
                                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
                    except:
                        pass
                
                # Add face detection boxes if available
                if result.get('faces'):
                    try:
                        faces = eval(result['faces']) if isinstance(result['faces'], str) else result['faces']
                        for face in faces:
                            if 'bbox' in face:
                                bbox = face['bbox']
                                # bbox format: [x, y, w, h]
                                x, y, w, h = bbox
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
  python final_photo_search.py --stats                    # Show database stats
        """
    )
    
    parser.add_argument('--index', type=str, metavar='DIRECTORY',
                       help='Index photos from the specified directory')
    parser.add_argument('--search', type=str, metavar='QUERY',
                       help='Search photos using natural language query')
    parser.add_argument('--stats', action='store_true',
                       help='Show database statistics')
    parser.add_argument('--limit', type=int, default=5, metavar='N',
                       help='Maximum number of search results (default: 5)')
    parser.add_argument('--db', type=str, default='photos.db', metavar='PATH',
                       help='Database file path (default: photos.db)')
    
    args = parser.parse_args()
    
    # Check if no arguments provided
    if not any([args.index, args.search, args.stats]):
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
        
        elif args.search:
            searcher.search_photos(args.search, limit=args.limit)
        
        elif args.stats:
            searcher.show_stats()
            
    except KeyboardInterrupt:
        print("\n⚠️ Operation cancelled by user")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🚀 Ultimate On-Device Photo Search System")
    print("=" * 50)
    main()
