"""
Phase 8: Add Face Embeddings
On-Device Photo Search Prototype - Face Recognition Extension
"""

import numpy as np
import cv2
from PIL import Image
import os
from typing import List, Dict, Tuple, Optional
import sqlite3
import pickle

# Import our previous implementations
from clip_model import CLIPEmbeddingExtractor
from yolo_enhancement import EnhancedPhotoDatabase

try:
    # Try to import face_recognition library
    import face_recognition  # type: ignore
    FACE_RECOGNITION_AVAILABLE = True
    print("✅ face_recognition library available")
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    face_recognition = None  # type: ignore
    print("⚠️ face_recognition library not available. Using OpenCV for face detection only.")

class FaceEmbeddingExtractor:
    """Face detection and embedding extraction using face_recognition or OpenCV"""
    
    def __init__(self, use_face_recognition: bool = True):
        """
        Initialize face embedding extractor
        
        Args:
            use_face_recognition: Whether to use face_recognition library (if available)
        """
        print("👤 Initializing Face Embedding Extractor...")
        
        self.use_face_recognition = use_face_recognition and FACE_RECOGNITION_AVAILABLE
        
        if not self.use_face_recognition:
            # Initialize OpenCV face detector as fallback
            try:
                # Try different paths for the cascade file
                import cv2
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'  # type: ignore
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                if self.face_cascade.empty():
                    # Try alternative path
                    self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
                print("✅ Using OpenCV face detection (basic detection only)")
            except Exception as e:
                print(f"⚠️ Could not load OpenCV face cascade: {e}")
                self.face_cascade = None
        else:
            print("✅ Using face_recognition library (full embeddings)")
    
    def detect_faces_opencv(self, image_path: str) -> List[Dict]:
        """
        Detect faces using OpenCV (basic detection, no embeddings)
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of face detection results
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return []
            
            # Check if cascade is loaded
            if self.face_cascade is None or self.face_cascade.empty():
                print("⚠️ Face cascade not loaded, cannot detect faces")
                return []
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            face_data = []
            for i, (x, y, w, h) in enumerate(faces):
                face_data.append({
                    'face_id': f"face_{i}",
                    'bbox': [int(x), int(y), int(w), int(h)],
                    'confidence': 1.0,  # OpenCV doesn't provide confidence
                    'embedding': None  # No embedding with OpenCV
                })
            
            return face_data
            
        except Exception as e:
            print(f"❌ Error detecting faces with OpenCV: {e}")
            return []
    
    def detect_faces_face_recognition(self, image_path: str) -> List[Dict]:
        """
        Detect faces and extract embeddings using face_recognition library
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of face detection results with embeddings
        """
        try:
            # Check if face_recognition is available
            if not FACE_RECOGNITION_AVAILABLE or face_recognition is None:
                return []
            
            # Load image
            image = face_recognition.load_image_file(image_path)  # type: ignore
            
            # Find face locations
            face_locations = face_recognition.face_locations(image, model="hog")  # type: ignore
            
            if not face_locations:
                return []
            
            # Get face encodings (embeddings)
            face_encodings = face_recognition.face_encodings(image, face_locations)  # type: ignore
            
            face_data = []
            for i, (top, right, bottom, left) in enumerate(face_locations):
                # Get embedding if available
                embedding = face_encodings[i] if i < len(face_encodings) else None
                
                face_data.append({
                    'face_id': f"face_{i}",
                    'bbox': [left, top, right - left, bottom - top],  # Convert to x, y, w, h
                    'confidence': 1.0,  # face_recognition doesn't provide confidence scores
                    'embedding': embedding
                })
            
            return face_data
            
        except Exception as e:
            print(f"❌ Error detecting faces with face_recognition: {e}")
            return []
    
    def get_face_embeddings(self, image_path: str) -> List[Dict]:
        """
        Extract face embeddings from an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of face embeddings and metadata
        """
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return []
        
        print(f"👤 Detecting faces in: {os.path.basename(image_path)}")
        
        # Use appropriate face detection method
        if self.use_face_recognition:
            faces = self.detect_faces_face_recognition(image_path)
        else:
            faces = self.detect_faces_opencv(image_path)
        
        print(f"👤 Detected {len(faces)} faces")
        
        for i, face in enumerate(faces):
            bbox = face['bbox']
            print(f"   Face {i+1}: bbox=({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]})")
            if face['embedding'] is not None:
                print(f"   Embedding shape: {face['embedding'].shape}")
        
        return faces
    
    def compare_faces(self, known_embedding: np.ndarray, unknown_embedding: np.ndarray, 
                     tolerance: float = 0.6) -> Tuple[bool, float]:
        """
        Compare two face embeddings
        
        Args:
            known_embedding: Reference face embedding
            unknown_embedding: Face embedding to compare
            tolerance: Similarity tolerance (lower = more strict)
            
        Returns:
            Tuple of (is_match, distance)
        """
        if (self.use_face_recognition and FACE_RECOGNITION_AVAILABLE and 
            face_recognition is not None and 
            known_embedding is not None and unknown_embedding is not None):
            # Use face_recognition's comparison
            distance = face_recognition.face_distance([known_embedding], unknown_embedding)[0]  # type: ignore
            is_match = distance <= tolerance
            return is_match, float(distance)
        else:
            # Basic comparison not available without embeddings
            return False, 1.0


class FaceEnhancedPhotoDatabase(EnhancedPhotoDatabase):
    """Database enhanced with face embeddings support"""
    
    def __init__(self, db_path: str = "photos.db"):
        """Initialize face-enhanced database"""
        super().__init__(db_path)
        self.add_faces_column()
    
    def add_faces_column(self):
        """Add faces column to existing photos table"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if faces column exists
            cursor.execute("PRAGMA table_info(photos)")
            columns = [column[1] for column in cursor.fetchall()]
            
            if 'faces' not in columns:
                # Add faces column for storing face data
                cursor.execute('ALTER TABLE photos ADD COLUMN faces TEXT')
                print("✅ Added 'faces' column to database")
            else:
                print("ℹ️ 'faces' column already exists")
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"❌ Error adding faces column: {e}")
    
    def serialize_faces_data(self, faces_data: List[Dict]) -> str:
        """
        Serialize faces data to string for database storage
        
        Args:
            faces_data: List of face detection results
            
        Returns:
            Serialized string
        """
        if not faces_data:
            return ""
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_faces = []
        for face in faces_data:
            serializable_face = face.copy()
            if face.get('embedding') is not None:
                # Convert numpy array to base64 string
                embedding_bytes = pickle.dumps(face['embedding'])
                serializable_face['embedding'] = embedding_bytes.hex()
            else:
                serializable_face['embedding'] = None
            serializable_faces.append(serializable_face)
        
        # Convert to string representation
        import json
        return json.dumps(serializable_faces)
    
    def deserialize_faces_data(self, faces_str: str) -> List[Dict]:
        """
        Deserialize faces data from database string
        
        Args:
            faces_str: Serialized faces string
            
        Returns:
            List of face detection results
        """
        if not faces_str:
            return []
        
        try:
            import json
            faces_data = json.loads(faces_str)
            
            # Convert embedding strings back to numpy arrays
            for face in faces_data:
                if face.get('embedding'):
                    embedding_bytes = bytes.fromhex(face['embedding'])
                    face['embedding'] = pickle.loads(embedding_bytes)
            
            return faces_data
            
        except Exception as e:
            print(f"❌ Error deserializing faces data: {e}")
            return []
    
    def update_photo_faces(self, photo_id: str, faces_data: List[Dict]) -> bool:
        """
        Update face embeddings for a photo
        
        Args:
            photo_id: Unique identifier for the photo
            faces_data: List of face detection results
            
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Serialize faces data
            faces_str = self.serialize_faces_data(faces_data)
            
            cursor.execute('UPDATE photos SET faces = ? WHERE id = ?', (faces_str, photo_id))
            rows_affected = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            if rows_affected > 0:
                face_count = len(faces_data)
                print(f"👤 Updated faces for photo {photo_id}: {face_count} faces")
                return True
            else:
                print(f"⚠️ Photo not found: {photo_id}")
                return False
                
        except Exception as e:
            print(f"❌ Error updating faces for {photo_id}: {e}")
            return False
    
    def search_photos_with_faces(self) -> List[Tuple[str, str, List[Dict]]]:
        """
        Search for photos that contain faces
        
        Returns:
            List of tuples: (photo_id, path, faces_data)
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, path, faces FROM photos 
                WHERE faces IS NOT NULL AND faces != ""
            ''')
            
            rows = cursor.fetchall()
            conn.close()
            
            results = []
            for photo_id, path, faces_str in rows:
                faces_data = self.deserialize_faces_data(faces_str)
                if faces_data:  # Only include photos with actual face data
                    results.append((photo_id, path, faces_data))
            
            print(f"👤 Found {len(results)} photos with faces")
            return results
            
        except Exception as e:
            print(f"❌ Error searching photos with faces: {e}")
            return []


class FaceEnhancedPhotoIndexer:
    """Photo indexer with face recognition capabilities"""
    
    def __init__(self, db_path: str = "photos.db"):
        """
        Initialize face-enhanced photo indexer
        
        Args:
            db_path: Path to the SQLite database
        """
        print("🚀 Initializing Face-Enhanced Photo Indexer...")
        
        # Initialize components
        self.clip_extractor = CLIPEmbeddingExtractor()
        self.face_extractor = FaceEmbeddingExtractor()
        self.database = FaceEnhancedPhotoDatabase(db_path)
        
        print("✅ Face-Enhanced Photo Indexer ready!")
    
    def reprocess_photos_for_faces(self) -> dict:
        """
        Add face embeddings to existing photos
        
        Returns:
            Dictionary with processing results
        """
        print("👤 Reprocessing existing photos for face detection...")
        
        try:
            # Get all photos from database
            photo_data = self.database.get_all_embeddings()
            
            if not photo_data:
                print("❌ No photos found in database!")
                return {'total': 0, 'successful': 0, 'failed': 0, 'faces_found': 0}
            
            successful = 0
            failed = 0
            total_faces = 0
            
            for photo_id, path, _ in photo_data:
                try:
                    print(f"\n👤 Processing: {os.path.basename(path)}")
                    
                    # Check if file still exists
                    if not os.path.exists(path):
                        print(f"⚠️ File not found, skipping: {path}")
                        failed += 1
                        continue
                    
                    # Detect faces
                    faces_data = self.face_extractor.get_face_embeddings(path)
                    total_faces += len(faces_data)
                    
                    # Update database
                    if self.database.update_photo_faces(photo_id, faces_data):
                        successful += 1
                    else:
                        failed += 1
                        
                except Exception as e:
                    print(f"❌ Error processing {path}: {e}")
                    failed += 1
            
            results = {
                'total': len(photo_data),
                'successful': successful,
                'failed': failed,
                'faces_found': total_faces
            }
            
            print(f"\n📈 Face Detection Summary:")
            print(f"   Total photos: {results['total']}")
            print(f"   Successfully processed: {results['successful']}")
            print(f"   Failed: {results['failed']}")
            print(f"   Total faces found: {results['faces_found']}")
            
            return results
            
        except Exception as e:
            print(f"❌ Error reprocessing photos for faces: {e}")
            return {'total': 0, 'successful': 0, 'failed': 0, 'faces_found': 0}


def install_face_recognition():
    """Install face_recognition library"""
    print("📦 Installing face_recognition library...")
    
    try:
        import subprocess
        import sys
        
        # Install face_recognition
        subprocess.check_call([sys.executable, "-m", "pip", "install", "face_recognition"])
        print("✅ face_recognition installed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to install face_recognition: {e}")
        print("ℹ️ You can install it manually with: pip install face_recognition")
        print("ℹ️ Continuing with OpenCV face detection only...")
        return False


def test_face_enhancement():
    """Test the face recognition enhancement"""
    print("\n" + "="*60)
    print("🧪 Testing Face Recognition Enhancement")
    print("="*60)
    
    try:
        # Check if face_recognition is available, offer to install if not
        if not FACE_RECOGNITION_AVAILABLE:
            print("⚠️ face_recognition library not found")
            print("ℹ️ This library provides advanced face embeddings")
            print("ℹ️ Without it, we'll use basic OpenCV face detection")
            
            # For now, continue with OpenCV
            print("📝 Continuing with OpenCV face detection...")
        
        # Initialize face-enhanced indexer
        face_indexer = FaceEnhancedPhotoIndexer("photos.db")
        
        # Test face detection on sample photos
        sample_folder = "./sample_photos"
        
        if os.path.exists(sample_folder):
            image_files = [f for f in os.listdir(sample_folder) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if image_files:
                print(f"\n👤 Testing face detection on {len(image_files)} sample images...")
                
                # Test detection on each image
                for image_file in image_files[:3]:  # Test first 3 images
                    image_path = os.path.join(sample_folder, image_file)
                    print(f"\n{'='*40}")
                    faces = face_indexer.face_extractor.get_face_embeddings(image_path)
                    print(f"📸 {image_file}: {len(faces)} faces detected")
        
        # Reprocess existing photos to add face data
        print(f"\n👤 Adding face detection to existing photos...")
        results = face_indexer.reprocess_photos_for_faces()
        
        # Test face search
        if results['faces_found'] > 0:
            print(f"\n🔍 Testing face-based search...")
            photos_with_faces = face_indexer.database.search_photos_with_faces()
            
            print(f"👤 Photos containing faces:")
            for photo_id, path, faces_data in photos_with_faces:
                face_count = len(faces_data)
                print(f"   {os.path.basename(path)}: {face_count} face(s)")
        
        print(f"\n✅ Face recognition enhancement test completed!")
        print(f"🎉 Phase 8 complete - ready for Phase 9 (Final Integration)!")
        
        return face_indexer
        
    except Exception as e:
        print(f"❌ Face enhancement test failed: {e}")
        raise


if __name__ == "__main__":
    # Test the face recognition enhancement
    test_face = test_face_enhancement()
