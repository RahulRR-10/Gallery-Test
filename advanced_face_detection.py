"""
Advanced Face Detection & Recognition Module
Upgraded face detection using state-of-the-art models
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import os
import pickle
from PIL import Image

# Try importing advanced face detection libraries
try:
    import insightface
    HAS_INSIGHTFACE = True
except ImportError:
    HAS_INSIGHTFACE = False
    print("⚠️ InsightFace not available - using fallback")

try:
    from mtcnn import MTCNN
    HAS_MTCNN = True
except ImportError:
    HAS_MTCNN = False
    print("⚠️ MTCNN not available - using fallback")

try:
    from facenet_pytorch import MTCNN as FaceNetMTCNN, InceptionResnetV1
    HAS_FACENET = True
except ImportError:
    HAS_FACENET = False
    print("⚠️ FaceNet not available - using fallback")

# Fallback to basic face detection
try:
    import face_recognition
    HAS_FACE_RECOGNITION = True
except ImportError:
    HAS_FACE_RECOGNITION = False


class AdvancedFaceDetector:
    """State-of-the-art face detection and recognition"""
    
    def __init__(self, detection_method: str = "auto"):
        """
        Initialize advanced face detector
        
        Args:
            detection_method: "insightface", "mtcnn", "facenet", "basic", or "auto"
        """
        self.detection_method = detection_method
        self.detector = None
        self.recognizer = None
        
        print(f"🎯 Initializing advanced face detection...")
        
        # Auto-select best available method
        if detection_method == "auto":
            if HAS_INSIGHTFACE:
                detection_method = "insightface"
            elif HAS_MTCNN:
                detection_method = "mtcnn"
            elif HAS_FACENET:
                detection_method = "facenet"
            elif HAS_FACE_RECOGNITION:
                detection_method = "basic"
            else:
                detection_method = "opencv"
        
        self.detection_method = detection_method
        self._initialize_detector()
    
    def _initialize_detector(self):
        """Initialize the selected face detection method"""
        try:
            if self.detection_method == "insightface":
                self._init_insightface()
            elif self.detection_method == "mtcnn":
                self._init_mtcnn()
            elif self.detection_method == "facenet":
                self._init_facenet()
            elif self.detection_method == "basic":
                self._init_basic()
            else:
                self._init_opencv()
                
            print(f"✅ Advanced face detection ready: {self.detection_method}")
            
        except Exception as e:
            print(f"⚠️ Error initializing {self.detection_method}: {e}")
            print("🔄 Falling back to OpenCV...")
            self._init_opencv()
    
    def _init_insightface(self):
        """Initialize InsightFace (best accuracy)"""
        if not HAS_INSIGHTFACE:
            raise ImportError("InsightFace not available")
            
        # Initialize InsightFace app
        self.detector = insightface.app.FaceAnalysis(providers=['CPUExecutionProvider'])
        self.detector.prepare(ctx_id=0, det_size=(640, 640))
        print("🚀 InsightFace initialized - SOTA face detection!")
    
    def _init_mtcnn(self):
        """Initialize MTCNN (good balance)"""
        if not HAS_MTCNN:
            raise ImportError("MTCNN not available")
            
        self.detector = MTCNN(min_face_size=20, scale_factor=0.709, steps_threshold=[0.6, 0.7, 0.7])
        print("🎯 MTCNN initialized - high accuracy detection!")
    
    def _init_facenet(self):
        """Initialize FaceNet (PyTorch)"""
        if not HAS_FACENET:
            raise ImportError("FaceNet not available")
            
        self.detector = FaceNetMTCNN(image_size=160, margin=0, min_face_size=20)
        self.recognizer = InceptionResnetV1(pretrained='vggface2').eval()
        print("🧠 FaceNet initialized - PyTorch face recognition!")
    
    def _init_basic(self):
        """Initialize basic face_recognition library"""
        if not HAS_FACE_RECOGNITION:
            raise ImportError("face_recognition not available")
            
        print("📸 Basic face_recognition initialized")
    
    def _init_opencv(self):
        """Initialize OpenCV Haar cascades (fallback)"""
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.detector = cv2.CascadeClassifier(cascade_path)
        print("🔧 OpenCV Haar cascades initialized (fallback)")
    
    def detect_faces(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Detect faces in an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of face dictionaries with bbox, confidence, and embedding
        """
        try:
            if self.detection_method == "insightface":
                return self._detect_insightface(image_path)
            elif self.detection_method == "mtcnn":
                return self._detect_mtcnn(image_path)
            elif self.detection_method == "facenet":
                return self._detect_facenet(image_path)
            elif self.detection_method == "basic":
                return self._detect_basic(image_path)
            else:
                return self._detect_opencv(image_path)
                
        except Exception as e:
            print(f"⚠️ Face detection error for {image_path}: {e}")
            return []
    
    def _detect_insightface(self, image_path: str) -> List[Dict[str, Any]]:
        """Detect faces using InsightFace"""
        img = cv2.imread(image_path)
        if img is None:
            return []
            
        faces = self.detector.get(img)
        
        results = []
        for face in faces:
            bbox = face.bbox.astype(int)
            results.append({
                'bbox': bbox.tolist(),  # [x1, y1, x2, y2]
                'confidence': float(face.det_score),
                'embedding': face.embedding.tolist(),
                'landmarks': face.kps.tolist() if hasattr(face, 'kps') else None,
                'age': float(face.age) if hasattr(face, 'age') else None,
                'gender': face.sex if hasattr(face, 'sex') else None
            })
        
        return results
    
    def _detect_mtcnn(self, image_path: str) -> List[Dict[str, Any]]:
        """Detect faces using MTCNN"""
        img = cv2.imread(image_path)
        if img is None:
            return []
            
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        faces = self.detector.detect_faces(img_rgb)
        
        results = []
        for face in faces:
            bbox = face['box']  # [x, y, width, height]
            # Convert to [x1, y1, x2, y2]
            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h
            
            results.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': float(face['confidence']),
                'embedding': None,  # MTCNN doesn't provide embeddings directly
                'landmarks': face.get('keypoints', None)
            })
        
        return results
    
    def _detect_facenet(self, image_path: str) -> List[Dict[str, Any]]:
        """Detect faces using FaceNet"""
        img = Image.open(image_path)
        
        # Detect faces
        boxes, probs = self.detector.detect(img)
        
        results = []
        if boxes is not None:
            for box, prob in zip(boxes, probs):
                if prob > 0.5:  # Confidence threshold
                    results.append({
                        'bbox': box.tolist(),
                        'confidence': float(prob),
                        'embedding': None,  # Can extract embeddings separately
                        'landmarks': None
                    })
        
        return results
    
    def _detect_basic(self, image_path: str) -> List[Dict[str, Any]]:
        """Detect faces using face_recognition library"""
        img = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(img, model="hog")
        face_encodings = face_recognition.face_encodings(img, face_locations)
        
        results = []
        for location, encoding in zip(face_locations, face_encodings):
            # face_recognition returns (top, right, bottom, left)
            top, right, bottom, left = location
            
            results.append({
                'bbox': [left, top, right, bottom],  # Convert to [x1, y1, x2, y2]
                'confidence': 0.9,  # face_recognition doesn't provide confidence
                'embedding': encoding.tolist(),
                'landmarks': None
            })
        
        return results
    
    def _detect_opencv(self, image_path: str) -> List[Dict[str, Any]]:
        """Detect faces using OpenCV (fallback)"""
        img = cv2.imread(image_path)
        if img is None:
            return []
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        results = []
        for (x, y, w, h) in faces:
            results.append({
                'bbox': [x, y, x + w, y + h],
                'confidence': 0.8,  # OpenCV doesn't provide confidence
                'embedding': None,
                'landmarks': None
            })
        
        return results
    
    def get_face_summary(self, faces: List[Dict[str, Any]]) -> str:
        """Generate a summary of detected faces"""
        if not faces:
            return "No faces detected"
        
        summary = f"{len(faces)} face{'s' if len(faces) > 1 else ''} detected"
        
        if self.detection_method == "insightface":
            # Add age/gender info if available
            ages = [f['age'] for f in faces if f.get('age')]
            genders = [f['gender'] for f in faces if f.get('gender')]
            
            if ages:
                avg_age = sum(ages) / len(ages)
                summary += f" (avg age: {avg_age:.0f})"
            
            if genders:
                gender_count = {}
                for g in genders:
                    gender_count[g] = gender_count.get(g, 0) + 1
                summary += f" {gender_count}"
        
        return summary


def test_advanced_face_detection():
    """Test the advanced face detection system"""
    print("🧪 Testing Advanced Face Detection System")
    print("=" * 50)
    
    detector = AdvancedFaceDetector()
    
    # Test on a sample image
    sample_images = [
        "sample_photos/IMG_20250420_175508232.jpg",
        "sample_photos/IMG-20250525-WA0000.jpg",
        "sample_photos/IMG_20250810_214151276.jpg"
    ]
    
    for img_path in sample_images:
        if os.path.exists(img_path):
            print(f"\n🔍 Testing: {os.path.basename(img_path)}")
            faces = detector.detect_faces(img_path)
            summary = detector.get_face_summary(faces)
            print(f"   Result: {summary}")
            
            if faces:
                for i, face in enumerate(faces):
                    bbox = face['bbox']
                    conf = face['confidence']
                    print(f"   Face {i+1}: bbox={bbox}, confidence={conf:.3f}")
        else:
            print(f"⚠️ Image not found: {img_path}")
    
    print("\n✅ Advanced face detection test completed!")


if __name__ == "__main__":
    test_advanced_face_detection()
