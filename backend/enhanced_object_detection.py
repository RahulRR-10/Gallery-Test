#!/usr/bin/env python3
"""
Enhanced Object Detection Pipeline
================================
Advanced object detection with multiple models and confidence-based filtering
"""

import numpy as np
from typing import List, Dict, Optional
from ultralytics import YOLO
import logging

logger = logging.getLogger(__name__)

class EnhancedObjectDetector:
    """Enhanced object detection with multiple models and smart filtering"""
    
    def __init__(self):
        """Initialize enhanced object detector"""
        self.models = {}
        self._load_models()
    
    def _load_models(self):
        """Load multiple YOLO models for different purposes"""
        try:
            # Primary general-purpose model
            self.models['general'] = YOLO('yolov8x.pt')
            logger.info("✅ Loaded YOLOv8x general model")
            
            # Optional: Segmentation model for better accuracy
            # self.models['segmentation'] = YOLO('yolov8x-seg.pt')
            
            # Optional: Specialized models
            # self.models['face'] = YOLO('yolov8n-face.pt')
            
        except Exception as e:
            logger.error(f"Failed to load YOLO models: {e}")
    
    def detect_objects_enhanced(self, image_path: str, 
                              confidence_threshold: float = 0.25,
                              max_objects: int = 15) -> List[Dict]:
        """
        Enhanced object detection with adaptive thresholding and filtering
        
        Args:
            image_path: Path to image file
            confidence_threshold: Minimum confidence threshold
            max_objects: Maximum number of objects to return
            
        Returns:
            List of detected objects with enhanced metadata
        """
        if 'general' not in self.models:
            return []
        
        try:
            results = self.models['general'](image_path, verbose=False)
            all_detections = []
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        confidence = float(box.conf.cpu().numpy()[0])
                        class_id = int(box.cls.cpu().numpy()[0])
                        class_name = self.models['general'].names[class_id]
                        bbox = box.xyxy.cpu().numpy()[0].tolist()
                        
                        # Calculate box area for importance scoring
                        x1, y1, x2, y2 = bbox
                        area = (x2 - x1) * (y2 - y1)
                        
                        detection = {
                            'class': class_name,
                            'confidence': confidence,
                            'bbox': bbox,
                            'area': area,
                            'importance_score': confidence * np.log(1 + area)  # Confidence + size
                        }
                        all_detections.append(detection)
            
            # Apply adaptive filtering
            filtered_objects = self._apply_adaptive_filtering(
                all_detections, confidence_threshold, max_objects
            )
            
            return filtered_objects
            
        except Exception as e:
            logger.error(f"Object detection failed for {image_path}: {e}")
            return []
    
    def _apply_adaptive_filtering(self, detections: List[Dict], 
                                min_confidence: float, 
                                max_objects: int) -> List[Dict]:
        """
        Apply adaptive filtering to improve detection quality
        """
        if not detections:
            return []
        
        # Step 1: Remove very low confidence detections
        filtered = [d for d in detections if d['confidence'] >= min_confidence]
        
        # Step 2: Remove duplicate/overlapping detections of same class
        filtered = self._remove_duplicate_detections(filtered)
        
        # Step 3: Prioritize important objects (people, faces, large objects)
        priority_classes = {'person', 'face', 'car', 'dog', 'cat', 'bicycle'}
        for detection in filtered:
            if detection['class'] in priority_classes:
                detection['importance_score'] *= 1.5
        
        # Step 4: Sort by importance and limit count
        filtered.sort(key=lambda x: x['importance_score'], reverse=True)
        filtered = filtered[:max_objects]
        
        # Step 5: Clean up metadata for storage
        for detection in filtered:
            detection.pop('area', None)
            detection.pop('importance_score', None)
            detection['confidence'] = round(detection['confidence'], 3)
        
        return filtered
    
    def _remove_duplicate_detections(self, detections: List[Dict], 
                                   iou_threshold: float = 0.5) -> List[Dict]:
        """Remove overlapping detections of the same class"""
        if len(detections) <= 1:
            return detections
        
        # Group by class
        class_groups = {}
        for detection in detections:
            class_name = detection['class']
            if class_name not in class_groups:
                class_groups[class_name] = []
            class_groups[class_name].append(detection)
        
        # Apply NMS per class
        filtered_detections = []
        for class_name, class_detections in class_groups.items():
            if len(class_detections) == 1:
                filtered_detections.extend(class_detections)
            else:
                # Simple NMS implementation
                class_detections.sort(key=lambda x: x['confidence'], reverse=True)
                keep = [class_detections[0]]  # Keep highest confidence
                
                for detection in class_detections[1:]:
                    should_keep = True
                    for kept_detection in keep:
                        if self._calculate_iou(detection['bbox'], kept_detection['bbox']) > iou_threshold:
                            should_keep = False
                            break
                    if should_keep:
                        keep.append(detection)
                
                filtered_detections.extend(keep)
        
        return filtered_detections
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate Intersection over Union (IoU) of two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

# Example usage integration with existing system
def integrate_enhanced_detection():
    """Example of how to integrate enhanced detection into existing system"""
    
    # In UltimatePhotoSearcher.__init__():
    # self.enhanced_detector = EnhancedObjectDetector()
    
    # Replace _detect_objects method:
    def _detect_objects_enhanced(self, image_path: str) -> List[Dict]:
        """Enhanced object detection method"""
        if hasattr(self, 'enhanced_detector'):
            return self.enhanced_detector.detect_objects_enhanced(
                image_path, 
                confidence_threshold=0.25,
                max_objects=12
            )
        else:
            # Fallback to original method
            return self._detect_objects_original(image_path)