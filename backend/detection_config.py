# Object Detection Configuration
OBJECT_DETECTION_CONFIG = {
    # Confidence thresholds for different object types
    "confidence_thresholds": {
        "person": 0.3,          # Lower threshold for people (important)
        "face": 0.25,           # Lower threshold for faces
        "car": 0.4,             # Medium threshold for vehicles
        "animal": 0.35,         # Lower threshold for pets/animals
        "food": 0.4,            # Medium threshold for food items
        "default": 0.3          # Default threshold for other objects
    },
    
    # Maximum objects per photo to avoid clutter
    "max_objects_per_photo": 12,
    
    # Priority classes (get boosted importance scores)
    "priority_classes": [
        "person", "face", "dog", "cat", "car", "bicycle", 
        "cake", "pizza", "bottle", "cup", "book", "laptop"
    ],
    
    # Object class mappings for better search
    "class_aliases": {
        "person": ["people", "human", "man", "woman", "child"],
        "car": ["vehicle", "automobile"],
        "dog": ["puppy", "canine"],
        "cat": ["kitten", "feline"],
        "bicycle": ["bike", "cycling"],
        "cake": ["birthday", "celebration"],
        "pizza": ["food", "dinner", "lunch"]
    },
    
    # Model settings
    "model_settings": {
        "primary_model": "yolov8x.pt",
        "backup_model": "yolov8l.pt",
        "enable_segmentation": False,
        "enable_pose_detection": False
    }
}