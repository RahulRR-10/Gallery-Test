#!/usr/bin/env python3
"""
Automatic Photo Indexing Service
Watches the sample_photos folder for changes and automatically indexes new photos
"""

import os
import time
import threading
from pathlib import Path
from typing import Set, Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import logging
from photo_database import PhotoDatabase
from final_photo_search import UltimatePhotoSearcher

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PhotoIndexHandler(FileSystemEventHandler):
    """Handler for photo file system events"""
    
    def __init__(self, watch_folder: str = "sample_photos"):
        self.watch_folder = watch_folder
        self.db = PhotoDatabase("photos.db")
        self.photo_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        self.processing_queue: Set[str] = set()
        self.processing_lock = threading.Lock()
        
    def is_photo_file(self, filepath: str) -> bool:
        """Check if file is a photo based on extension"""
        return Path(filepath).suffix.lower() in self.photo_extensions
    
    def on_created(self, event):
        """Handle new file creation"""
        if not event.is_directory and self.is_photo_file(event.src_path):
            logger.info(f"New photo detected: {event.src_path}")
            self.queue_for_processing(event.src_path)
    
    def on_moved(self, event):
        """Handle file moves (includes renames)"""
        if not event.is_directory and self.is_photo_file(event.dest_path):
            logger.info(f"Photo moved/renamed: {event.src_path} -> {event.dest_path}")
            self.queue_for_processing(event.dest_path)
    
    def on_modified(self, event):
        """Handle file modifications"""
        if not event.is_directory and self.is_photo_file(event.src_path):
            # Only process if file is stable (not being written to)
            threading.Timer(2.0, lambda: self.queue_for_processing(event.src_path)).start()
    
    def queue_for_processing(self, filepath: str):
        """Add file to processing queue with debouncing"""
        with self.processing_lock:
            if filepath not in self.processing_queue:
                self.processing_queue.add(filepath)
                # Process after a short delay to allow file operations to complete
                threading.Timer(3.0, lambda: self.process_photo(filepath)).start()
    
    def process_photo(self, filepath: str):
        """Process a single photo file"""
        try:
            with self.processing_lock:
                if filepath in self.processing_queue:
                    self.processing_queue.remove(filepath)
                else:
                    return  # Already processed
            
            # Check if file still exists and is stable
            if not os.path.exists(filepath):
                logger.warning(f"File no longer exists: {filepath}")
                return
            
            # Wait for file to be stable (not being written to)
            file_size = -1
            for _ in range(3):
                current_size = os.path.getsize(filepath)
                if current_size == file_size:
                    break
                file_size = current_size
                time.sleep(1)
            
            # Check if photo is already indexed
            existing_photo = self.db.get_photo_by_path(filepath)
            if existing_photo:
                logger.info(f"Photo already indexed: {filepath}")
                return
            
            logger.info(f"Indexing new photo: {filepath}")
            
            # Initialize minimal searcher for photo indexing
            searcher = UltimatePhotoSearcher("photos.db")
            
            # Index the single photo
            success = searcher.index_single_photo(filepath)
            
            if success:
                logger.info(f"Successfully indexed: {filepath}")
                
                # Optionally run face detection on the new photo
                self.detect_faces_for_photo(filepath)
            else:
                logger.error(f"Failed to index: {filepath}")
                
        except Exception as e:
            logger.error(f"Error processing photo {filepath}: {e}")
    
    def detect_faces_for_photo(self, filepath: str):
        """Detect faces for a single photo (optional)"""
        try:
            from advanced_face_detection import AdvancedFaceDetector
            import sqlite3
            
            # Get photo ID
            photo = self.db.get_photo_by_path(filepath)
            if not photo:
                return
            
            photo_id = photo['id']
            
            # Check if faces already detected
            conn = sqlite3.connect("photos.db")
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM faces WHERE photo_id = ?', (photo_id,))
            existing_count = cursor.fetchone()[0]
            conn.close()
            
            if existing_count > 0:
                return  # Already processed
            
            logger.info(f"Detecting faces for: {filepath}")
            
            # Detect faces
            detector = AdvancedFaceDetector()
            faces = detector.detect_faces(filepath)
            
            # Save faces to database
            for i, face in enumerate(faces):
                if 'embedding' in face and face['embedding'] is not None:
                    face_id = f"{photo_id}_face_{i}"
                    bbox_str = f"{face['bbox'][0]},{face['bbox'][1]},{face['bbox'][2]},{face['bbox'][3]}"
                    embedding = face['embedding'] if isinstance(face['embedding'], list) else face['embedding'].tolist()
                    
                    import numpy as np
                    embedding_array = np.array(embedding)
                    
                    self.db.insert_face(
                        face_id,
                        photo_id,
                        bbox_str,
                        embedding_array,
                        "insightface"
                    )
            
            if faces:
                logger.info(f"Detected {len(faces)} faces in: {filepath}")
                
        except Exception as e:
            logger.error(f"Error detecting faces for {filepath}: {e}")

class AutoPhotoIndexer:
    """Main service for automatic photo indexing"""
    
    def __init__(self, watch_folder: str = "sample_photos"):
        self.watch_folder = watch_folder
        self.observer: Optional[Observer] = None
        self.handler = PhotoIndexHandler(watch_folder)
        self.is_running = False
        
    def start(self):
        """Start the file watcher service"""
        if self.is_running:
            logger.warning("Auto indexer already running")
            return
        
        try:
            # Ensure watch folder exists
            os.makedirs(self.watch_folder, exist_ok=True)
            
            # Setup file system observer
            self.observer = Observer()
            self.observer.schedule(self.handler, self.watch_folder, recursive=True)
            self.observer.start()
            
            self.is_running = True
            logger.info(f"Auto photo indexer started - watching: {self.watch_folder}")
            
        except Exception as e:
            logger.error(f"Failed to start auto indexer: {e}")
            raise
    
    def stop(self):
        """Stop the file watcher service"""
        if not self.is_running:
            return
        
        try:
            if self.observer:
                self.observer.stop()
                self.observer.join()
                self.observer = None
            
            self.is_running = False
            logger.info("Auto photo indexer stopped")
            
        except Exception as e:
            logger.error(f"Error stopping auto indexer: {e}")
    
    def get_status(self) -> dict:
        """Get current status of the auto indexer"""
        return {
            "running": self.is_running,
            "watch_folder": self.watch_folder,
            "queue_size": len(self.handler.processing_queue) if self.handler else 0
        }

# Global instance
_auto_indexer: Optional[AutoPhotoIndexer] = None

def get_auto_indexer() -> AutoPhotoIndexer:
    """Get or create the global auto indexer instance"""
    global _auto_indexer
    if _auto_indexer is None:
        _auto_indexer = AutoPhotoIndexer()
    return _auto_indexer

def start_auto_indexing():
    """Start automatic photo indexing"""
    indexer = get_auto_indexer()
    indexer.start()
    return indexer.get_status()

def stop_auto_indexing():
    """Stop automatic photo indexing"""
    indexer = get_auto_indexer()
    indexer.stop()
    return indexer.get_status()

def get_auto_indexing_status():
    """Get auto indexing status"""
    indexer = get_auto_indexer()
    return indexer.get_status()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Auto Photo Indexer")
    parser.add_argument("--start", action="store_true", help="Start the auto indexer")
    parser.add_argument("--stop", action="store_true", help="Stop the auto indexer")
    parser.add_argument("--status", action="store_true", help="Check status")
    parser.add_argument("--watch-folder", default="sample_photos", help="Folder to watch")
    
    args = parser.parse_args()
    
    if args.start:
        indexer = AutoPhotoIndexer(args.watch_folder)
        indexer.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            indexer.stop()
    elif args.stop:
        stop_auto_indexing()
    elif args.status:
        status = get_auto_indexing_status()
        print(f"Status: {status}")
    else:
        print("Use --start, --stop, or --status")