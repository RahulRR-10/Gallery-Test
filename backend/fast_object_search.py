"""
Fast Object Search - Direct database search for object tags
"""

import sqlite3
from typing import List, Dict, Optional
from photo_database import PhotoDatabase

class FastObjectSearch:
    """Fast object search using direct database queries"""
    
    def __init__(self, db_path: str = "photos.db"):
        self.db_path = db_path
    
    def search_by_object(self, object_name: str, limit: int = 100) -> List[Dict]:
        """
        Fast search for photos containing a specific object
        
        Args:
            object_name: Name of object to search for (e.g., "dog", "car", "person")
            limit: Maximum number of results
            
        Returns:
            List of photo dictionaries
        """
        try:
            db = PhotoDatabase(self.db_path)
            results = db.search_photos_by_objects(object_name, limit)
            
            # Convert to the expected format
            formatted_results = []
            for result in results:
                formatted_result = {
                    'id': result['id'],
                    'path': result['path'],
                    'similarity': result['similarity'],
                    'timestamp': result['timestamp'],
                    'objects': result.get('objects', ''),
                    'faces': result.get('faces', ''),
                    'exif_date': result.get('exif_date'),
                    'created_date': result.get('created_date')
                }
                formatted_results.append(formatted_result)
            
            return formatted_results
            
        except Exception as e:
            print(f"❌ Error in fast object search: {e}")
            return []

    def get_all_detected_objects(self) -> List[str]:
        """
        Get list of all detected objects across all photos
        
        Returns:
            List of unique object names
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT DISTINCT objects FROM photos WHERE objects IS NOT NULL AND objects != ""')
            rows = cursor.fetchall()
            conn.close()
            
            # Parse and collect all unique object names
            all_objects = set()
            for (objects_str,) in rows:
                if objects_str and objects_str != 'None':
                    # Split comma-separated objects
                    objects = [obj.strip() for obj in objects_str.split(',') if obj.strip()]
                    all_objects.update(objects)
            
            return sorted(list(all_objects))
            
        except Exception as e:
            print(f"❌ Error getting detected objects: {e}")
            return []

if __name__ == "__main__":
    # Test the fast object search
    searcher = FastObjectSearch()
    
    print("🔍 Testing Fast Object Search")
    print("=" * 50)
    
    # Test object search
    test_objects = ["dog", "person", "car"]
    
    for obj in test_objects:
        print(f"\n🎯 Searching for '{obj}':")
        results = searcher.search_by_object(obj, limit=5)
        if results:
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result['path']} (similarity: {result['similarity']:.3f})")
                if result.get('objects'):
                    print(f"     Objects: {result['objects']}")
        else:
            print(f"  No results found for '{obj}'")
    
    # Test getting all objects
    print(f"\n📊 All detected objects:")
    all_objects = searcher.get_all_detected_objects()
    print(f"Found {len(all_objects)} unique objects:")
    for obj in all_objects[:20]:  # Show first 20
        print(f"  - {obj}")
    if len(all_objects) > 20:
        print(f"  ... and {len(all_objects) - 20} more")