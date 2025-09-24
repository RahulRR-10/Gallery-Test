#!/usr/bin/env python3
"""
Lightweight Person Search - No Model Loading Required
===================================================
Fast person searches using only database queries for labeled clusters
"""

import sqlite3
import os
import json
import logging
from typing import List, Dict, Optional
from photo_database import PhotoDatabase

logger = logging.getLogger(__name__)

class LightweightPersonSearch:
    """Lightweight person search using only database queries"""
    
    def __init__(self, db_path: str = "photos.db"):
        self.db_path = db_path
        self.db = PhotoDatabase(db_path)
    
    def search_person_photos(self, person_label: str, limit: int = 10, time_filter: Optional[str] = None) -> List[Dict]:
        """
        Search for photos of a specific person using only database queries
        
        Args:
            person_label: Name of the person (cluster label)
            limit: Maximum number of results
            time_filter: Optional time filter string
            
        Returns:
            List of photo results with basic metadata
        """
        try:
            # Get cluster for this person
            cluster = self.db.get_cluster_by_label(person_label)
            if not cluster:
                logger.warning(f"No cluster found for person: {person_label}")
                return []
            
            cluster_id = cluster['cluster_id']
            
            # Get photos containing this cluster
            photo_ids = self.db.get_photos_with_clusters([cluster_id])
            if not photo_ids:
                logger.info(f"No photos found for person: {person_label}")
                return []
            
            logger.info(f"Found {len(photo_ids)} photos with {person_label} (cluster: {cluster_id})")
            
            # Apply time filtering if requested
            if time_filter:
                filtered_photos = self._apply_time_filter(photo_ids, time_filter)
                if not filtered_photos:
                    logger.info(f"No photos for '{person_label}' within time range: {time_filter}")
                    return []
                photo_ids = filtered_photos
            
            # Get photo metadata
            results = []
            all_embeddings = self.db.get_all_embeddings()  # Get path lookup
            path_lookup = {pid: path for pid, path, _ in all_embeddings}
            
            # Limit results
            photo_ids = photo_ids[:limit]
            
            for photo_id in photo_ids:
                path = path_lookup.get(photo_id, '')
                if not path or not os.path.exists(path):
                    continue
                
                # Get faces for this photo from this cluster
                target_faces = self._get_cluster_faces_for_photo(photo_id, cluster_id)
                
                result = {
                    'path': path,
                    'similarity': 1.0,  # Perfect match for labeled cluster
                    'target_faces': target_faces,
                    'person_label': person_label
                }
                results.append(result)
                
            logger.info(f"Returning {len(results)} photos for {person_label}")
            return results
            
        except Exception as e:
            logger.error(f"Error in lightweight person search for '{person_label}': {e}")
            return []
    
    def _apply_time_filter(self, photo_ids: List[str], time_filter: str) -> List[str]:
        """Apply time filtering to photo IDs"""
        try:
            # Import temporal parser only if needed
            from temporal_search import TemporalParser
            temporal_parser = TemporalParser()
            
            start_ts, end_ts = temporal_parser.parse_time_expression(time_filter)
            
            # Get time-filtered photos
            time_filtered = self.db.search_photos_by_time(start_ts, end_ts, use_exif=True)
            if not time_filtered:
                time_filtered = self.db.search_photos_by_time(start_ts, end_ts, use_exif=False)
            
            if not time_filtered:
                return []
            
            # Filter photo_ids to only include those in time range
            allowed_ids = {pid for pid, _, _, _ in time_filtered}
            filtered_ids = [pid for pid in photo_ids if pid in allowed_ids]
            
            return filtered_ids
            
        except Exception as e:
            logger.error(f"Error applying time filter '{time_filter}': {e}")
            return photo_ids  # Return original list if time filtering fails
    
    def _get_cluster_faces_for_photo(self, photo_id: str, cluster_id: str) -> List[Dict]:
        """Get face bounding boxes for a specific cluster in a photo"""
        try:
            faces = self.db.get_faces_by_photo(photo_id)
            cluster_faces = [f for f in faces if f.get('cluster_id') == cluster_id]
            
            target_faces = []
            for face in cluster_faces:
                try:
                    bbox = face.get('bbox')
                    if isinstance(bbox, str):
                        bbox = json.loads(bbox)
                    
                    target_faces.append({
                        'bbox': bbox,
                        'person_label': self._get_cluster_label(cluster_id)
                    })
                except Exception:
                    continue
            
            return target_faces
            
        except Exception as e:
            logger.error(f"Error getting cluster faces for photo {photo_id}: {e}")
            return []
    
    def _get_cluster_label(self, cluster_id: str) -> str:
        """Get label for a cluster ID"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT label FROM face_clusters WHERE cluster_id = ?", (cluster_id,))
            result = cursor.fetchone()
            conn.close()
            
            return result[0] if result else "Unknown"
            
        except Exception as e:
            logger.error(f"Error getting cluster label for {cluster_id}: {e}")
            return "Unknown"
    
    def get_all_person_labels(self) -> List[str]:
        """Get all available person labels"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT DISTINCT TRIM(label) 
                FROM face_clusters 
                WHERE label IS NOT NULL AND label != '' AND label != 'Unknown'
                ORDER BY label
            """)
            
            labels = [row[0] for row in cursor.fetchall() if row[0]]
            conn.close()
            
            return labels
            
        except Exception as e:
            logger.error(f"Error getting person labels: {e}")
            return []

def test_lightweight_search():
    """Test the lightweight person search"""
    search = LightweightPersonSearch()
    
    # Test with known person
    results = search.search_person_photos("Amisha", limit=5)
    print(f"Found {len(results)} photos for Amisha")
    for result in results[:3]:  # Show first 3
        print(f"  - {os.path.basename(result['path'])} ({len(result['target_faces'])} faces)")

if __name__ == "__main__":
    test_lightweight_search()