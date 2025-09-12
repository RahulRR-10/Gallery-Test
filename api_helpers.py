"""
API Helper Functions

This module provides helper functions to integrate FastAPI with the existing CLI system.
It maintains the principle of keeping all AI logic intact and editable.
"""

import os
import json
import sqlite3
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class APIHelpers:
    """Helper class for API integration with existing CLI functions"""
    
    def __init__(self, db_path: str = "photos.db"):
        self.db_path = db_path
    
    def get_db_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    def get_photo_by_id(self, photo_id: int) -> Optional[Dict[str, Any]]:
        """Get photo details by ID"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Get basic photo info
                cursor.execute("""
                    SELECT id, path, timestamp, objects 
                    FROM photos 
                    WHERE id = ?
                """, (photo_id,))
                
                result = cursor.fetchone()
                if not result:
                    return None
                
                photo = {
                    "id": result[0],
                    "path": result[1],
                    "timestamp": result[2],
                    "objects": json.loads(result[3]) if result[3] else []
                }
                
                # Get faces
                cursor.execute("""
                    SELECT cluster_id, bbox, confidence, age, gender
                    FROM faces 
                    WHERE photo_id = ?
                """, (photo_id,))
                
                faces = []
                for face_row in cursor.fetchall():
                    face = {
                        "cluster_id": face_row[0],
                        "bbox": json.loads(face_row[1]) if face_row[1] else [],
                        "confidence": face_row[2],
                        "age": face_row[3],
                        "gender": face_row[4]
                    }
                    faces.append(face)
                
                photo["faces"] = faces
                
                # Get relationships for people in this photo
                photo["relationships"] = self.get_photo_relationships(photo_id)
                
                return photo
                
        except Exception as e:
            logger.error(f"Error getting photo {photo_id}: {e}")
            return None
    
    def get_photo_relationships(self, photo_id: int) -> List[Dict[str, Any]]:
        """Get relationships for people in a photo"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Get cluster IDs for faces in this photo
                cursor.execute("""
                    SELECT DISTINCT cluster_id 
                    FROM faces 
                    WHERE photo_id = ? AND cluster_id IS NOT NULL
                """, (photo_id,))
                
                cluster_ids = [row[0] for row in cursor.fetchall()]
                
                if not cluster_ids:
                    return []
                
                # Get relationships
                relationships = []
                placeholders = ','.join(['?' for _ in cluster_ids])
                
                cursor.execute(f"""
                    SELECT ri.cluster_id_a, ri.cluster_id_b, ri.inferred_type, ri.confidence
                    FROM relationship_inferences ri
                    WHERE ri.cluster_id_a IN ({placeholders}) 
                       OR ri.cluster_id_b IN ({placeholders})
                """, cluster_ids + cluster_ids)
                
                for rel_row in cursor.fetchall():
                    relationship = {
                        "person1": rel_row[0],
                        "person2": rel_row[1],
                        "type": rel_row[2],
                        "confidence": rel_row[3]
                    }
                    relationships.append(relationship)
                
                return relationships
                
        except Exception as e:
            logger.error(f"Error getting relationships for photo {photo_id}: {e}")
            return []
    
    def get_face_clusters(self) -> List[Dict[str, Any]]:
        """Get all face clusters with sample photos"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Get clusters with labels
                cursor.execute("""
                    SELECT cluster_id, label 
                    FROM face_clusters 
                    ORDER BY cluster_id
                """)
                
                clusters = []
                for cluster_row in cursor.fetchall():
                    cluster_id = cluster_row[0]
                    label = cluster_row[1]
                    
                    # Get sample photos for this cluster
                    cursor.execute("""
                        SELECT p.path, f.photo_id
                        FROM faces f
                        JOIN photos p ON f.photo_id = p.id
                        WHERE f.cluster_id = ?
                        ORDER BY p.timestamp DESC
                        LIMIT 5
                    """, (cluster_id,))
                    
                    sample_photos = []
                    photo_count = 0
                    
                    for photo_row in cursor.fetchall():
                        sample_photos.append({
                            "path": photo_row[0],
                            "filename": os.path.basename(photo_row[0]),
                            "photo_id": photo_row[1]
                        })
                        photo_count += 1
                    
                    # Get total photo count
                    cursor.execute("""
                        SELECT COUNT(DISTINCT photo_id) 
                        FROM faces 
                        WHERE cluster_id = ?
                    """, (cluster_id,))
                    
                    total_count = cursor.fetchone()[0]
                    
                    cluster = {
                        "cluster_id": cluster_id,
                        "label": label,
                        "photo_count": total_count,
                        "sample_photos": sample_photos
                    }
                    clusters.append(cluster)
                
                return clusters
                
        except Exception as e:
            logger.error(f"Error getting face clusters: {e}")
            return []
    
    def get_stats(self) -> Dict[str, int]:
        """Get system statistics"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                stats = {}
                
                # Count photos
                cursor.execute("SELECT COUNT(*) FROM photos")
                stats["photos"] = cursor.fetchone()[0]
                
                # Count faces
                cursor.execute("SELECT COUNT(*) FROM faces")
                stats["faces"] = cursor.fetchone()[0]
                
                # Count clusters
                cursor.execute("SELECT COUNT(DISTINCT cluster_id) FROM faces WHERE cluster_id IS NOT NULL")
                stats["clusters"] = cursor.fetchone()[0]
                
                # Count relationships
                cursor.execute("SELECT COUNT(*) FROM relationship_inferences")
                stats["relationships"] = cursor.fetchone()[0]
                
                # Count groups
                cursor.execute("SELECT COUNT(*) FROM groups")
                stats["groups"] = cursor.fetchone()[0]
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"photos": 0, "faces": 0, "clusters": 0, "relationships": 0, "groups": 0}
    
    def get_recent_photos(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent photos for browsing"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT id, path, timestamp, objects
                    FROM photos 
                    ORDER BY timestamp DESC, id DESC
                    LIMIT ?
                """, (limit,))
                
                photos = []
                for row in cursor.fetchall():
                    photo = {
                        "id": row[0],
                        "path": row[1],
                        "timestamp": row[2],
                        "objects": json.loads(row[3]) if row[3] else [],
                        "similarity": 1.0  # Default for browsing
                    }
                    photos.append(photo)
                
                return photos
                
        except Exception as e:
            logger.error(f"Error getting recent photos: {e}")
            return []
    
    def search_photos_simple(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Simple text search in photo metadata"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Search in objects and file paths
                search_term = f"%{query.lower()}%"
                
                cursor.execute("""
                    SELECT id, path, timestamp, objects
                    FROM photos 
                    WHERE LOWER(path) LIKE ? 
                       OR LOWER(objects) LIKE ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (search_term, search_term, limit))
                
                photos = []
                for row in cursor.fetchall():
                    photo = {
                        "id": row[0],
                        "path": row[1],
                        "timestamp": row[2],
                        "objects": json.loads(row[3]) if row[3] else [],
                        "similarity": 0.8  # Estimated similarity for simple search
                    }
                    photos.append(photo)
                
                return photos
                
        except Exception as e:
            logger.error(f"Error in simple search: {e}")
            return []
    
    def get_groups(self) -> List[Dict[str, Any]]:
        """Get all groups"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT group_name, cluster_ids 
                    FROM groups 
                    ORDER BY group_name
                """)
                
                groups = []
                for row in cursor.fetchall():
                    group = {
                        "group_name": row[0],
                        "cluster_ids": json.loads(row[1]) if row[1] else [],
                        "member_count": len(json.loads(row[1])) if row[1] else 0
                    }
                    groups.append(group)
                
                return groups
                
        except Exception as e:
            logger.error(f"Error getting groups: {e}")
            return []
    
    def get_relationships(self) -> List[Dict[str, Any]]:
        """Get all relationships"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT r.cluster_id_a, r.cluster_id_b, ri.inferred_type, ri.confidence,
                           c1.label as person1_label, c2.label as person2_label
                    FROM relationships r
                    LEFT JOIN relationship_inferences ri ON r.cluster_id_a = ri.cluster_id_a AND r.cluster_id_b = ri.cluster_id_b
                    LEFT JOIN face_clusters c1 ON r.cluster_id_a = c1.cluster_id
                    LEFT JOIN face_clusters c2 ON r.cluster_id_b = c2.cluster_id
                    ORDER BY ri.confidence DESC
                """)
                
                relationships = []
                for row in cursor.fetchall():
                    relationship = {
                        "person1_cluster": row[0],
                        "person2_cluster": row[1],
                        "relationship_type": row[2] or "unknown",
                        "confidence": row[3] or 0.0,
                        "person1_label": row[4],
                        "person2_label": row[5]
                    }
                    relationships.append(relationship)
                
                return relationships
                
        except Exception as e:
            logger.error(f"Error getting relationships: {e}")
            return []
# ...existing code...

    def label_face_cluster(self, cluster_id: str, name: str) -> bool:
        """Label a face cluster (person) with a name"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE face_clusters
                    SET label = ?
                    WHERE cluster_id = ?
                """, (name, cluster_id))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error labeling cluster {cluster_id}: {e}")
            return False

    def create_group(self, group_name: str, cluster_ids: list) -> bool:
        """Create a new group with given cluster IDs"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO groups (group_name, cluster_ids)
                    VALUES (?, ?)
                """, (group_name, json.dumps(cluster_ids)))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error creating group {group_name}: {e}")
            return False

    def get_relationships_for_person(self, cluster_id: str) -> list:
        """Get relationships for a specific person (cluster)"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT r.cluster_id_a, r.cluster_id_b, ri.inferred_type, ri.confidence,
                           c1.label as person1_label, c2.label as person2_label
                    FROM relationships r
                    LEFT JOIN relationship_inferences ri ON r.cluster_id_a = ri.cluster_id_a AND r.cluster_id_b = ri.cluster_id_b
                    LEFT JOIN face_clusters c1 ON r.cluster_id_a = c1.cluster_id
                    LEFT JOIN face_clusters c2 ON r.cluster_id_b = c2.cluster_id
                    WHERE r.cluster_id_a = ? OR r.cluster_id_b = ?
                    ORDER BY ri.confidence DESC
                """, (cluster_id, cluster_id))
                relationships = []
                for row in cursor.fetchall():
                    relationship = {
                        "person1_cluster": row[0],
                        "person2_cluster": row[1],
                        "relationship_type": row[2] or "unknown",
                        "confidence": row[3] or 0.0,
                        "person1_label": row[4],
                        "person2_label": row[5]
                    }
                    relationships.append(relationship)
                return relationships
        except Exception as e:
            logger.error(f"Error getting relationships for cluster {cluster_id}: {e}")
            return []

# ...existing code...