"""
Phase 3: Build SQLite Database
On-Device Photo Search Prototype - Database Implementation
"""

import sqlite3
import numpy as np
import pickle
import os
import time
import re
import json
from typing import List, Tuple, Optional
from datetime import datetime
from PIL import Image
from PIL.ExifTags import TAGS

class PhotoDatabase:
    """SQLite database for storing photo embeddings and metadata"""
    
    def __init__(self, db_path: str = "photos.db"):
        """
        Initialize the photo database
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        print(f"🗄️ Initializing photo database: {db_path}")
        
        # Create database and table if they don't exist
        self.create_database()
        
    def create_database(self):
        """Create the database and photos table"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create photos table with all required columns
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS photos (
                    id TEXT PRIMARY KEY,
                    path TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    exif_timestamp INTEGER,
                    clip_embedding BLOB NOT NULL,
                    file_size INTEGER,
                    image_width INTEGER,
                    image_height INTEGER,
                    created_date TEXT,
                    exif_date TEXT,
                    indexed_date TEXT
                )
            ''')
            
            # Add exif_timestamp column if it doesn't exist (for existing databases)
            try:
                cursor.execute('ALTER TABLE photos ADD COLUMN exif_timestamp INTEGER')
                cursor.execute('ALTER TABLE photos ADD COLUMN exif_date TEXT')
            except sqlite3.OperationalError:
                # Columns already exist
                pass
            
            # Create index on path for faster lookups
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_path ON photos(path)
            ''')
            
            # Create index on timestamp for chronological queries
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_timestamp ON photos(timestamp)
            ''')
            
            # Create index on EXIF timestamp for temporal queries
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_exif_timestamp ON photos(exif_timestamp)
            ''')

            # Stage 2: Faces table (per-face records)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS faces (
                    id TEXT PRIMARY KEY,
                    photo_id TEXT NOT NULL,
                    bbox TEXT NOT NULL,
                    embedding BLOB,
                    method TEXT,
                    cluster_id TEXT,
                    created_at TEXT,
                    FOREIGN KEY(photo_id) REFERENCES photos(id)
                )
            ''')

            # Stage 2: Face clusters (labels and counts)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS face_clusters (
                    cluster_id TEXT PRIMARY KEY,
                    label TEXT,
                    num_faces INTEGER DEFAULT 0,
                    updated_at TEXT
                )
            ''')

            # Stage 2: Relationships (co-occurrence of clusters)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS relationships (
                    cluster_id_a TEXT NOT NULL,
                    cluster_id_b TEXT NOT NULL,
                    count INTEGER DEFAULT 0,
                    weight REAL DEFAULT 0,
                    PRIMARY KEY (cluster_id_a, cluster_id_b)
                )
            ''')

            # Phase 3: Groups table for organizing clusters into named groups
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS groups (
                    group_name TEXT PRIMARY KEY,
                    cluster_ids TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT
                )
            ''')

            # Indexes for Stage 2 tables
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_faces_photo ON faces(photo_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_faces_cluster ON faces(cluster_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_rel_a ON relationships(cluster_id_a)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_rel_b ON relationships(cluster_id_b)')
            
            conn.commit()
            conn.close()
            
            print("✅ Database and table created successfully!")
            
        except Exception as e:
            print(f"❌ Error creating database: {e}")
            raise
    
    def serialize_embedding(self, embedding: np.ndarray) -> bytes:
        """Convert NumPy array to bytes for storage"""
        return pickle.dumps(embedding.astype(np.float32))
    
    def deserialize_embedding(self, embedding_bytes: bytes) -> np.ndarray:
        """Convert bytes back to NumPy array"""
        return pickle.loads(embedding_bytes)
    
    def extract_exif_timestamp(self, image_path: str) -> Tuple[Optional[int], Optional[str]]:
        """
        Extract EXIF timestamp from image file
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (unix_timestamp, iso_date_string) or (None, None) if not found
        """
        try:
            with Image.open(image_path) as img:
                exif_data = img.getexif()
                
                if exif_data:
                    # Try different EXIF date fields in order of preference
                    date_fields = [
                        'DateTime',           # Date and time of image creation
                        'DateTimeOriginal',   # Date and time when original image was taken
                        'DateTimeDigitized'   # Date and time when image was digitized
                    ]
                    
                    for field in date_fields:
                        for tag_id, value in exif_data.items():
                            tag = TAGS.get(tag_id, tag_id)
                            if tag == field and value:
                                try:
                                    # Parse EXIF date format: "YYYY:MM:DD HH:MM:SS"
                                    dt = datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
                                    unix_timestamp = int(dt.timestamp())
                                    iso_date = dt.isoformat()
                                    return unix_timestamp, iso_date
                                except ValueError:
                                    continue
                    
                    # If no standard date fields, try to extract from GPSInfo or other fields
                    # This is for future enhancement
                    
            # If EXIF extraction fails, try to extract date from filename
            filename = os.path.basename(image_path)
            
            # Pattern for Windows Camera app: WIN_YYYYMMDD_HH_MM_SS_Pro.jpg
            import re
            win_pattern = r'WIN_(\d{8})_(\d{2})_(\d{2})_(\d{2})_'
            match = re.search(win_pattern, filename)
            if match:
                date_str = match.group(1)  # YYYYMMDD
                hour = match.group(2)
                minute = match.group(3)
                second = match.group(4)
                
                try:
                    # Parse the date components
                    year = int(date_str[:4])
                    month = int(date_str[4:6])
                    day = int(date_str[6:8])
                    
                    dt = datetime(year, month, day, int(hour), int(minute), int(second))
                    unix_timestamp = int(dt.timestamp())
                    iso_date = dt.isoformat()
                    print(f"📅 Extracted date from filename: {filename} -> {iso_date}")
                    return unix_timestamp, iso_date
                except ValueError as e:
                    print(f"⚠️ Error parsing filename date {filename}: {e}")
            
            # Pattern for IMG_YYYYMMDD_HHMMSS.jpg or similar
            img_pattern = r'(\d{8})_(\d{6})'
            match = re.search(img_pattern, filename)
            if match:
                date_str = match.group(1)  # YYYYMMDD
                time_str = match.group(2)  # HHMMSS
                
                try:
                    year = int(date_str[:4])
                    month = int(date_str[4:6])
                    day = int(date_str[6:8])
                    hour = int(time_str[:2])
                    minute = int(time_str[2:4])
                    second = int(time_str[4:6])
                    
                    dt = datetime(year, month, day, hour, minute, second)
                    unix_timestamp = int(dt.timestamp())
                    iso_date = dt.isoformat()
                    print(f"📅 Extracted date from filename: {filename} -> {iso_date}")
                    return unix_timestamp, iso_date
                except ValueError as e:
                    print(f"⚠️ Error parsing filename date {filename}: {e}")
            
            # Pattern for WhatsApp: IMG-YYYYMMDD-WA*.jpg
            wa_pattern = r'IMG-(\d{8})-WA'
            match = re.search(wa_pattern, filename)
            if match:
                date_str = match.group(1)  # YYYYMMDD
                
                try:
                    year = int(date_str[:4])
                    month = int(date_str[4:6])
                    day = int(date_str[6:8])
                    
                    # WhatsApp doesn't include time, default to noon
                    dt = datetime(year, month, day, 12, 0, 0)
                    unix_timestamp = int(dt.timestamp())
                    iso_date = dt.isoformat()
                    print(f"📅 Extracted date from WhatsApp filename: {filename} -> {iso_date}")
                    return unix_timestamp, iso_date
                except ValueError as e:
                    print(f"⚠️ Error parsing WhatsApp filename date {filename}: {e}")
                                    
        except Exception as e:
            print(f"⚠️ Could not extract EXIF from {image_path}: {e}")
            
        return None, None
    
    def insert_photo(self, photo_id: str, path: str, timestamp: int, embedding: np.ndarray, 
                    file_size: Optional[int] = None, image_width: Optional[int] = None, 
                    image_height: Optional[int] = None) -> bool:
        """
        Insert a photo record into the database
        
        Args:
            photo_id: Unique identifier for the photo
            path: File path to the photo
            timestamp: Unix timestamp when photo was taken/created
            embedding: CLIP embedding vector
            file_size: Size of the image file in bytes
            image_width: Width of the image in pixels
            image_height: Height of the image in pixels
            
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Serialize the embedding
            embedding_blob = self.serialize_embedding(embedding)
            
            # Extract EXIF timestamp
            exif_timestamp, exif_date = self.extract_exif_timestamp(path)
            
            # Get current datetime
            indexed_date = datetime.now().isoformat()
            created_date = datetime.fromtimestamp(timestamp).isoformat()
            
            # Insert the record with EXIF data
            cursor.execute('''
                INSERT OR REPLACE INTO photos 
                (id, path, timestamp, exif_timestamp, clip_embedding, file_size, image_width, image_height, created_date, exif_date, indexed_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (photo_id, path, timestamp, exif_timestamp, embedding_blob, file_size, image_width, image_height, created_date, exif_date, indexed_date))
            
            conn.commit()
            conn.close()
            
            # Log EXIF extraction result
            if exif_timestamp:
                print(f"📸 Inserted photo: {photo_id} -> {path} (EXIF: {exif_date})")
            else:
                print(f"📸 Inserted photo: {photo_id} -> {path} (No EXIF date)")
            return True
            
        except Exception as e:
            print(f"❌ Error inserting photo {photo_id}: {e}")
            return False
    
    def get_all_embeddings(self) -> List[Tuple[str, str, np.ndarray]]:
        """
        Retrieve all photo embeddings from the database
        
        Returns:
            List of tuples: (photo_id, path, embedding)
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT id, path, clip_embedding FROM photos')
            rows = cursor.fetchall()
            
            conn.close()
            
            # Deserialize embeddings
            results = []
            for photo_id, path, embedding_blob in rows:
                embedding = self.deserialize_embedding(embedding_blob)
                results.append((photo_id, path, embedding))
            
            print(f"📊 Retrieved {len(results)} photo embeddings")
            return results
            
        except Exception as e:
            print(f"❌ Error retrieving embeddings: {e}")
            return []
    
    def get_photo_by_id(self, photo_id: str) -> Optional[dict]:
        """
        Get a specific photo record by ID
        
        Args:
            photo_id: Unique identifier for the photo
            
        Returns:
            Dictionary with photo metadata or None if not found
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, path, timestamp, exif_timestamp, file_size, image_width, image_height, 
                       created_date, exif_date, indexed_date
                FROM photos WHERE id = ?
            ''', (photo_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    'id': row[0],
                    'path': row[1],
                    'timestamp': row[2],
                    'exif_timestamp': row[3],
                    'file_size': row[4],
                    'image_width': row[5],
                    'image_height': row[6],
                    'created_date': row[7],
                    'exif_date': row[8],
                    'indexed_date': row[9]
                }
            return None
            
        except Exception as e:
            print(f"❌ Error retrieving photo {photo_id}: {e}")
            return None

    def get_photo_by_path(self, path: str) -> Optional[dict]:
        """Lookup a photo row by path (handles both relative and absolute paths)."""
        try:
            import os as _os
            canon = _os.path.abspath(path)
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Try exact match first
            cursor.execute('''
                SELECT id, path, timestamp, exif_timestamp, file_size, image_width, image_height, 
                       created_date, exif_date, indexed_date
                FROM photos WHERE path = ?
            ''', (canon,))
            row = cursor.fetchone()
            
            # If not found, try case-insensitive filename match
            if not row:
                filename = _os.path.basename(canon)
                cursor.execute('''
                    SELECT id, path, timestamp, exif_timestamp, file_size, image_width, image_height, 
                           created_date, exif_date, indexed_date
                    FROM photos WHERE LOWER(path) LIKE LOWER(?)
                ''', (f'%{filename}',))
                row = cursor.fetchone()
            
            conn.close()
            if row:
                return {
                    'id': row[0],
                    'path': row[1],
                    'timestamp': row[2],
                    'exif_timestamp': row[3],
                    'file_size': row[4],
                    'image_width': row[5],
                    'image_height': row[6],
                    'created_date': row[7],
                    'exif_date': row[8],
                    'indexed_date': row[9]
                }
            return None
        except Exception as e:
            print(f"❌ Error retrieving photo by path {path}: {e}")
            return None
    
    def search_photos_by_time(self, start_timestamp: Optional[int] = None, 
                             end_timestamp: Optional[int] = None,
                             use_exif: bool = True) -> List[Tuple[str, str, int, Optional[int]]]:
        """
        Search photos by time range
        
        Args:
            start_timestamp: Start of time range (Unix timestamp)
            end_timestamp: End of time range (Unix timestamp)  
            use_exif: Whether to use EXIF timestamps (True) or file timestamps (False)
            
        Returns:
            List of tuples: (photo_id, path, timestamp, exif_timestamp)
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Choose which timestamp field to use
            timestamp_field = 'exif_timestamp' if use_exif else 'timestamp'
            
            # Build query based on time range
            if start_timestamp and end_timestamp:
                query = f'''
                    SELECT id, path, timestamp, exif_timestamp 
                    FROM photos 
                    WHERE {timestamp_field} BETWEEN ? AND ?
                    ORDER BY {timestamp_field} DESC
                '''
                cursor.execute(query, (start_timestamp, end_timestamp))
            elif start_timestamp:
                query = f'''
                    SELECT id, path, timestamp, exif_timestamp 
                    FROM photos 
                    WHERE {timestamp_field} >= ?
                    ORDER BY {timestamp_field} DESC
                '''
                cursor.execute(query, (start_timestamp,))
            elif end_timestamp:
                query = f'''
                    SELECT id, path, timestamp, exif_timestamp 
                    FROM photos 
                    WHERE {timestamp_field} <= ?
                    ORDER BY {timestamp_field} DESC
                '''
                cursor.execute(query, (end_timestamp,))
            else:
                # No time filter, return all photos ordered by time
                query = f'''
                    SELECT id, path, timestamp, exif_timestamp 
                    FROM photos 
                    WHERE {timestamp_field} IS NOT NULL
                    ORDER BY {timestamp_field} DESC
                '''
                cursor.execute(query)
            
            rows = cursor.fetchall()
            conn.close()
            
            results = [(row[0], row[1], row[2], row[3]) for row in rows]
            print(f"🕒 Found {len(results)} photos in time range")
            return results
            
        except Exception as e:
            print(f"❌ Error searching photos by time: {e}")
            return []
    
    def delete_photo(self, photo_id: str) -> bool:
        """
        Delete a photo record from the database
        
        Args:
            photo_id: Unique identifier for the photo
            
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM photos WHERE id = ?', (photo_id,))
            rows_affected = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            if rows_affected > 0:
                print(f"🗑️ Deleted photo: {photo_id}")
                return True
            else:
                print(f"⚠️ Photo not found: {photo_id}")
                return False
                
        except Exception as e:
            print(f"❌ Error deleting photo {photo_id}: {e}")
            return False
    
    def get_database_stats(self) -> dict:
        """
        Get statistics about the database
        
        Returns:
            Dictionary with database statistics
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get total count
            cursor.execute('SELECT COUNT(*) FROM photos')
            total_photos = cursor.fetchone()[0]
            
            # Get database file size
            db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
            
            # Get date range
            cursor.execute('SELECT MIN(timestamp), MAX(timestamp) FROM photos')
            date_range = cursor.fetchone()
            
            conn.close()
            
            stats = {
                'total_photos': total_photos,
                'database_size_mb': db_size / (1024 * 1024),
                'earliest_photo': datetime.fromtimestamp(date_range[0]).isoformat() if date_range[0] else None,
                'latest_photo': datetime.fromtimestamp(date_range[1]).isoformat() if date_range[1] else None
            }
            
            return stats
            
        except Exception as e:
            print(f"❌ Error getting database stats: {e}")
            return {}
    
    def clear_database(self) -> bool:
        """
        Clear all records from the database (for testing)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM photos')
            conn.commit()
            conn.close()
            
            print("🧹 Database cleared successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error clearing database: {e}")
            return False

    # =====================
    # Stage 2: Faces & People
    # =====================

    def insert_face(self, face_id: str, photo_id: str, bbox: str, embedding: Optional[np.ndarray], method: Optional[str]) -> bool:
        """Insert a face record."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            embedding_blob = self.serialize_embedding(embedding) if isinstance(embedding, np.ndarray) else None
            created_at = datetime.now().isoformat()
            cursor.execute('''
                INSERT OR REPLACE INTO faces (id, photo_id, bbox, embedding, method, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (face_id, photo_id, bbox, embedding_blob, method, created_at))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"❌ Error inserting face {face_id}: {e}")
            return False

    def get_faces_by_photo(self, photo_id: str) -> List[dict]:
        """Get all faces for a photo."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT id, bbox, embedding, method, cluster_id FROM faces WHERE photo_id = ?', (photo_id,))
            rows = cursor.fetchall()
            conn.close()
            results = []
            for fid, bbox, emb_blob, method, cluster_id in rows:
                embedding = self.deserialize_embedding(emb_blob) if emb_blob else None
                results.append({'id': fid, 'bbox': bbox, 'embedding': embedding, 'method': method, 'cluster_id': cluster_id})
            return results
        except Exception as e:
            print(f"❌ Error retrieving faces for photo {photo_id}: {e}")
            return []

    def get_all_face_embeddings(self) -> List[Tuple[str, np.ndarray]]:
        """Return (face_id, embedding) for all faces with embeddings."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT id, embedding FROM faces WHERE embedding IS NOT NULL')
            rows = cursor.fetchall()
            conn.close()
            results = []
            for fid, emb_blob in rows:
                try:
                    embedding = self.deserialize_embedding(emb_blob)
                    if isinstance(embedding, np.ndarray):
                        results.append((fid, embedding.astype(np.float32)))
                except Exception:
                    continue
            return results
        except Exception as e:
            print(f"❌ Error retrieving face embeddings: {e}")
            return []

    def assign_cluster_to_faces(self, cluster_id: str, face_ids: List[str]) -> None:
        """Assign a cluster_id to given face_ids and update cluster counts."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.executemany('UPDATE faces SET cluster_id = ? WHERE id = ?', [(cluster_id, fid) for fid in face_ids])
            # Update cluster count
            cursor.execute('SELECT COUNT(*) FROM faces WHERE cluster_id = ?', (cluster_id,))
            num = cursor.fetchone()[0]
            updated_at = datetime.now().isoformat()
            cursor.execute('''
                INSERT INTO face_clusters (cluster_id, num_faces, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(cluster_id)
                DO UPDATE SET num_faces=excluded.num_faces, updated_at=excluded.updated_at
            ''', (cluster_id, num, updated_at))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"❌ Error assigning cluster {cluster_id}: {e}")

    def get_unclustered_faces_with_embeddings(self) -> List[Tuple[str, str, np.ndarray]]:
        """Return (face_id, photo_id, embedding) for faces without cluster and with embeddings."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT id, photo_id, embedding FROM faces WHERE cluster_id IS NULL AND embedding IS NOT NULL')
            rows = cursor.fetchall()
            conn.close()
            results = []
            for fid, pid, emb_blob in rows:
                try:
                    emb = self.deserialize_embedding(emb_blob)
                    results.append((fid, pid, emb.astype(np.float32)))
                except Exception:
                    continue
            return results
        except Exception as e:
            print(f"❌ Error retrieving unclustered faces: {e}")
            return []

    def get_cluster_centroids(self) -> List[Tuple[str, np.ndarray, int]]:
        """Return (cluster_id, centroid_embedding, count)."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT DISTINCT cluster_id FROM faces WHERE cluster_id IS NOT NULL')
            clusters = [r[0] for r in cursor.fetchall() if r[0]]
            centroids: List[Tuple[str, np.ndarray, int]] = []
            for cid in clusters:
                cursor.execute('SELECT embedding FROM faces WHERE cluster_id = ? AND embedding IS NOT NULL', (cid,))
                embs = []
                for (emb_blob,) in cursor.fetchall():
                    try:
                        embs.append(self.deserialize_embedding(emb_blob).astype(np.float32))
                    except Exception:
                        continue
                if embs:
                    import numpy as _np
                    mat = _np.stack(embs).astype(_np.float32)
                    # L2-normalize then average then renormalize
                    mat = mat / ( _np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8 )
                    centroid = mat.mean(axis=0)
                    centroid = centroid / ( _np.linalg.norm(centroid) + 1e-8 )
                    centroids.append((cid, centroid, len(embs)))
            conn.close()
            return centroids
        except Exception as e:
            print(f"❌ Error computing cluster centroids: {e}")
            return []

    def upsert_cluster(self, cluster_id: str, label: Optional[str] = None) -> None:
        """Create or update a cluster record."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            updated_at = datetime.now().isoformat()
            cursor.execute('''
                INSERT INTO face_clusters (cluster_id, label, num_faces, updated_at)
                VALUES (?, ?, COALESCE((SELECT COUNT(*) FROM faces WHERE cluster_id = ?),0), ?)
                ON CONFLICT(cluster_id)
                DO UPDATE SET label=excluded.label, num_faces=excluded.num_faces, updated_at=excluded.updated_at
            ''', (cluster_id, label, cluster_id, updated_at))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"❌ Error upserting cluster {cluster_id}: {e}")

    def label_cluster(self, cluster_id: str, label: str) -> None:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            updated_at = datetime.now().isoformat()
            cursor.execute('UPDATE face_clusters SET label = ?, updated_at = ? WHERE cluster_id = ?', (label, updated_at, cluster_id))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"❌ Error labeling cluster {cluster_id}: {e}")

    def get_clusters(self) -> List[dict]:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT cluster_id, label, num_faces, updated_at FROM face_clusters ORDER BY num_faces DESC')
            rows = cursor.fetchall()
            conn.close()
            return [{'cluster_id': r[0], 'label': r[1], 'num_faces': r[2], 'updated_at': r[3]} for r in rows]
        except Exception as e:
            print(f"❌ Error retrieving clusters: {e}")
            return []

    def get_cluster_by_label(self, label: str) -> Optional[dict]:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT cluster_id, label, num_faces FROM face_clusters WHERE label = ?', (label,))
            row = cursor.fetchone()
            conn.close()
            if row:
                return {'cluster_id': row[0], 'label': row[1], 'num_faces': row[2]}
            return None
        except Exception as e:
            print(f"❌ Error getting cluster by label '{label}': {e}")
            return None

    def build_relationships_from_photos(self) -> None:
        """Compute co-occurrence relationships between clusters from per-photo faces."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            # Get all photo_ids
            cursor.execute('SELECT id FROM photos')
            photos = [r[0] for r in cursor.fetchall()]
            from itertools import combinations
            rel_counts = {}
            for pid in photos:
                cursor.execute('SELECT DISTINCT cluster_id FROM faces WHERE photo_id = ? AND cluster_id IS NOT NULL', (pid,))
                clusters = [r[0] for r in cursor.fetchall() if r[0]]
                clusters = sorted(set(clusters))
                for a, b in combinations(clusters, 2):
                    key = (a, b)
                    rel_counts[key] = rel_counts.get(key, 0) + 1
            # Upsert relationships
            for (a, b), cnt in rel_counts.items():
                weight = float(cnt)
                cursor.execute('''
                    INSERT INTO relationships (cluster_id_a, cluster_id_b, count, weight)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(cluster_id_a, cluster_id_b)
                    DO UPDATE SET count=excluded.count, weight=excluded.weight
                ''', (a, b, cnt, weight))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"❌ Error building relationships: {e}")

    def get_related_clusters(self, cluster_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT cluster_id_b, weight FROM relationships WHERE cluster_id_a = ? ORDER BY weight DESC LIMIT ?', (cluster_id, top_k))
            rows1 = cursor.fetchall()
            cursor.execute('SELECT cluster_id_a, weight FROM relationships WHERE cluster_id_b = ? ORDER BY weight DESC LIMIT ?', (cluster_id, top_k))
            rows2 = cursor.fetchall()
            conn.close()
            results = rows1 + rows2
            return results
        except Exception as e:
            print(f"❌ Error getting related clusters: {e}")
            return []

    def get_photos_with_clusters(self, cluster_ids: List[str]) -> List[str]:
        """Return photo_ids that contain any of the cluster_ids."""
        if not cluster_ids:
            return []
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            qmarks = ','.join('?' for _ in cluster_ids)
            cursor.execute(f'SELECT DISTINCT photo_id FROM faces WHERE cluster_id IN ({qmarks})', tuple(cluster_ids))
            rows = cursor.fetchall()
            conn.close()
            return [r[0] for r in rows]
        except Exception as e:
            print(f"❌ Error getting photos with clusters: {e}")
            return []

    # Phase 3: Group Management Methods
    def create_group(self, group_name: str, cluster_ids: List[str]) -> bool:
        """
        Create a new group with specified cluster IDs
        
        Args:
            group_name: Name of the group (e.g., "family", "coworkers")
            cluster_ids: List of cluster IDs to include in the group
            
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Verify all cluster IDs exist
            for cluster_id in cluster_ids:
                cursor.execute('SELECT cluster_id FROM face_clusters WHERE cluster_id = ?', (cluster_id,))
                if not cursor.fetchone():
                    print(f"❌ Cluster ID '{cluster_id}' does not exist")
                    conn.close()
                    return False
            
            # Create group (will replace if exists)
            cluster_ids_json = json.dumps(cluster_ids)
            created_at = datetime.now().isoformat()
            
            cursor.execute('''
                INSERT OR REPLACE INTO groups (group_name, cluster_ids, created_at, updated_at)
                VALUES (?, ?, ?, ?)
            ''', (group_name, cluster_ids_json, created_at, created_at))
            
            conn.commit()
            conn.close()
            
            print(f"✅ Group '{group_name}' created with {len(cluster_ids)} clusters")
            return True
            
        except Exception as e:
            print(f"❌ Error creating group: {e}")
            return False

    def add_to_group(self, group_name: str, cluster_id: str) -> bool:
        """
        Add a cluster to an existing group
        
        Args:
            group_name: Name of the existing group
            cluster_id: Cluster ID to add
            
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Verify cluster exists
            cursor.execute('SELECT cluster_id FROM face_clusters WHERE cluster_id = ?', (cluster_id,))
            if not cursor.fetchone():
                print(f"❌ Cluster ID '{cluster_id}' does not exist")
                conn.close()
                return False
            
            # Get existing group
            cursor.execute('SELECT cluster_ids FROM groups WHERE group_name = ?', (group_name,))
            row = cursor.fetchone()
            if not row:
                print(f"❌ Group '{group_name}' does not exist")
                conn.close()
                return False
            
            # Add cluster if not already in group
            existing_cluster_ids = json.loads(row[0])
            if cluster_id not in existing_cluster_ids:
                existing_cluster_ids.append(cluster_id)
                updated_cluster_ids = json.dumps(existing_cluster_ids)
                updated_at = datetime.now().isoformat()
                
                cursor.execute('''
                    UPDATE groups SET cluster_ids = ?, updated_at = ?
                    WHERE group_name = ?
                ''', (updated_cluster_ids, updated_at, group_name))
                
                conn.commit()
                print(f"✅ Added cluster '{cluster_id}' to group '{group_name}'")
            else:
                print(f"ℹ️ Cluster '{cluster_id}' already in group '{group_name}'")
            
            conn.close()
            return True
            
        except Exception as e:
            print(f"❌ Error adding to group: {e}")
            return False

    def remove_from_group(self, group_name: str, cluster_id: str) -> bool:
        """
        Remove a cluster from a group
        
        Args:
            group_name: Name of the group
            cluster_id: Cluster ID to remove
            
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get existing group
            cursor.execute('SELECT cluster_ids FROM groups WHERE group_name = ?', (group_name,))
            row = cursor.fetchone()
            if not row:
                print(f"❌ Group '{group_name}' does not exist")
                conn.close()
                return False
            
            # Remove cluster if in group
            existing_cluster_ids = json.loads(row[0])
            if cluster_id in existing_cluster_ids:
                existing_cluster_ids.remove(cluster_id)
                updated_cluster_ids = json.dumps(existing_cluster_ids)
                updated_at = datetime.now().isoformat()
                
                cursor.execute('''
                    UPDATE groups SET cluster_ids = ?, updated_at = ?
                    WHERE group_name = ?
                ''', (updated_cluster_ids, updated_at, group_name))
                
                conn.commit()
                print(f"✅ Removed cluster '{cluster_id}' from group '{group_name}'")
            else:
                print(f"ℹ️ Cluster '{cluster_id}' not in group '{group_name}'")
            
            conn.close()
            return True
            
        except Exception as e:
            print(f"❌ Error removing from group: {e}")
            return False

    def list_groups(self) -> List[dict]:
        """
        List all groups and their members
        
        Returns:
            List of group dictionaries with name, cluster_ids, and metadata
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT group_name, cluster_ids, created_at, updated_at
                FROM groups
                ORDER BY group_name
            ''')
            
            groups = []
            for row in cursor.fetchall():
                group_name, cluster_ids_json, created_at, updated_at = row
                cluster_ids = json.loads(cluster_ids_json)
                
                # Get labels for clusters
                cluster_labels = {}
                for cluster_id in cluster_ids:
                    cursor.execute('SELECT label FROM face_clusters WHERE cluster_id = ?', (cluster_id,))
                    label_row = cursor.fetchone()
                    cluster_labels[cluster_id] = label_row[0] if label_row and label_row[0] else "Unlabeled"
                
                groups.append({
                    'group_name': group_name,
                    'cluster_ids': cluster_ids,
                    'cluster_labels': cluster_labels,
                    'created_at': created_at,
                    'updated_at': updated_at
                })
            
            conn.close()
            return groups
            
        except Exception as e:
            print(f"❌ Error listing groups: {e}")
            return []

    def get_group_cluster_ids(self, group_name: str) -> List[str]:
        """
        Get cluster IDs for a specific group
        
        Args:
            group_name: Name of the group
            
        Returns:
            List of cluster IDs in the group
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT cluster_ids FROM groups WHERE group_name = ?', (group_name,))
            row = cursor.fetchone()
            
            conn.close()
            
            if row:
                return json.loads(row[0])
            else:
                print(f"❌ Group '{group_name}' not found")
                return []
                
        except Exception as e:
            print(f"❌ Error getting group cluster IDs: {e}")
            return []

    def delete_group(self, group_name: str) -> bool:
        """
        Delete a group
        
        Args:
            group_name: Name of the group to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM groups WHERE group_name = ?', (group_name,))
            rows_affected = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            if rows_affected > 0:
                print(f"✅ Group '{group_name}' deleted")
                return True
            else:
                print(f"❌ Group '{group_name}' not found")
                return False
                
        except Exception as e:
            print(f"❌ Error deleting group: {e}")
            return False


def test_database():
    """Test the photo database implementation"""
    print("\n" + "="*60)
    print("🧪 Testing Photo Database Implementation")
    print("="*60)
    
    try:
        # Initialize database
        db = PhotoDatabase("test_photos.db")
        
        # Clear database for clean test
        db.clear_database()
        
        # Create test embeddings
        print("\n📊 Creating test data...")
        test_photos = [
            {
                'id': 'photo_001',
                'path': '/sample_photos/sunset.jpg',
                'timestamp': int(time.time()) - 86400,  # Yesterday
                'embedding': np.random.randn(512).astype(np.float32),
                'file_size': 2048576,  # 2MB
                'width': 1920,
                'height': 1080
            },
            {
                'id': 'photo_002', 
                'path': '/sample_photos/dog.jpg',
                'timestamp': int(time.time()) - 43200,  # 12 hours ago
                'embedding': np.random.randn(512).astype(np.float32),
                'file_size': 1536000,  # 1.5MB
                'width': 1280,
                'height': 960
            },
            {
                'id': 'photo_003',
                'path': '/sample_photos/car.jpg', 
                'timestamp': int(time.time()),  # Now
                'embedding': np.random.randn(512).astype(np.float32),
                'file_size': 3072000,  # 3MB
                'width': 2048,
                'height': 1536
            }
        ]
        
        # Test insertion
        print("\n💾 Testing photo insertion...")
        for photo in test_photos:
            success = db.insert_photo(
                photo['id'], 
                photo['path'], 
                photo['timestamp'], 
                photo['embedding'],
                photo['file_size'],
                photo['width'],
                photo['height']
            )
            if not success:
                raise Exception(f"Failed to insert {photo['id']}")
        
        # Test retrieval of all embeddings
        print("\n📤 Testing embedding retrieval...")
        embeddings = db.get_all_embeddings()
        print(f"   Retrieved {len(embeddings)} embeddings")
        
        for photo_id, path, embedding in embeddings:
            print(f"   {photo_id}: {path} -> embedding shape: {embedding.shape}")
        
        # Test individual photo retrieval
        print("\n🔍 Testing individual photo retrieval...")
        photo_info = db.get_photo_by_id('photo_002')
        if photo_info:
            print(f"   Found photo: {photo_info['id']}")
            print(f"   Path: {photo_info['path']}")
            print(f"   Size: {photo_info['file_size']} bytes")
            print(f"   Dimensions: {photo_info['image_width']}x{photo_info['image_height']}")
            print(f"   Created: {photo_info['created_date']}")
        
        # Test database statistics
        print("\n📈 Testing database statistics...")
        stats = db.get_database_stats()
        print(f"   Total photos: {stats['total_photos']}")
        print(f"   Database size: {stats['database_size_mb']:.2f} MB")
        print(f"   Date range: {stats['earliest_photo']} to {stats['latest_photo']}")
        
        # Test deletion
        print("\n🗑️ Testing photo deletion...")
        deleted = db.delete_photo('photo_002')
        if deleted:
            print("   Photo deleted successfully")
        
        # Verify deletion
        remaining = db.get_all_embeddings()
        print(f"   Remaining photos: {len(remaining)}")
        
        # Test embedding serialization/deserialization
        print("\n🔄 Testing embedding serialization...")
        test_embedding = np.random.randn(512).astype(np.float32)
        serialized = db.serialize_embedding(test_embedding)
        deserialized = db.deserialize_embedding(serialized)
        
        if np.allclose(test_embedding, deserialized):
            print("   ✅ Embedding serialization working correctly")
        else:
            print("   ❌ Embedding serialization failed")
        
        print(f"\n✅ Database test completed successfully!")
        print(f"🎉 Phase 3 complete - ready for Phase 4 (Index Sample Images)!")
        
        # Cleanup test database
        os.remove("test_photos.db")
        print("🧹 Test database cleaned up")
        
        return db
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        raise


if __name__ == "__main__":
    # Test the database implementation
    test_db = test_database()
