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
