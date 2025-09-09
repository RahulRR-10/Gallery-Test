"""
Phase 3: Build SQLite Database
On-Device Photo Search Prototype - Database Implementation
"""

import sqlite3
import numpy as np
import pickle
import os
import time
from typing import List, Tuple, Optional
from datetime import datetime

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
                    clip_embedding BLOB NOT NULL,
                    file_size INTEGER,
                    image_width INTEGER,
                    image_height INTEGER,
                    created_date TEXT,
                    indexed_date TEXT
                )
            ''')
            
            # Create index on path for faster lookups
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_path ON photos(path)
            ''')
            
            # Create index on timestamp for chronological queries
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_timestamp ON photos(timestamp)
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
            
            # Get current datetime
            indexed_date = datetime.now().isoformat()
            created_date = datetime.fromtimestamp(timestamp).isoformat()
            
            # Insert the record
            cursor.execute('''
                INSERT OR REPLACE INTO photos 
                (id, path, timestamp, clip_embedding, file_size, image_width, image_height, created_date, indexed_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (photo_id, path, timestamp, embedding_blob, file_size, image_width, image_height, created_date, indexed_date))
            
            conn.commit()
            conn.close()
            
            print(f"📸 Inserted photo: {photo_id} -> {path}")
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
                SELECT id, path, timestamp, file_size, image_width, image_height, created_date, indexed_date
                FROM photos WHERE id = ?
            ''', (photo_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    'id': row[0],
                    'path': row[1],
                    'timestamp': row[2],
                    'file_size': row[3],
                    'image_width': row[4],
                    'image_height': row[5],
                    'created_date': row[6],
                    'indexed_date': row[7]
                }
            return None
            
        except Exception as e:
            print(f"❌ Error retrieving photo {photo_id}: {e}")
            return None
    
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
