#!/usr/bin/env python3
"""
Fast Face Clustering Script
Optimized for speed - only loads necessary components
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import numpy as np
from sklearn.cluster import DBSCAN

# Minimal imports
from photo_database import PhotoDatabase
from advanced_face_detection import AdvancedFaceDetector

def fast_face_detection():
    """Fast face detection - only load face detection models"""
    print("Starting fast face detection...")
    
    db = PhotoDatabase("photos.db")
    detector = AdvancedFaceDetector()
    
    # Get all photos directly from database
    import sqlite3
    conn = sqlite3.connect("photos.db")
    cursor = conn.cursor()
    cursor.execute('SELECT id, path FROM photos')
    photos = [{'id': row[0], 'path': row[1]} for row in cursor.fetchall()]
    conn.close()
    
    total = len(photos)
    processed = 0
    
    for photo in photos:
        try:
            photo_id = photo['id']
            photo_path = photo['path']
            
            # Skip if faces already detected
            conn = sqlite3.connect("photos.db")
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM faces WHERE photo_id = ?', (photo_id,))
            existing_count = cursor.fetchone()[0]
            conn.close()
            
            if existing_count > 0:
                processed += 1
                continue
            
            if not os.path.exists(photo_path):
                processed += 1
                continue
            
            # Detect faces
            faces = detector.detect_faces(photo_path)
            
            # Save to database
            for i, face in enumerate(faces):
                if 'embedding' in face and face['embedding'] is not None:
                    face_id = f"{photo_id}_face_{i}"
                    bbox_str = f"{face['bbox'][0]},{face['bbox'][1]},{face['bbox'][2]},{face['bbox'][3]}"
                    # Convert embedding list to numpy array
                    embedding = np.array(face['embedding'])
                    db.insert_face(
                        face_id,
                        photo_id,
                        bbox_str,
                        embedding,
                        "insightface"
                    )
            
            processed += 1
            
            if processed % 10 == 0:
                print(f"Processed {processed}/{total} photos")
                
        except Exception as e:
            print(f"Error processing {photo.get('path', 'unknown')}: {e}")
            processed += 1
            continue
    
    print("Face detection complete: {}/{} photos".format(processed, total))

def fast_face_clustering():
    """Fast face clustering - minimal dependencies"""
    print("Starting fast face clustering...")
    
    db = PhotoDatabase("photos.db")
    
    # Get face embeddings
    face_embeddings = db.get_all_face_embeddings()
    if not face_embeddings:
        print("WARNING: No face embeddings found to cluster")
        return
    
    print(f"Found {len(face_embeddings)} face embeddings")
    
    # Prepare data
    face_ids = [fid for fid, _ in face_embeddings]
    embeddings = [emb for _, emb in face_embeddings]
    X = np.stack([emb / (np.linalg.norm(emb) + 1e-8) for emb in embeddings]).astype(np.float32)
    
    # Cluster faces
    print("Clustering faces...")
    clustering = DBSCAN(eps=0.4, min_samples=3, metric='cosine')
    labels = clustering.fit_predict(X)
    
    # Group into clusters
    clusters = {}
    for fid, lbl in zip(face_ids, labels):
        if lbl < 0:  # Skip noise
            continue
        clusters.setdefault(f"cluster_{lbl}", []).append(fid)
    
    if not clusters:
        print("WARNING: No clusters found (all noise)")
        return
    
    # Save clusters
    print("Saving {} clusters...".format(len(clusters)))
    for cluster_id, members in clusters.items():
        db.assign_cluster_to_faces(cluster_id, members)
        db.upsert_cluster(cluster_id)
    
    # Build relationships
    print("Building relationships...")
    db.build_relationships_from_photos()
    
    print("Created {} face clusters".format(len(clusters)))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--detect":
            fast_face_detection()
        elif sys.argv[1] == "--cluster":
            fast_face_clustering()
        elif sys.argv[1] == "--all":
            fast_face_detection()
            fast_face_clustering()
    else:
        print("Usage: python fast_clustering.py [--detect|--cluster|--all]")