"""
Relationship Mapping Module - Phase 1 Implementation
Building person relationship graphs from existing face clustering data

PRIVACY & USER CONSENT: This module processes biometric face data and runs entirely on-device.
- Requires explicit user opt-in before enabling relationship mapping
- Stores all relationship data locally only; never uploads biometric data
- Provides functionality to delete person data and undo automatic groupings
"""

import sqlite3
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set
import json
import time
from collections import defaultdict, Counter
from photo_database import PhotoDatabase
import os
from datetime import datetime, timedelta

class RelationshipMapper:
    """
    Builds and analyzes relationship graphs from existing face clustering data
    """
    
    def __init__(self, db_path: str = "photos.db"):
        """
        Initialize relationship mapper
        
        Args:
            db_path: Path to the photo database
        """
        self.db_path = db_path
        self.db = PhotoDatabase(db_path)
        
    def build_cooccurrence_graph(self) -> nx.Graph:
        """
        Build co-occurrence graph from existing face cluster data
        
        Returns:
            NetworkX Graph with person nodes and co-occurrence edges
        """
        print("🔗 Building co-occurrence graph from existing face clusters...")
        
        # Create the graph
        graph = nx.Graph()
        
        # Get all photos with faces and their cluster assignments
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Query: Get all photos that have clustered faces
        cursor.execute('''
            SELECT DISTINCT f.photo_id, f.cluster_id, p.timestamp, p.exif_timestamp, p.path
            FROM faces f
            JOIN photos p ON f.photo_id = p.id
            WHERE f.cluster_id IS NOT NULL AND f.cluster_id != ""
            ORDER BY f.photo_id
        ''')
        
        face_data = cursor.fetchall()
        
        if not face_data:
            print("⚠️ No clustered faces found. Run --cluster-faces first.")
            conn.close()
            return graph
        
        # Group by photo to find co-occurrences
        photo_clusters = defaultdict(set)
        photo_timestamps = {}
        photo_paths = {}
        
        for photo_id, cluster_id, timestamp, exif_timestamp, path in face_data:
            photo_clusters[photo_id].add(cluster_id)
            # Prefer EXIF timestamp if available
            photo_timestamps[photo_id] = exif_timestamp if exif_timestamp else timestamp
            photo_paths[photo_id] = path
        
        # Get cluster labels and create nodes
        cursor.execute('''
            SELECT cluster_id, label, num_faces
            FROM face_clusters
            WHERE cluster_id IS NOT NULL
        ''')
        
        cluster_info = cursor.fetchall()
        cluster_labels = {cluster_id: label for cluster_id, label, _ in cluster_info}
        cluster_face_counts = {cluster_id: num_faces for cluster_id, _, num_faces in cluster_info}
        
        # Add nodes to graph
        for cluster_id, label, num_faces in cluster_info:
            # Get representative embedding for this cluster
            cursor.execute('''
                SELECT embedding FROM faces 
                WHERE cluster_id = ? AND embedding IS NOT NULL 
                LIMIT 1
            ''', (cluster_id,))
            
            emb_result = cursor.fetchone()
            repr_embedding = None
            if emb_result:
                try:
                    repr_embedding = self.db.deserialize_embedding(emb_result[0])
                except:
                    repr_embedding = None
            
            # Count photos this person appears in
            n_photos = len([pid for pid, clusters in photo_clusters.items() 
                           if cluster_id in clusters])
            
            graph.add_node(cluster_id, 
                          label=label or f"Person_{cluster_id}",
                          n_photos=n_photos,
                          num_faces=num_faces,
                          repr_embedding=repr_embedding)
        
        # Build co-occurrence edges
        edge_weights = defaultdict(int)
        edge_last_seen = defaultdict(int)
        
        for photo_id, clusters in photo_clusters.items():
            clusters_list = list(clusters)
            timestamp = photo_timestamps[photo_id]
            
            # For each pair of people in this photo
            for i in range(len(clusters_list)):
                for j in range(i + 1, len(clusters_list)):
                    cluster_a, cluster_b = clusters_list[i], clusters_list[j]
                    
                    # Ensure consistent edge direction (alphabetical)
                    if cluster_a > cluster_b:
                        cluster_a, cluster_b = cluster_b, cluster_a
                    
                    edge_key = (cluster_a, cluster_b)
                    edge_weights[edge_key] += 1
                    edge_last_seen[edge_key] = max(edge_last_seen[edge_key], timestamp)
        
        # Add edges to graph
        for (cluster_a, cluster_b), weight in edge_weights.items():
            if cluster_a in graph.nodes and cluster_b in graph.nodes:
                # Normalize edge weight to avoid bias toward very active people
                n_photos_a = graph.nodes[cluster_a]['n_photos']
                n_photos_b = graph.nodes[cluster_b]['n_photos']
                normalized_weight = weight / min(n_photos_a, n_photos_b) if min(n_photos_a, n_photos_b) > 0 else 0
                
                graph.add_edge(cluster_a, cluster_b,
                              weight=weight,
                              normalized_weight=normalized_weight,
                              last_seen_timestamp=edge_last_seen[(cluster_a, cluster_b)])
        
        conn.close()
        
        print(f"✅ Built co-occurrence graph: {len(graph.nodes)} people, {len(graph.edges)} relationships")
        return graph
    
    def save_graph(self, graph: nx.Graph, path: str = "relationship_graph.json"):
        """
        Save graph to JSON file (GraphML has issues with numpy arrays)
        
        Args:
            graph: NetworkX graph to save
            path: Output file path
        """
        try:
            # Convert graph to JSON-serializable format
            graph_data = {
                'nodes': [],
                'edges': []
            }
            
            # Convert nodes
            for node_id, data in graph.nodes(data=True):
                node_data = dict(data)
                # Convert numpy array to list for JSON
                if 'repr_embedding' in node_data and node_data['repr_embedding'] is not None:
                    node_data['repr_embedding'] = node_data['repr_embedding'].tolist()
                
                graph_data['nodes'].append({
                    'id': node_id,
                    'data': node_data
                })
            
            # Convert edges
            for u, v, data in graph.edges(data=True):
                graph_data['edges'].append({
                    'source': u,
                    'target': v,
                    'data': dict(data)
                })
            
            # Save to JSON
            with open(path, 'w') as f:
                json.dump(graph_data, f, indent=2)
            
            print(f"💾 Saved relationship graph to: {path}")
            
        except Exception as e:
            print(f"❌ Error saving graph: {e}")
    
    def load_graph(self, path: str = "relationship_graph.json") -> Optional[nx.Graph]:
        """
        Load graph from JSON file
        
        Args:
            path: Input file path
            
        Returns:
            NetworkX graph or None if failed
        """
        try:
            if not os.path.exists(path):
                print(f"⚠️ Graph file not found: {path}")
                return None
            
            with open(path, 'r') as f:
                graph_data = json.load(f)
            
            # Reconstruct NetworkX graph
            graph = nx.Graph()
            
            # Add nodes
            for node_info in graph_data['nodes']:
                node_id = node_info['id']
                data = node_info['data']
                
                # Convert embedding list back to numpy array
                if 'repr_embedding' in data and data['repr_embedding'] is not None:
                    try:
                        data['repr_embedding'] = np.array(data['repr_embedding'], dtype=np.float32)
                    except:
                        data['repr_embedding'] = None
                
                graph.add_node(node_id, **data)
            
            # Add edges
            for edge_info in graph_data['edges']:
                source = edge_info['source']
                target = edge_info['target']
                data = edge_info['data']
                graph.add_edge(source, target, **data)
            
            print(f"📂 Loaded relationship graph: {len(graph.nodes)} people, {len(graph.edges)} relationships")
            return graph
            
        except Exception as e:
            print(f"❌ Error loading graph: {e}")
            return None
    
    def get_graph_stats(self, graph: nx.Graph) -> Dict:
        """
        Get statistics about the relationship graph
        
        Args:
            graph: NetworkX graph
            
        Returns:
            Dictionary of graph statistics
        """
        if not graph.nodes:
            return {"error": "Empty graph"}
        
        # Calculate average connections
        if len(graph.nodes) == 0:
            avg_connections = 0
        else:
            # Each edge contributes 2 to total degree (one for each endpoint)
            avg_connections = (2 * len(graph.edges)) / len(graph.nodes)
        
        stats = {
            "total_people": len(graph.nodes),
            "total_relationships": len(graph.edges),
            "avg_connections": avg_connections,
            "labeled_people": sum(1 for _, data in graph.nodes(data=True) 
                                if data.get('label') and not data['label'].startswith('Person_')),
            "total_photos_represented": sum(data.get('n_photos', 0) for _, data in graph.nodes(data=True)),
            "total_events_represented": sum(data.get('n_events', 0) for _, data in graph.nodes(data=True))
        }
        
        # Find most connected people by counting edges
        node_degrees = {}
        for edge in graph.edges():
            node_a, node_b = edge
            node_degrees[node_a] = node_degrees.get(node_a, 0) + 1
            node_degrees[node_b] = node_degrees.get(node_b, 0) + 1
        
        if node_degrees:
            most_connected = max(node_degrees.items(), key=lambda x: x[1])
            stats["most_connected_person"] = {
                "cluster_id": most_connected[0],
                "label": graph.nodes[most_connected[0]].get('label', 'Unlabeled'),
                "connections": most_connected[1]
            }
        
        # Find strongest relationship
        if graph.edges:
            strongest_edge = max(graph.edges(data=True), key=lambda x: x[2].get('weight', 0))
            cluster_a, cluster_b, edge_data = strongest_edge
            stats["strongest_relationship"] = {
                "person_a": graph.nodes[cluster_a].get('label', f'Person_{cluster_a}'),
                "person_b": graph.nodes[cluster_b].get('label', f'Person_{cluster_b}'),
                "co_occurrences": edge_data.get('weight', 0),
                "normalized_weight": edge_data.get('normalized_weight', 0)
            }
        
        return stats
    
    def update_relationships_table(self, graph: nx.Graph):
        """
        Update the relationships table in the database with graph data
        
        Args:
            graph: NetworkX graph with relationship data
        """
        print("💾 Updating relationships table...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Clear existing relationships
        cursor.execute('DELETE FROM relationships')
        
        # Insert new relationships from graph
        for cluster_a, cluster_b, edge_data in graph.edges(data=True):
            weight = edge_data.get('weight', 0)
            normalized_weight = edge_data.get('normalized_weight', 0.0)
            
            cursor.execute('''
                INSERT INTO relationships (cluster_id_a, cluster_id_b, count, weight)
                VALUES (?, ?, ?, ?)
            ''', (cluster_a, cluster_b, weight, normalized_weight))
        
        conn.commit()
        conn.close()
        
        print(f"✅ Updated relationships table with {len(graph.edges)} relationships")
    
    def group_photos_into_events(self, window_hours: int = 48) -> Dict[str, List[str]]:
        """
        Group photos into events based on temporal proximity
        
        Args:
            window_hours: Photos within this time window belong to same event
            
        Returns:
            Dictionary mapping event_id -> list of photo_ids
        """
        print(f"📅 Grouping photos into events (window: {window_hours} hours)...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all photos with timestamps, preferring EXIF timestamps
        cursor.execute('''
            SELECT id, path, timestamp, exif_timestamp
            FROM photos
            ORDER BY COALESCE(exif_timestamp, timestamp)
        ''')
        
        photos = cursor.fetchall()
        conn.close()
        
        if not photos:
            print("⚠️ No photos found for event clustering")
            return {}
        
        # Sort photos by timestamp (prefer EXIF, fallback to file timestamp)
        photo_times = []
        for photo_id, path, timestamp, exif_timestamp in photos:
            effective_timestamp = exif_timestamp if exif_timestamp else timestamp
            photo_times.append((photo_id, effective_timestamp, path))
        
        photo_times.sort(key=lambda x: x[1])
        
        # Group photos into events based on time gaps
        events = {}
        current_event_id = None
        current_event_photos = []
        last_timestamp = None
        window_seconds = window_hours * 3600
        
        for photo_id, timestamp, path in photo_times:
            if timestamp is None:
                continue
                
            # Start new event if this is first photo or time gap is too large
            if (last_timestamp is None or 
                abs(timestamp - last_timestamp) > window_seconds):
                
                # Save previous event if it exists
                if current_event_id and current_event_photos:
                    events[current_event_id] = current_event_photos.copy()
                
                # Start new event
                current_event_id = f"event_{len(events) + 1}"
                current_event_photos = [photo_id]
            else:
                # Add to current event
                current_event_photos.append(photo_id)
            
            last_timestamp = timestamp
        
        # Don't forget the last event
        if current_event_id and current_event_photos:
            events[current_event_id] = current_event_photos
        
        # Log event statistics
        event_sizes = [len(photos) for photos in events.values()]
        avg_size = sum(event_sizes) / len(event_sizes) if event_sizes else 0
        
        print(f"✅ Created {len(events)} events from {len(photo_times)} photos")
        print(f"📊 Average event size: {avg_size:.1f} photos")
        print(f"📊 Largest event: {max(event_sizes) if event_sizes else 0} photos")
        
        return events
    
    def build_enhanced_cooccurrence_graph(self, window_hours: int = 48) -> nx.Graph:
        """
        Build co-occurrence graph enhanced with event context
        
        Args:
            window_hours: Time window for event clustering
            
        Returns:
            NetworkX Graph with event-aware relationship data
        """
        print("🔗 Building enhanced co-occurrence graph with event context...")
        
        # First group photos into events
        events = self.group_photos_into_events(window_hours)
        
        if not events:
            print("⚠️ No events found. Falling back to basic co-occurrence graph.")
            return self.build_cooccurrence_graph()
        
        # Create the graph
        graph = nx.Graph()
        
        # Get face cluster data
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Query: Get all photos with faces and their cluster assignments
        cursor.execute('''
            SELECT DISTINCT f.photo_id, f.cluster_id, p.timestamp, p.exif_timestamp, p.path
            FROM faces f
            JOIN photos p ON f.photo_id = p.id
            WHERE f.cluster_id IS NOT NULL AND f.cluster_id != ""
            ORDER BY f.photo_id
        ''')
        
        face_data = cursor.fetchall()
        
        if not face_data:
            print("⚠️ No clustered faces found.")
            conn.close()
            return graph
        
        # Group by photo to find co-occurrences
        photo_clusters = defaultdict(set)
        photo_timestamps = {}
        
        for photo_id, cluster_id, timestamp, exif_timestamp, path in face_data:
            photo_clusters[photo_id].add(cluster_id)
            photo_timestamps[photo_id] = exif_timestamp if exif_timestamp else timestamp
        
        # Create reverse mapping: photo_id -> event_id
        photo_to_event = {}
        for event_id, photo_ids in events.items():
            for photo_id in photo_ids:
                photo_to_event[photo_id] = event_id
        
        # Get cluster labels and create nodes
        cursor.execute('''
            SELECT cluster_id, label, num_faces
            FROM face_clusters
            WHERE cluster_id IS NOT NULL
        ''')
        
        cluster_info = cursor.fetchall()
        cluster_labels = {cluster_id: label for cluster_id, label, _ in cluster_info}
        
        # Add nodes to graph with event context
        for cluster_id, label, num_faces in cluster_info:
            # Get representative embedding
            cursor.execute('''
                SELECT embedding FROM faces 
                WHERE cluster_id = ? AND embedding IS NOT NULL 
                LIMIT 1
            ''', (cluster_id,))
            
            emb_result = cursor.fetchone()
            repr_embedding = None
            if emb_result:
                try:
                    repr_embedding = self.db.deserialize_embedding(emb_result[0])
                except:
                    repr_embedding = None
            
            # Count photos and events this person appears in
            person_photos = [pid for pid, clusters in photo_clusters.items() 
                           if cluster_id in clusters]
            person_events = set()
            for photo_id in person_photos:
                if photo_id in photo_to_event:
                    person_events.add(photo_to_event[photo_id])
            
            graph.add_node(cluster_id, 
                          label=label or f"Person_{cluster_id}",
                          n_photos=len(person_photos),
                          n_events=len(person_events),
                          num_faces=num_faces,
                          repr_embedding=repr_embedding)
        
        # Build enhanced co-occurrence edges with event context
        edge_weights = defaultdict(int)
        edge_events = defaultdict(set)
        edge_last_seen = defaultdict(int)
        
        for photo_id, clusters in photo_clusters.items():
            clusters_list = list(clusters)
            timestamp = photo_timestamps.get(photo_id, 0)
            event_id = photo_to_event.get(photo_id)
            
            # For each pair of people in this photo
            for i in range(len(clusters_list)):
                for j in range(i + 1, len(clusters_list)):
                    cluster_a, cluster_b = clusters_list[i], clusters_list[j]
                    
                    # Ensure consistent edge direction
                    if cluster_a > cluster_b:
                        cluster_a, cluster_b = cluster_b, cluster_a
                    
                    edge_key = (cluster_a, cluster_b)
                    edge_weights[edge_key] += 1
                    edge_last_seen[edge_key] = max(edge_last_seen[edge_key], timestamp)
                    
                    # Track which events they appear together in
                    if event_id:
                        edge_events[edge_key].add(event_id)
        
        # Add enhanced edges to graph
        for (cluster_a, cluster_b), weight in edge_weights.items():
            if cluster_a in graph.nodes and cluster_b in graph.nodes:
                # Calculate event-based metrics
                shared_events = len(edge_events.get((cluster_a, cluster_b), set()))
                
                n_events_a = graph.nodes[cluster_a]['n_events']
                n_events_b = graph.nodes[cluster_b]['n_events']
                
                # Event co-occurrence fraction (how often they appear together in events)
                min_events = min(n_events_a, n_events_b)
                event_cooccurrence_fraction = shared_events / min_events if min_events > 0 else 0
                
                # Normalized weight (same as before)
                n_photos_a = graph.nodes[cluster_a]['n_photos']
                n_photos_b = graph.nodes[cluster_b]['n_photos']
                normalized_weight = weight / min(n_photos_a, n_photos_b) if min(n_photos_a, n_photos_b) > 0 else 0
                
                graph.add_edge(cluster_a, cluster_b,
                              weight=weight,
                              normalized_weight=normalized_weight,
                              shared_events=shared_events,
                              event_cooccurrence_fraction=event_cooccurrence_fraction,
                              last_seen_timestamp=edge_last_seen[(cluster_a, cluster_b)])
        
        conn.close()
        
        print(f"✅ Built enhanced graph: {len(graph.nodes)} people, {len(graph.edges)} relationships")
        print(f"📅 Event context: {len(events)} events analyzed")
        
        return graph
    
    def detect_communities(self, graph: nx.Graph) -> Dict[str, int]:
        """
        Detect communities in the relationship graph using simple clustering
        
        Args:
            graph: NetworkX graph with relationship data
            
        Returns:
            Dictionary mapping cluster_id -> community_id
        """
        try:
            # Simple community detection based on connected components
            # For now, treat each connected component as a community
            communities = list(nx.connected_components(graph))
            
            # Create mapping from cluster_id to community_id
            cluster_to_community = {}
            for community_id, community in enumerate(communities):
                for cluster_id in community:
                    cluster_to_community[cluster_id] = community_id
            
            print(f"🏘️ Detected {len(communities)} communities (connected components)")
            return cluster_to_community
            
        except Exception as e:
            print(f"⚠️ Community detection failed: {e}")
            return {}
    
    def infer_relationship_type(self, graph: nx.Graph, cluster_a: str, cluster_b: str, 
                               edge_data: Dict, object_context: Optional[Dict] = None) -> Tuple[str, float]:
        """
        Infer relationship type based on co-occurrence patterns and context
        
        Args:
            graph: NetworkX graph
            cluster_a, cluster_b: The two people in the relationship
            edge_data: Edge data with co-occurrence statistics
            object_context: Optional object detection context
            
        Returns:
            Tuple of (relationship_type, confidence_score)
        """
        # Extract metrics
        weight = edge_data.get('weight', 0)
        shared_events = edge_data.get('shared_events', 0)
        event_cooccurrence_fraction = edge_data.get('event_cooccurrence_fraction', 0.0)
        
        # Get node data
        node_a = graph.nodes[cluster_a]
        node_b = graph.nodes[cluster_b]
        n_events_a = node_a.get('n_events', 0)
        n_events_b = node_b.get('n_events', 0)
        
        # Relationship inference heuristics
        confidence = 0.5  # Base confidence
        
        # Family heuristics: High co-occurrence across many different events
        if (event_cooccurrence_fraction >= 0.6 and 
            shared_events >= 3 and 
            weight >= 5):
            confidence = min(0.9, 0.6 + (event_cooccurrence_fraction - 0.6) * 0.75)
            return "family", confidence
        
        # Close friends: High co-occurrence but maybe fewer events than family
        elif (event_cooccurrence_fraction >= 0.4 and 
              shared_events >= 2 and 
              weight >= 3):
            confidence = min(0.8, 0.5 + (event_cooccurrence_fraction - 0.4) * 0.75)
            return "close_friend", confidence
        
        # Coworkers: Moderate co-occurrence with potential office context
        # This would benefit from object detection context (laptops, offices, etc.)
        elif (event_cooccurrence_fraction >= 0.3 and 
              shared_events >= 2 and 
              weight >= 2):
            # Could enhance this with object detection for office/work contexts
            if object_context and any(obj in ['laptop', 'computer', 'office'] 
                                    for obj in object_context.get('common_objects', [])):
                confidence = 0.7
                return "coworker", confidence
            else:
                confidence = 0.6
                return "friend", confidence
        
        # Acquaintances: Lower co-occurrence
        elif (shared_events >= 1 and weight >= 1):
            confidence = 0.4 + min(0.3, event_cooccurrence_fraction)
            return "acquaintance", confidence
        
        # Unknown: Very limited interaction
        else:
            return "unknown", 0.3
    
    def infer_all_relationships(self, graph: nx.Graph) -> Dict[Tuple[str, str], Tuple[str, float]]:
        """
        Infer relationship types for all edges in the graph
        
        Args:
            graph: NetworkX graph with relationship data
            
        Returns:
            Dictionary mapping (cluster_a, cluster_b) -> (relationship_type, confidence)
        """
        print("🧠 Inferring relationship types...")
        
        relationships = {}
        relationship_counts = defaultdict(int)
        
        for cluster_a, cluster_b, edge_data in graph.edges(data=True):
            # Infer relationship type
            rel_type, confidence = self.infer_relationship_type(
                graph, cluster_a, cluster_b, edge_data
            )
            
            relationships[(cluster_a, cluster_b)] = (rel_type, confidence)
            relationship_counts[rel_type] += 1
        
        # Log statistics
        print(f"✅ Inferred {len(relationships)} relationships:")
        for rel_type, count in relationship_counts.items():
            print(f"  {rel_type.replace('_', ' ').title()}: {count}")
        
        return relationships
    
    def save_relationships_to_db(self, relationships: Dict[Tuple[str, str], Tuple[str, float]]):
        """
        Save inferred relationships to database
        
        Args:
            relationships: Dictionary of relationship inferences
        """
        print("💾 Saving relationship inferences to database...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create relationships inference table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS relationship_inferences (
                cluster_id_a TEXT NOT NULL,
                cluster_id_b TEXT NOT NULL,
                inferred_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (cluster_id_a, cluster_id_b)
            )
        ''')
        
        # Clear existing inferences
        cursor.execute('DELETE FROM relationship_inferences')
        
        # Insert new inferences
        for (cluster_a, cluster_b), (rel_type, confidence) in relationships.items():
            cursor.execute('''
                INSERT INTO relationship_inferences 
                (cluster_id_a, cluster_id_b, inferred_type, confidence, created_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (cluster_a, cluster_b, rel_type, confidence, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
        
        print(f"✅ Saved {len(relationships)} relationship inferences to database")
    
    def get_relationship_summary(self, graph: nx.Graph, 
                                relationships: Dict[Tuple[str, str], Tuple[str, float]]) -> Dict:
        """
        Generate a comprehensive relationship summary
        
        Args:
            graph: NetworkX graph
            relationships: Inferred relationships
            
        Returns:
            Dictionary with relationship summary statistics
        """
        summary = {
            "total_relationships": len(relationships),
            "relationship_types": defaultdict(list),
            "high_confidence_relationships": [],
            "person_relationship_counts": defaultdict(lambda: defaultdict(int))
        }
        
        # Analyze relationships
        for (cluster_a, cluster_b), (rel_type, confidence) in relationships.items():
            label_a = graph.nodes[cluster_a].get('label', cluster_a)
            label_b = graph.nodes[cluster_b].get('label', cluster_b)
            
            # Group by relationship type
            summary["relationship_types"][rel_type].append({
                "person_a": label_a,
                "person_b": label_b,
                "confidence": confidence
            })
            
            # Track high confidence relationships
            if confidence >= 0.7:
                summary["high_confidence_relationships"].append({
                    "person_a": label_a,
                    "person_b": label_b,
                    "type": rel_type,
                    "confidence": confidence
                })
            
            # Count relationship types per person
            summary["person_relationship_counts"][label_a][rel_type] += 1
            summary["person_relationship_counts"][label_b][rel_type] += 1
        
        return summary


def infer_relationships_cli(db_path: str = "photos.db", event_window: int = 48):
    """
    CLI function to infer relationship types from enhanced graph data
    
    Args:
        db_path: Database path
        event_window: Time window for event clustering
    """
    mapper = RelationshipMapper(db_path)
    
    # Load or build enhanced graph
    print("🔗 Loading enhanced relationship graph...")
    graph = mapper.load_graph("enhanced_relationship_graph.json")
    
    if not graph or len(graph.nodes) == 0:
        print("⚠️ No enhanced relationship graph found. Building one...")
        graph = mapper.build_enhanced_cooccurrence_graph(event_window)
        if not graph or len(graph.nodes) == 0:
            print("❌ Failed to build relationship graph")
            return
    
    # Detect communities
    communities = mapper.detect_communities(graph)
    
    # Infer relationship types
    relationships = mapper.infer_all_relationships(graph)
    
    if not relationships:
        print("⚠️ No relationships found to infer")
        return None, None
    
    # Save to database
    mapper.save_relationships_to_db(relationships)
    
    # Generate and display summary
    summary = mapper.get_relationship_summary(graph, relationships)
    
    print("\n🧠 Relationship Inference Summary:")
    print("=" * 50)
    print(f"Total relationships analyzed: {summary['total_relationships']}")
    
    # Show relationship type breakdown
    print("\n📊 Relationship Types:")
    for rel_type, relationships_list in summary["relationship_types"].items():
        print(f"  {rel_type.replace('_', ' ').title()}: {len(relationships_list)}")
    
    # Show high confidence relationships
    if summary["high_confidence_relationships"]:
        print("\n🎯 High Confidence Relationships (70%+):")
        for rel in summary["high_confidence_relationships"]:
            print(f"  {rel['person_a']} ↔ {rel['person_b']}: {rel['type'].replace('_', ' ').title()} ({rel['confidence']:.1%})")
    
    # Show detailed breakdown by type
    print("\n📋 Detailed Relationship Analysis:")
    for rel_type, relationships_list in summary["relationship_types"].items():
        if relationships_list:
            print(f"\n{rel_type.replace('_', ' ').title()}:")
            for rel in relationships_list:
                print(f"  • {rel['person_a']} ↔ {rel['person_b']} ({rel['confidence']:.1%} confidence)")
    
    print(f"\n✅ Relationship inference complete! Results saved to database.")
    return relationships, summary


if __name__ == "__main__":
    # Example usage
    print("🚀 Relationship Mapping - Phase 2, Stage 3")
    print("Inferring relationship types from co-occurrence patterns...")
    
    # Infer relationships
    result = infer_relationships_cli()
    
    if result and result[0]:
        relationships, summary = result
        print("\n✅ Phase 2, Stage 3 Complete!")
        print("Next steps:")
        print("- Review relationship inferences with --infer-relationships")
        print("- Move to Stage 4: Enhanced search integration")
    else:
        print("\n⚠️ No relationships inferred. Make sure you have:")
        print("1. Built enhanced relationships with --enhanced-relationships")
        print("2. Clustered and labeled faces")
        print("3. Photos with temporal data")


def build_relationships_cli(db_path: str = "photos.db", save_graph: bool = True):
    """
    CLI function to build relationship graph
    
    Args:
        db_path: Database path
        save_graph: Whether to save graph to file
    """
    mapper = RelationshipMapper(db_path)
    
    # Build the graph
    graph = mapper.build_cooccurrence_graph()
    
    if not graph.nodes:
        print("❌ No relationship graph could be built")
        return
    
    # Show statistics
    stats = mapper.get_graph_stats(graph)
    print("\n📊 Relationship Graph Statistics:")
    print(f"👥 Total people: {stats['total_people']}")
    print(f"🔗 Total relationships: {stats['total_relationships']}")
    print(f"🏷️ Labeled people: {stats['labeled_people']}")
    print(f"📸 Photos represented: {stats['total_photos_represented']}")
    print(f"🌐 Average connections per person: {stats['avg_connections']:.1f}")
    
    if 'most_connected_person' in stats:
        mc = stats['most_connected_person']
        print(f"🌟 Most connected: {mc['label']} ({mc['connections']} connections)")
    
    if 'strongest_relationship' in stats:
        sr = stats['strongest_relationship']
        print(f"💪 Strongest relationship: {sr['person_a']} ↔ {sr['person_b']} ({sr['co_occurrences']} photos)")
    
    # Update database
    mapper.update_relationships_table(graph)
    
    # Save graph file
    if save_graph:
        mapper.save_graph(graph)
    
    return graph


if __name__ == "__main__":
    # Example usage
    print("🚀 Relationship Mapping - Phase 1, Stage 1")
    print("Building co-occurrence graph from existing face clusters...")
    
    # Build relationships
    graph = build_relationships_cli()
    
    if graph and len(graph.nodes) > 0:
        print("\n✅ Phase 1, Stage 1 Complete!")
        print("Next steps:")
        print("- Run --build-relationships in the main CLI")
        print("- Label more people with --label-person")
        print("- Move to Stage 2: Event clustering and relationship inference")
    else:
        print("\n⚠️ No relationships found. Make sure you have:")
        print("1. Indexed photos with --index")
        print("2. Clustered faces with --cluster-faces")
        print("3. Labeled some people with --label-person")
