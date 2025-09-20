import pytest
from fastapi.testclient import TestClient
from api_server import app

client = TestClient(app)

def test_status():
    response = client.get("/api/status")
    assert response.status_code == 200
    assert "status" in response.json()

# ...existing code...

def test_stats():
    response = client.get("/api/stats")
    assert response.status_code == 200
    # Match actual keys returned by your endpoint
    assert "total_clusters" in response.json()
    assert "total_faces" in response.json()
    assert "total_groups" in response.json()

# ...existing code...

def test_search_photos():
    payload = {
        "query": "flower",
        "limit": 5
    }
    response = client.post("/api/search", json=payload)
    assert response.status_code == 200
    assert "results" in response.json()

def test_get_photo_by_id():
    # Assumes at least one photo exists with id=1
    response = client.get("/api/photos/1")
    # Accept 404 if not found, but should not error
    assert response.status_code in [200, 404]

def test_index_photos():
    payload = {
        "directory": "sample_photos",
        "recursive": True
    }
    response = client.post("/api/index", json=payload)
    assert response.status_code == 200
    assert "task_id" in response.json()

def test_list_face_clusters():
    response = client.get("/api/faces/clusters")
    assert response.status_code == 200
    assert "clusters" in response.json()

def test_label_face_cluster():
    payload = {
        "cluster_id": "cluster_1",
        "name": "Alice"
    }
    response = client.post("/api/faces/clusters/cluster_1/label", json=payload)
    # Accept 404 if cluster doesn't exist
    assert response.status_code in [200, 404]

def test_list_groups():
    response = client.get("/api/groups")
    assert response.status_code == 200
    assert "groups" in response.json()

def test_create_group():
    payload = {
        "group_name": "family",
        "cluster_ids": ["cluster_1", "cluster_2"]
    }
    response = client.post("/api/groups", json=payload)
    assert response.status_code == 200
    assert "success" in response.json()

def test_list_relationships():
    response = client.get("/api/relationships")
    assert response.status_code == 200
    assert "relationships" in response.json()

def test_get_relationships_for_person():
    # Assumes at least one cluster exists with id=cluster_1
    response = client.get("/api/relationships/cluster_1")
    # Accept 404 if not found
    assert response.status_code in [200, 404]

def test_start_face_clustering():
    response = client.post("/api/faces/cluster")
    assert response.status_code == 200
    assert "task_id" in response.json()

def test_build_relationships():
    response = client.post("/api/relationships/build")
    assert response.status_code == 200
    assert "task_id" in response.json()

def test_get_task_status():
    # This test assumes a task_id "test_task" exists; adjust as needed
    response = client.get("/api/tasks/test_task")
    assert response.status_code in [200, 404]

# Optional: Test root endpoint
def test_root():
    response = client.get("/")
    assert response.status_code == 200

# Optional: Test image serving (if images exist)
def test_image_serving():
    response = client.get("/images/sample.jpg")
    assert response.status_code in [200, 404]

# Run with: pytest test_api.py
