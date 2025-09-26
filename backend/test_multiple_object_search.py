#!/usr/bin/env python3

import requests
import json

print("🔍 Testing Multiple Object Search")
print("=" * 50)

# Test single object search
print("\n1. Testing single object: 'dog'")
response = requests.post("http://localhost:8000/api/search", 
                        json={"query": "dog", "limit": 5})
if response.status_code == 200:
    data = response.json()
    print(f"   Results: {data['total']} photos")
    print(f"   Method: {data['search_method']}")
else:
    print(f"   Error: {response.status_code}")

# Test multiple object search
print("\n2. Testing multiple objects: 'dog and person'")
response = requests.post("http://localhost:8000/api/search", 
                        json={"query": "dog and person", "limit": 5})
if response.status_code == 200:
    data = response.json()
    print(f"   Results: {data['total']} photos")
    print(f"   Method: {data['search_method']}")
    if data['results']:
        print("   Sample results:")
        for i, result in enumerate(data['results'][:3], 1):
            print(f"     {i}. {result['filename']} - Objects: {result['objects']}")
else:
    print(f"   Error: {response.status_code}")

# Test multiple object search with different query
print("\n3. Testing multiple objects: 'person with car'")
response = requests.post("http://localhost:8000/api/search", 
                        json={"query": "person with car", "limit": 5})
if response.status_code == 200:
    data = response.json()
    print(f"   Results: {data['total']} photos")
    print(f"   Method: {data['search_method']}")
    if data['results']:
        print("   Sample results:")
        for i, result in enumerate(data['results'][:3], 1):
            print(f"     {i}. {result['filename']} - Objects: {result['objects']}")
else:
    print(f"   Error: {response.status_code}")

print("\n✅ Multiple object search test completed!")