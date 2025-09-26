#!/usr/bin/env python3
"""Test the multiple object search with clean output"""

import requests
import json

API_BASE = "http://localhost:8000"

def test_search(query, description):
    """Test a search query and display clean results"""
    print(f"\n🔍 {description}")
    print(f"Query: '{query}'")
    print("-" * 50)
    
    try:
        response = requests.post(
            f"{API_BASE}/api/search",
            json={"query": query}
        )
        
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            method = data.get('method', 'unknown')
            
            print(f"Results: {len(results)} photos found")
            print(f"Method: {method}")
            
            if results:
                print("\nPhoto details:")
                for i, photo in enumerate(results[:3], 1):  # Show first 3
                    filename = photo.get('filename', 'Unknown')
                    objects = photo.get('objects', [])
                    
                    # Parse objects if they're JSON strings
                    if objects and isinstance(objects[0], str):
                        try:
                            parsed_objects = []
                            for obj_str in objects:
                                if obj_str.startswith('['):
                                    obj_list = json.loads(obj_str)
                                    for obj in obj_list:
                                        if isinstance(obj, dict) and 'class' in obj:
                                            parsed_objects.append(obj['class'])
                                else:
                                    parsed_objects.extend(obj_str.split(', '))
                            objects = list(set(parsed_objects))  # Remove duplicates
                        except:
                            pass
                    
                    print(f"  {i}. {filename}")
                    if objects:
                        print(f"     Objects: {', '.join(objects)}")
                    
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"Error: {e}")

def main():
    print("🚀 Testing Multiple Object Search")
    print("=" * 60)
    
    # Test various search queries
    tests = [
        ("dog", "Single object search"),
        ("person", "Single object search - person"),
        ("car", "Single object search - car"),
        ("dog and person", "Multiple objects - intersection"),
        ("person with car", "Multiple objects - intersection"),
        ("person and car", "Multiple objects - intersection"),
        ("cup and person", "Multiple objects - intersection")
    ]
    
    for query, description in tests:
        test_search(query, description)
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    main()