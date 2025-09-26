#!/usr/bin/env python3
"""Debug the full search flow"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fast_object_search import FastObjectSearch

def debug_search_flow():
    """Debug the complete search flow"""
    
    # Test 1: Direct FastObjectSearch
    print("=== TEST 1: Direct FastObjectSearch ===")
    fs = FastObjectSearch()
    results = fs.search_by_object('car', 5)
    print(f"FastObjectSearch found: {len(results)} results")
    
    for i, result in enumerate(results[:2], 1):
        print(f"  {i}. Path: {result.get('path', 'NO PATH')}")
        print(f"     ID: {result.get('id', 'NO ID')}")
        print(f"     Objects: {result.get('objects', 'NO OBJECTS')}")
        print(f"     Similarity: {result.get('similarity', 'NO SIMILARITY')}")
        
    # Test 2: Check if results have all required fields
    print("\n=== TEST 2: Result Field Analysis ===")
    if results:
        sample = results[0]
        required_fields = ['id', 'path', 'similarity', 'objects']
        
        for field in required_fields:
            value = sample.get(field)
            print(f"  {field}: {value} (type: {type(value)})")
        
        # Test result conversion like in API
        print("\n=== TEST 3: API Conversion Test ===")
        
        # This is similar to what happens in the API
        photo_responses = []
        for result in results:
            # Parse objects if it's a string
            objects = result.get("objects", [])
            if isinstance(objects, str):
                objects = [obj.strip() for obj in objects.split(",") if obj.strip()]
            
            print(f"  Original objects: {result.get('objects', 'NONE')}")
            print(f"  Parsed objects: {objects}")
            
            # Mock PhotoResponse creation
            photo_response = {
                "id": str(result.get("id", "")),
                "filename": os.path.basename(result.get("path", "")),
                "path": result.get("path", ""),
                "similarity_score": result.get("similarity", 0.0),
                "objects": objects,
                "timestamp": result.get("timestamp")
            }
            photo_responses.append(photo_response)
            break  # Just check first one
        
        print(f"  Converted result count: {len(photo_responses)}")
        if photo_responses:
            print(f"  Sample converted result: {photo_responses[0]}")

if __name__ == "__main__":
    debug_search_flow()