#!/usr/bin/env python3

"""
Performance test for optimized photo gallery loading
Tests the new pagination and performance improvements
"""

import asyncio
import httpx
import time
import sys
import os

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from api_helpers import APIHelpers

def test_pagination_performance():
    """Test the pagination performance improvements"""
    print("🧪 Testing pagination performance...")
    
    # Initialize API Helper
    helper = APIHelpers()
    
    # Test 1: Original get_recent_photos (if it exists)
    print("\n📊 Testing get_recent_photos with different limits:")
    
    # Test small batches (optimal for pagination)
    limits = [10, 20, 50, 100]
    
    for limit in limits:
        start_time = time.time()
        photos = helper.get_recent_photos(limit=limit, offset=0)
        end_time = time.time()
        
        print(f"  Limit {limit:3d}: {len(photos):3d} photos in {(end_time - start_time)*1000:.1f}ms")
    
    # Test pagination offsets
    print("\n📄 Testing pagination offsets:")
    
    offsets = [0, 20, 40, 60, 80]
    limit = 20
    
    for offset in offsets:
        start_time = time.time()
        photos = helper.get_recent_photos(limit=limit, offset=offset)
        end_time = time.time()
        
        print(f"  Offset {offset:3d}: {len(photos):3d} photos in {(end_time - start_time)*1000:.1f}ms")
    
    # Test total count
    print("\n🔢 Testing photo count:")
    start_time = time.time()
    total = helper.get_photos_count()
    end_time = time.time()
    
    print(f"  Total photos: {total} in {(end_time - start_time)*1000:.1f}ms")

async def test_api_performance():
    """Test the API endpoint performance"""
    print("\n🌐 Testing API endpoint performance...")
    
    base_url = "http://localhost:8000"
    
    async with httpx.AsyncClient() as client:
        # Test API pagination
        limits = [10, 20, 50]
        
        for limit in limits:
            start_time = time.time()
            try:
                response = await client.get(f"{base_url}/api/photos?limit={limit}&offset=0", timeout=10.0)
                if response.status_code == 200:
                    data = response.json()
                    photo_count = len(data.get('results', []))
                    pagination = data.get('pagination', {})
                    end_time = time.time()
                    
                    print(f"  API Limit {limit:2d}: {photo_count:3d} photos in {(end_time - start_time)*1000:.1f}ms")
                    print(f"    Total: {pagination.get('total', 'N/A')}, Has more: {pagination.get('has_more', 'N/A')}")
                else:
                    print(f"  API Limit {limit:2d}: Error {response.status_code}")
            except Exception as e:
                print(f"  API Limit {limit:2d}: Connection error - {e}")

def test_object_parsing_performance():
    """Test the optimized object parsing performance"""
    print("\n🏷️  Testing object parsing performance...")
    
    helper = APIHelpers()
    
    # Get a sample of photos with objects
    photos = helper.get_recent_photos(limit=50, offset=0)
    photos_with_objects = [p for p in photos if p.get('objects')]
    
    if photos_with_objects:
        print(f"  Found {len(photos_with_objects)} photos with objects")
        
        # Test object parsing performance
        start_time = time.time()
        total_objects = 0
        for photo in photos_with_objects:
            objects = photo.get('objects', [])
            total_objects += len(objects)
        end_time = time.time()
        
        print(f"  Parsed {total_objects} objects in {(end_time - start_time)*1000:.1f}ms")
        print(f"  Average: {(end_time - start_time)*1000/len(photos_with_objects):.2f}ms per photo")
    else:
        print("  No photos with objects found")

def main():
    """Run all performance tests"""
    print("🚀 Gallery Performance Test Suite")
    print("=" * 50)
    
    # Test database performance
    test_pagination_performance()
    
    # Test object parsing
    test_object_parsing_performance()
    
    # Test API performance (requires server to be running)
    print("\n⚠️  API tests require the server to be running on http://localhost:8000")
    try:
        asyncio.run(test_api_performance())
    except KeyboardInterrupt:
        print("\n⏹️  API tests skipped")
    
    print("\n✅ Performance tests completed!")

if __name__ == "__main__":
    main()