#!/usr/bin/env python3

from fast_object_search import FastObjectSearch

print("🔍 Testing FastObjectSearch directly")
print("=" * 50)

try:
    searcher = FastObjectSearch()
    
    print("1. Testing single object search: 'dog'")
    results = searcher.search_by_object("dog", 5)
    print(f"   Found {len(results)} photos")
    
    if results:
        for i, result in enumerate(results[:3], 1):
            filename = result['path'].split('/')[-1]
            print(f"   {i}. {filename} - Objects: {result.get('objects', 'N/A')}")
    
    print("\n2. Testing single object search: 'person'")
    results = searcher.search_by_object("person", 5)
    print(f"   Found {len(results)} photos")
    
    if results:
        for i, result in enumerate(results[:3], 1):
            filename = result['path'].split('/')[-1]
            print(f"   {i}. {filename} - Objects: {result.get('objects', 'N/A')}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ FastObjectSearch test completed!")