#!/usr/bin/env python3
"""Test fast object search directly"""

from fast_object_search import FastObjectSearch

def test_fast_search():
    fs = FastObjectSearch()
    
    # Test single object searches
    objects_to_test = ['car', 'person', 'dog', 'cup']
    
    for obj in objects_to_test:
        try:
            results = fs.search_by_object(obj, 5)
            print(f"'{obj}': Found {len(results)} results")
            for i, result in enumerate(results[:3], 1):
                filename = result.get('filename', result.get('path', '').split('/')[-1] if 'path' in result else 'unknown')
                print(f"  {i}. {filename}")
        except Exception as e:
            print(f"'{obj}': Error - {e}")
        print()

if __name__ == "__main__":
    test_fast_search()