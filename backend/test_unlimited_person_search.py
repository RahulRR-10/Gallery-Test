#!/usr/bin/env python3

from lightweight_person_search import LightweightPersonSearch
import logging

# Set up logging to see the results
logging.basicConfig(level=logging.INFO)

print("🔍 Testing Unlimited Person Search")
print("=" * 50)

# Test the lightweight person search
search = LightweightPersonSearch()

# Test with Rahul (should return all 64 photos, not just 10)
print("Searching for Rahul...")
results = search.search_person_photos("Rahul")

print(f"📊 Results: {len(results)} photos found")
print(f"🎯 Expected: Around 64 photos (all photos with Rahul)")

# Show first few results
for i, result in enumerate(results[:5], 1):
    filename = result['path'].split('/')[-1]
    print(f"  {i}. {filename} (similarity: {result['similarity']})")

if len(results) > 5:
    print(f"  ... and {len(results) - 5} more photos")

print("\n✅ Test completed!")