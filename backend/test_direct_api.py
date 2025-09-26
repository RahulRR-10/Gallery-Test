#!/usr/bin/env python3
"""Direct API test without server"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api_server import SearchRequest
import asyncio

async def test_search_logic():
    """Test the search logic directly"""
    from api_server import search_photos
    
    # Test single object search
    request = SearchRequest(query="car", limit=10)
    print(f"Testing query: {request.query}")
    
    try:
        result = await search_photos(request)
        print(f"Results: {len(result['results'])}")
        print(f"Method: {result['search_method']}")
        
        # Show first few results
        for i, photo in enumerate(result['results'][:3], 1):
            print(f"  {i}. {photo.filename}")
            print(f"     Objects: {photo.objects}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_search_logic())