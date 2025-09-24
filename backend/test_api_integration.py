#!/usr/bin/env python3
"""
Test the API integration with intelligent query parsing
"""

import requests
import json

def test_api_integration():
    test_queries = [
        {'query': 'amisha'},
        {'query': 'amisha cake'}, 
        {'query': 'akshara party'},
        {'query': 'john doe meeting'},
        {'query': 'beach sunset'},  # Object only
    ]

    base_url = 'http://localhost:8000'

    for test in test_queries:
        try:
            response = requests.post(f'{base_url}/api/search', json=test, timeout=5)
            if response.status_code == 200:
                result = response.json()
                print(f"Query: '{test['query']}'")
                print(f"  Method: {result.get('search_method', 'unknown')}")
                print(f"  Results: {result.get('total', 0)}")
                if 'message' in result:
                    print(f"  Message: {result['message']}")
                print()
            else:
                print(f"Query: '{test['query']}' - HTTP {response.status_code}")
                print(f"  Response: {response.text}")
        except requests.exceptions.ConnectionError:
            print('❌ API server not running. Please start with:')
            print('   python -m uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload')
            break
        except Exception as e:
            print(f"❌ Error testing '{test['query']}': {e}")

if __name__ == "__main__":
    test_api_integration()