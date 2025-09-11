#!/usr/bin/env python3
"""
API Test Script

Test the FastAPI backend endpoints to ensure they're working correctly.
"""

import requests
import json
import time
import sys
from typing import Dict, Any

class APITester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_connection(self) -> bool:
        """Test basic API connectivity"""
        try:
            response = self.session.get(f"{self.base_url}/")
            if response.status_code == 200:
                print("✅ API connection successful")
                return True
            else:
                print(f"❌ API connection failed: {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print("❌ Could not connect to API server")
            print("💡 Make sure the server is running: python start_api.py")
            return False
    
    def test_status(self) -> bool:
        """Test API status endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/api/status")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ API Status: {data.get('status')}")
                print(f"📊 Photos indexed: {data.get('photos_indexed', 0)}")
                return True
            else:
                print(f"❌ Status check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Status check error: {e}")
            return False
    
    def test_stats(self) -> bool:
        """Test stats endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/api/stats")
            if response.status_code == 200:
                data = response.json()
                print("✅ Stats retrieved:")
                print(f"   📸 Photos: {data.get('total_photos', 0)}")
                print(f"   👥 Faces: {data.get('total_faces', 0)}")
                print(f"   🔗 Clusters: {data.get('total_clusters', 0)}")
                print(f"   💕 Relationships: {data.get('total_relationships', 0)}")
                print(f"   👨‍👩‍👧‍👦 Groups: {data.get('total_groups', 0)}")
                print(f"   💾 DB Size: {data.get('database_size_mb', 0):.2f} MB")
                return True
            else:
                print(f"❌ Stats failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Stats error: {e}")
            return False
    
    def test_search(self, query: str = "flower") -> bool:
        """Test search endpoint"""
        try:
            payload = {
                "query": query,
                "limit": 5
            }
            
            response = self.session.post(
                f"{self.base_url}/api/search",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                print(f"✅ Search for '{query}' returned {len(results)} results")
                
                if results:
                    print("   Sample results:")
                    for i, result in enumerate(results[:3]):
                        print(f"   {i+1}. {result.get('filename')} (score: {result.get('similarity_score', 0):.3f})")
                
                return True
            else:
                print(f"❌ Search failed: {response.status_code}")
                if response.text:
                    print(f"   Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Search error: {e}")
            return False
    
    def test_faces(self) -> bool:
        """Test face clusters endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/api/faces/clusters")
            if response.status_code == 200:
                data = response.json()
                clusters = data.get("clusters", [])
                print(f"✅ Found {len(clusters)} face clusters")
                
                if clusters:
                    print("   Sample clusters:")
                    for i, cluster in enumerate(clusters[:3]):
                        label = cluster.get("label") or "Unlabeled"
                        count = cluster.get("photo_count", 0)
                        print(f"   {i+1}. {cluster.get('cluster_id')}: {label} ({count} photos)")
                
                return True
            else:
                print(f"❌ Face clusters failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Face clusters error: {e}")
            return False
    
    def test_groups(self) -> bool:
        """Test groups endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/api/groups")
            if response.status_code == 200:
                data = response.json()
                groups = data.get("groups", [])
                print(f"✅ Found {len(groups)} groups")
                
                if groups:
                    print("   Groups:")
                    for group in groups:
                        name = group.get("group_name")
                        count = group.get("member_count", 0)
                        print(f"   - {name}: {count} members")
                
                return True
            else:
                print(f"❌ Groups failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Groups error: {e}")
            return False
    
    def test_relationships(self) -> bool:
        """Test relationships endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/api/relationships")
            if response.status_code == 200:
                data = response.json()
                relationships = data.get("relationships", [])
                print(f"✅ Found {len(relationships)} relationships")
                
                if relationships:
                    print("   Sample relationships:")
                    for i, rel in enumerate(relationships[:5]):
                        person1 = rel.get("person1_label") or rel.get("person1_cluster")
                        person2 = rel.get("person2_label") or rel.get("person2_cluster")
                        rel_type = rel.get("relationship_type")
                        confidence = rel.get("confidence", 0)
                        print(f"   {i+1}. {person1} ↔ {person2}: {rel_type} ({confidence:.1%})")
                
                return True
            else:
                print(f"❌ Relationships failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Relationships error: {e}")
            return False
    
    def run_all_tests(self) -> bool:
        """Run all API tests"""
        print("🧪 Running API Tests...")
        print("=" * 50)
        
        tests = [
            ("Connection", self.test_connection),
            ("Status", self.test_status),
            ("Stats", self.test_stats),
            ("Search", self.test_search),
            ("Face Clusters", self.test_faces),
            ("Groups", self.test_groups),
            ("Relationships", self.test_relationships)
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            print(f"\n🔍 Testing {test_name}...")
            try:
                if test_func():
                    passed += 1
                else:
                    print(f"❌ {test_name} test failed")
            except Exception as e:
                print(f"❌ {test_name} test error: {e}")
        
        print("\n" + "=" * 50)
        print(f"🏁 Test Results: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All tests passed! API is working correctly.")
            return True
        else:
            print("⚠️  Some tests failed. Check the API server and database.")
            return False

def main():
    """Main test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the AI Photo Search API")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--test", choices=["connection", "status", "stats", "search", "faces", "groups", "relationships"], help="Run specific test")
    parser.add_argument("--query", default="flower", help="Search query for search test")
    
    args = parser.parse_args()
    
    tester = APITester(args.url)
    
    if args.test:
        # Run specific test
        test_methods = {
            "connection": tester.test_connection,
            "status": tester.test_status,
            "stats": tester.test_stats,
            "search": lambda: tester.test_search(args.query),
            "faces": tester.test_faces,
            "groups": tester.test_groups,
            "relationships": tester.test_relationships
        }
        
        if args.test in test_methods:
            success = test_methods[args.test]()
            sys.exit(0 if success else 1)
        else:
            print(f"Unknown test: {args.test}")
            sys.exit(1)
    else:
        # Run all tests
        success = tester.run_all_tests()
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
