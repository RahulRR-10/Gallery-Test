#!/usr/bin/env python3
"""Debug the search parsing logic"""

def test_query_parsing():
    from intelligent_query_parser import IntelligentQueryParser
    import os
    
    # Initialize parser
    db_path = "photos.db"
    parser = IntelligentQueryParser(db_path)
    
    # Test various queries
    queries = ['car', 'person', 'dog', 'cup', 'person and car', 'dog and person']
    
    for query in queries:
        print(f"\n🔍 Testing query: '{query}'")
        print("-" * 40)
        
        try:
            parsed = parser.parse_query(query)
            
            print(f"Person labels: {parsed.person_labels}")
            print(f"Object terms: {parsed.object_terms}")
            print(f"Time expressions: {parsed.time_expressions}")
            print(f"Has person labels: {bool(parsed.person_labels)}")
            print(f"Has object terms: {bool(parsed.object_terms)}")
            
            # Determine expected search path
            if parsed.person_labels and not parsed.object_terms:
                print("➜ Expected path: Person-only search")
            elif parsed.person_labels and parsed.object_terms:
                print("➜ Expected path: Person + Object search")
            elif parsed.object_terms:
                print("➜ Expected path: Object-only search")
            else:
                print("➜ Expected path: Contextual search")
                
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_query_parsing()