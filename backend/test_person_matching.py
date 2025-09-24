#!/usr/bin/env python3
"""
Debug person label matching
"""

import sqlite3
import sys
sys.path.append('.')

from intelligent_query_parser import IntelligentQueryParser

def test_person_labels():
    # Check what person labels exist in database
    conn = sqlite3.connect('photos.db')
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT DISTINCT label 
        FROM face_clusters 
        WHERE label IS NOT NULL AND label != '' AND label != 'Unknown'
    """)
    
    labels = cursor.fetchall()
    print('🏷️ Existing person labels in database:')
    for label in labels:
        print(f'  - "{label[0]}"')
    
    conn.close()
    
    # Test the parser
    parser = IntelligentQueryParser('photos.db')
    
    # Test some queries
    test_queries = [
        "amisha",
        "Amisha", 
        "amisha cake",
        "Amisha birthday cake",
        "akshara party",
        "john doe meeting"
    ]
    
    print('\n🔍 Testing parser with different queries:')
    for query in test_queries:
        print(f'\nQuery: "{query}"')
        parsed = parser.parse_query(query)
        print(f'  Persons found: {parsed.person_labels}')
        print(f'  Objects: {parsed.object_terms}')
        print(f'  Time: {parsed.time_expressions}')
        
        # Test the find_person_matches function directly
        words = query.split()
        persons, remaining = parser.find_person_matches(words)
        print(f'  Direct person match: {persons} (remaining: {remaining})')

if __name__ == "__main__":
    test_person_labels()