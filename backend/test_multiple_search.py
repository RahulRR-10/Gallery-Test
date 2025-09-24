#!/usr/bin/env python3

import sqlite3
from intelligent_query_parser import IntelligentQueryParser

def test_multiple_people_search():
    # Check existing cluster labels
    conn = sqlite3.connect('photos.db')
    cursor = conn.cursor()
    cursor.execute('SELECT cluster_id, label, num_faces FROM face_clusters WHERE label IS NOT NULL AND label != "" ORDER BY num_faces DESC')
    clusters = cursor.fetchall()
    print('Existing cluster labels:')
    for cluster_id, label, num_faces in clusters:
        print(f'  {cluster_id}: "{label}" ({num_faces} faces)')

    # Test parsing with real names
    parser = IntelligentQueryParser()
    queries = [
        'rahul and saarthi',
        'rahul saarthi',
        'photos with rahul and saarthi',
        'show me rahul and saarthi together'
    ]
    
    for query in queries:
        parsed = parser.parse_query(query)
        print(f'\nParsing query: "{query}"')
        print(f'Person labels found: {parsed.person_labels}')
        print(f'Object terms: {parsed.object_terms}')
        print(f'Original query: {parsed.original_query}')

    conn.close()

if __name__ == "__main__":
    test_multiple_people_search()