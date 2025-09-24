#!/usr/bin/env python3
"""
Intelligent Query Parser
========================
Parses multi-word search queries and intelligently routes words to
person search, object search, and time filtering.
"""

import re
import sqlite3
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ParsedQuery:
    """Structure to hold parsed query components"""
    person_labels: List[str]           # Detected person names
    object_terms: List[str]            # Object search terms
    time_expressions: List[str]        # Time-related words
    original_query: str                # Original input
    confidence_scores: Dict[str, float] # Confidence for each detection

class IntelligentQueryParser:
    """Parse multi-word queries into person, object, and time components"""
    
    def __init__(self, db_path: str = "photos.db"):
        self.db_path = db_path
        
        # Time-related keywords and patterns
        self.time_keywords = {
            # Years
            r'\b(19|20)\d{2}\b',  # 1900-2099
            
            # Months
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b',
            r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b',
            
            # Days and relative time
            r'\b(today|yesterday|tomorrow|weekend|weekday)\b',
            r'\b(last|this|next)\s+(week|month|year|summer|winter|spring|fall|autumn)\b',
            r'\b(last|this|next)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
            
            # Special occasions
            r'\b(birthday|christmas|thanksgiving|halloween|easter|valentine|wedding|graduation|vacation|holiday)\b',
            r'\b(party|celebration|anniversary|reunion)\b',
            
            # Time periods
            r'\b(morning|afternoon|evening|night|dawn|dusk|sunset|sunrise)\b',
            r'\b(days?|weeks?|months?|years?)\s+ago\b',
            
            # Seasons
            r'\b(spring|summer|fall|autumn|winter)\b'
        }
        
        # Common object synonyms to help with matching
        self.object_synonyms = {
            'car': ['vehicle', 'automobile', 'auto'],
            'dog': ['puppy', 'canine', 'pet'],
            'cat': ['kitten', 'feline', 'pet'],
            'person': ['people', 'human', 'man', 'woman', 'child', 'kid'],
            'bicycle': ['bike', 'cycling'],
            'food': ['meal', 'dinner', 'lunch', 'breakfast', 'snack'],
            'cake': ['birthday', 'dessert', 'sweet'],
            'drink': ['beverage', 'bottle', 'cup', 'glass'],
            'phone': ['mobile', 'smartphone', 'cellphone'],
            'computer': ['laptop', 'pc', 'desktop']
        }
        
    def get_all_cluster_labels(self) -> List[str]:
        """Get all existing cluster labels (person names) from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT DISTINCT TRIM(label) 
                FROM face_clusters 
                WHERE label IS NOT NULL AND label != '' AND label != 'Unknown'
            """)
            
            labels = [row[0] for row in cursor.fetchall() if row[0]]
            conn.close()
            
            logger.debug(f"Found {len(labels)} cluster labels: {labels}")
            return labels
            
        except Exception as e:
            logger.error(f"Error getting cluster labels: {e}")
            return []
    
    def is_time_expression(self, word: str) -> bool:
        """Check if a word or phrase is time-related"""
        word_lower = word.lower().strip()
        
        # Check against time keyword patterns
        for pattern in self.time_keywords:
            if re.search(pattern, word_lower, re.IGNORECASE):
                return True
                
        # Check for numeric patterns that might be dates
        if re.match(r'^\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?$', word):  # MM/DD or MM/DD/YYYY
            return True
            
        return False
    
    def find_person_matches(self, words: List[str]) -> Tuple[List[str], List[str]]:
        """
        Find person names from word list with improved multi-word name handling
        Returns: (matched_persons, remaining_words)
        """
        cluster_labels = self.get_all_cluster_labels()
        if not cluster_labels:
            return [], words
            
        matched_persons = []
        remaining_words = []
        processed_indices = set()  # Track which words we've already matched
        
        # Create a case-insensitive lookup
        label_lookup = {label.lower(): label for label in cluster_labels}
        
        # First pass: exact matches
        for i, word in enumerate(words):
            if i in processed_indices:
                continue
                
            word_lower = word.lower().strip()
            
            # Exact match
            if word_lower in label_lookup:
                original_label = label_lookup[word_lower]
                matched_persons.append(original_label)
                processed_indices.add(i)
                logger.debug(f"Exact person match: '{word}' -> '{original_label}'")
        
        # Second pass: multi-word name matching (e.g., "john doe")
        for i, word1 in enumerate(words):
            if i in processed_indices:
                continue
                
            # Try combining with next word(s)
            for j in range(i + 1, min(i + 3, len(words))):  # Try up to 2 additional words
                if j in processed_indices:
                    continue
                    
                # Create multi-word phrase
                phrase_words = [words[k] for k in range(i, j + 1)]
                phrase = " ".join(phrase_words).lower()
                
                if phrase in label_lookup:
                    original_label = label_lookup[phrase]
                    matched_persons.append(original_label)
                    # Mark all words in this phrase as processed
                    for k in range(i, j + 1):
                        processed_indices.add(k)
                    logger.debug(f"Multi-word person match: '{' '.join(phrase_words)}' -> '{original_label}'")
                    break
        
        # Third pass: partial matches for remaining words
        for i, word in enumerate(words):
            if i in processed_indices:
                continue
                
            word_lower = word.lower().strip()
            
            # Partial match (for names like "john" matching "john doe")
            partial_matches = [label for label in cluster_labels 
                             if word_lower in label.lower() and len(word_lower) >= 3]  # Avoid very short matches
            
            if partial_matches:
                # Use the first/best match
                best_match = partial_matches[0]
                matched_persons.append(best_match)
                processed_indices.add(i)
                logger.debug(f"Partial person match: '{word}' -> '{best_match}'")
        
        # Add unprocessed words to remaining (avoid duplicates)
        for i, word in enumerate(words):
            if i not in processed_indices:
                remaining_words.append(word)
        
        return matched_persons, remaining_words
    
    def expand_object_synonyms(self, word: str) -> List[str]:
        """Expand a word to include synonyms for better object matching"""
        word_lower = word.lower().strip()
        expanded = [word]  # Always include original
        
        # Check if word is a synonym for any object
        for main_object, synonyms in self.object_synonyms.items():
            if word_lower == main_object or word_lower in synonyms:
                expanded.extend([main_object] + synonyms)
                break
                
        return list(set(expanded))  # Remove duplicates
    
    def parse_query(self, query: str) -> ParsedQuery:
        """
        Parse a multi-word query into components
        
        Args:
            query: Multi-word search query (e.g., "John birthday cake 2024")
            
        Returns:
            ParsedQuery with separated components
        """
        if not query or not query.strip():
            return ParsedQuery([], [], [], "", {})
            
        original_query = query.strip()
        
        # Split into words/phrases, preserving quoted strings
        words = []
        current_word = ""
        in_quotes = False
        
        for char in query:
            if char == '"' and not in_quotes:
                in_quotes = True
                if current_word.strip():
                    words.append(current_word.strip())
                    current_word = ""
            elif char == '"' and in_quotes:
                in_quotes = False
                if current_word.strip():
                    words.append(current_word.strip())
                    current_word = ""
            elif char == ' ' and not in_quotes:
                if current_word.strip():
                    words.append(current_word.strip())
                    current_word = ""
            else:
                current_word += char
                
        if current_word.strip():
            words.append(current_word.strip())
            
        logger.debug(f"Parsed words from '{original_query}': {words}")
        
        # Step 1: Find time expressions
        time_expressions = []
        non_time_words = []
        
        for word in words:
            if self.is_time_expression(word):
                time_expressions.append(word)
                logger.debug(f"Time expression detected: '{word}'")
            else:
                non_time_words.append(word)
        
        # Step 2: Find person names from remaining words
        person_labels, remaining_words = self.find_person_matches(non_time_words)
        
        # Step 3: Remaining words are object terms
        object_terms = remaining_words
        
        # Step 4: Calculate confidence scores
        confidence_scores = {}
        
        for person in person_labels:
            confidence_scores[f"person:{person}"] = 1.0  # High confidence for exact matches
            
        for obj in object_terms:
            confidence_scores[f"object:{obj}"] = 0.8  # Medium confidence for objects
            
        for time_expr in time_expressions:
            confidence_scores[f"time:{time_expr}"] = 0.9  # High confidence for time patterns
        
        result = ParsedQuery(
            person_labels=person_labels,
            object_terms=object_terms,
            time_expressions=time_expressions,
            original_query=original_query,
            confidence_scores=confidence_scores
        )
        
        logger.info(f"Query parsing result: {result}")
        return result
    
    def format_time_filter(self, time_expressions: List[str]) -> Optional[str]:
        """Convert time expressions list to a single time filter string"""
        if not time_expressions:
            return None
            
        # Join multiple time expressions
        return " ".join(time_expressions)

# Example usage and testing functions
def test_query_parser():
    """Test the query parser with various inputs"""
    parser = IntelligentQueryParser()
    
    test_queries = [
        "John birthday cake 2024",
        "Amisha beach vacation last summer",
        "dog park yesterday",
        "wedding celebration Alice Bob",
        "christmas morning family 2023",
        "sunset beach car",
        "Akshara cake birthday party"
    ]
    
    print("🧠 Testing Intelligent Query Parser")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\n📝 Query: '{query}'")
        result = parser.parse_query(query)
        
        if result.person_labels:
            print(f"👤 Persons: {result.person_labels}")
        if result.object_terms:
            print(f"🎯 Objects: {result.object_terms}")
        if result.time_expressions:
            print(f"🕒 Time: {result.time_expressions}")
        if result.confidence_scores:
            print(f"📊 Confidence: {result.confidence_scores}")

if __name__ == "__main__":
    test_query_parser()
