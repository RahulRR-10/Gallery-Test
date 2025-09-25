#!/usr/bin/env python3
"""
🎭 Contextual & Scenario-Based Photo Search
==========================================
Enhanced search system that understands contexts, scenarios, activities,
emotions, and real-world situations in photos.

Examples:
- "birthday party" -> finds photos with cake, people celebrating, party decorations
- "outdoor adventure" -> finds photos with nature, hiking, camping, mountains
- "family gathering" -> finds photos with multiple people, food, indoor settings
- "romantic moment" -> finds photos with couples, sunsets, intimate settings
- "work meeting" -> finds photos with laptops, presentations, office settings
"""

import os
import sys
import sqlite3
import logging
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class ContextualQuery:
    """Structure for contextual search queries"""
    original_query: str
    scenario_type: str
    context_keywords: List[str]
    expected_objects: List[str]
    expected_people_count: Optional[int]
    expected_setting: str  # indoor/outdoor/mixed
    emotion_indicators: List[str]
    activity_type: str
    confidence: float

class ContextualSearchEngine:
    """Enhanced search engine that understands contexts and scenarios"""
    
    def __init__(self):
        """Initialize contextual search with scenario patterns"""
        
        # Define scenario patterns with associated objects, settings, and indicators
        self.scenario_patterns = {
            # 🎉 CELEBRATIONS & PARTIES
            "birthday_party": {
                "keywords": ["birthday", "party", "celebrate", "celebration", "bday", "cake", "candles"],
                "objects": ["cake", "person", "dining table", "chair", "cup", "bottle"],
                "people_count": (2, 20),
                "setting": "indoor",
                "emotions": ["happy", "joy", "celebration", "smiling"],
                "activity": "social"
            },
            "wedding": {
                "keywords": ["wedding", "marriage", "bride", "groom", "ceremony", "reception"],
                "objects": ["person", "dining table", "chair", "cake", "wine glass", "tie"],
                "people_count": (10, 100),
                "setting": "mixed",
                "emotions": ["love", "happiness", "formal"],
                "activity": "ceremony"
            },
            "christmas": {
                "keywords": ["christmas", "xmas", "holiday", "festive", "santa", "tree"],
                "objects": ["person", "dining table", "chair", "bottle", "cup"],
                "people_count": (2, 15),
                "setting": "indoor",
                "emotions": ["festive", "family", "cozy"],
                "activity": "holiday"
            },
            
            # 🏃‍♂️ OUTDOOR ACTIVITIES
            "outdoor_adventure": {
                "keywords": ["adventure", "hiking", "camping", "outdoor", "nature", "mountain", "trail"],
                "objects": ["person", "backpack", "bicycle", "sports ball"],
                "people_count": (1, 6),
                "setting": "outdoor",
                "emotions": ["adventurous", "active", "nature"],
                "activity": "sports"
            },
            "beach_vacation": {
                "keywords": ["beach", "vacation", "ocean", "sea", "sand", "swimming", "surf"],
                "objects": ["person", "surfboard", "umbrella", "sports ball"],
                "people_count": (1, 10),
                "setting": "outdoor",
                "emotions": ["relaxed", "fun", "vacation"],
                "activity": "leisure"
            },
            "sports_activity": {
                "keywords": ["sports", "game", "playing", "football", "basketball", "tennis", "soccer"],
                "objects": ["person", "sports ball", "bicycle", "tennis racket"],
                "people_count": (2, 22),
                "setting": "outdoor",
                "emotions": ["competitive", "active", "team"],
                "activity": "sports"
            },
            
            # 👨‍👩‍👧‍👦 FAMILY & RELATIONSHIPS
            "family_gathering": {
                "keywords": ["family", "gathering", "reunion", "together", "relatives", "home"],
                "objects": ["person", "dining table", "chair", "cup", "couch", "tv"],
                "people_count": (3, 20),
                "setting": "indoor",
                "emotions": ["family", "togetherness", "warm"],
                "activity": "social"
            },
            "romantic_moment": {
                "keywords": ["romantic", "couple", "date", "love", "intimate", "together"],
                "objects": ["person", "wine glass", "dining table", "couch"],
                "people_count": (2, 2),
                "setting": "mixed",
                "emotions": ["romantic", "intimate", "love"],
                "activity": "romantic"
            },
            "kids_playing": {
                "keywords": ["kids", "children", "playing", "toys", "playground", "fun"],
                "objects": ["person", "sports ball", "bicycle", "teddy bear", "kite"],
                "people_count": (1, 8),
                "setting": "mixed",
                "emotions": ["playful", "innocent", "fun"],
                "activity": "play"
            },
            
            # 🏢 WORK & PROFESSIONAL
            "work_meeting": {
                "keywords": ["work", "meeting", "office", "business", "conference", "presentation"],
                "objects": ["person", "laptop", "chair", "dining table", "tv", "cell phone"],
                "people_count": (2, 15),
                "setting": "indoor",
                "emotions": ["professional", "focused", "business"],
                "activity": "work"
            },
            "graduation": {
                "keywords": ["graduation", "graduate", "ceremony", "diploma", "school", "university"],
                "objects": ["person", "chair", "tie", "handbag"],
                "people_count": (1, 50),
                "setting": "mixed",
                "emotions": ["achievement", "proud", "formal"],
                "activity": "ceremony"
            },
            
            # 🍽️ FOOD & DINING
            "cooking": {
                "keywords": ["cooking", "kitchen", "chef", "recipe", "food", "preparation"],
                "objects": ["person", "dining table", "cup", "bowl", "knife", "bottle"],
                "people_count": (1, 4),
                "setting": "indoor",
                "emotions": ["creative", "focused", "domestic"],
                "activity": "cooking"
            },
            "dinner_party": {
                "keywords": ["dinner", "party", "dining", "restaurant", "meal", "feast"],
                "objects": ["person", "dining table", "chair", "cup", "wine glass", "fork", "knife"],
                "people_count": (2, 12),
                "setting": "indoor",
                "emotions": ["social", "celebration", "dining"],
                "activity": "dining"
            },
            
            # 🎵 ENTERTAINMENT & LEISURE
            "concert": {
                "keywords": ["concert", "music", "performance", "stage", "band", "singing"],
                "objects": ["person", "microphone", "tv"],
                "people_count": (1, 1000),
                "setting": "mixed",
                "emotions": ["energetic", "musical", "entertainment"],
                "activity": "entertainment"
            },
            "movie_night": {
                "keywords": ["movie", "film", "cinema", "watching", "tv", "couch"],
                "objects": ["person", "tv", "couch", "remote", "cup"],
                "people_count": (1, 6),
                "setting": "indoor",
                "emotions": ["relaxed", "entertainment", "cozy"],
                "activity": "leisure"
            },
            
            # 🏠 HOME & DOMESTIC
            "home_life": {
                "keywords": ["home", "house", "domestic", "daily", "routine", "comfortable"],
                "objects": ["person", "couch", "tv", "dining table", "chair", "book"],
                "people_count": (1, 6),
                "setting": "indoor",
                "emotions": ["comfortable", "domestic", "relaxed"],
                "activity": "domestic"
            },
            "garden": {
                "keywords": ["garden", "gardening", "plants", "flowers", "yard", "backyard"],
                "objects": ["person", "potted plant", "chair", "umbrella"],
                "people_count": (1, 4),
                "setting": "outdoor",
                "emotions": ["peaceful", "nature", "domestic"],
                "activity": "gardening"
            }
        }
        
        # Object synonyms for better matching
        self.object_synonyms = {
            "people": ["person", "man", "woman", "child", "adult"],
            "food": ["cake", "pizza", "hot dog", "sandwich", "apple", "banana"],
            "drinks": ["cup", "wine glass", "bottle", "coffee"],
            "furniture": ["chair", "couch", "dining table", "bed"],
            "technology": ["laptop", "cell phone", "tv", "remote", "keyboard"],
            "sports": ["sports ball", "tennis racket", "bicycle", "surfboard", "skis"],
            "vehicles": ["car", "truck", "bus", "motorcycle", "bicycle", "boat"],
            "animals": ["dog", "cat", "horse", "bird", "cow", "sheep"]
        }
        
        # Emotion-based keywords
        self.emotion_keywords = {
            "happy": ["happy", "joy", "smile", "laugh", "celebration", "fun", "cheerful"],
            "romantic": ["romantic", "love", "intimate", "couple", "date", "valentine"],
            "professional": ["work", "business", "meeting", "formal", "office", "corporate"],
            "family": ["family", "home", "together", "relatives", "parents", "children"],
            "adventure": ["adventure", "outdoor", "hiking", "travel", "explore", "nature"],
            "relaxed": ["relaxed", "calm", "peaceful", "leisure", "vacation", "cozy"]
        }
    
    def analyze_query(self, query: str) -> ContextualQuery:
        """Analyze a query to extract contextual information"""
        query_lower = query.lower().strip()
        
        best_match = None
        best_confidence = 0.0
        
        # Check each scenario pattern
        for scenario_name, pattern in self.scenario_patterns.items():
            confidence = self._calculate_scenario_confidence(query_lower, pattern)
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_match = (scenario_name, pattern)
        
        if best_match and best_confidence > 0.3:  # Minimum confidence threshold
            scenario_name, pattern = best_match
            
            return ContextualQuery(
                original_query=query,
                scenario_type=scenario_name,
                context_keywords=pattern["keywords"],
                expected_objects=pattern["objects"],
                expected_people_count=pattern.get("people_count"),
                expected_setting=pattern["setting"],
                emotion_indicators=pattern["emotions"],
                activity_type=pattern["activity"],
                confidence=best_confidence
            )
        else:
            # Fallback: general contextual analysis
            return self._analyze_general_context(query)
    
    def _calculate_scenario_confidence(self, query: str, pattern: Dict) -> float:
        """Calculate confidence score for a scenario match"""
        total_keywords = len(pattern["keywords"])
        matched_keywords = 0
        
        for keyword in pattern["keywords"]:
            if keyword in query:
                matched_keywords += 1
        
        # Base confidence from keyword matching
        keyword_confidence = matched_keywords / total_keywords if total_keywords > 0 else 0
        
        # Boost confidence for exact matches
        exact_match_boost = 0
        for keyword in pattern["keywords"]:
            if keyword == query.strip():
                exact_match_boost = 0.3
                break
        
        # Boost confidence for partial matches
        partial_match_boost = 0
        query_words = set(query.split())
        pattern_words = set(pattern["keywords"])
        common_words = query_words.intersection(pattern_words)
        if common_words:
            partial_match_boost = len(common_words) * 0.1
        
        final_confidence = min(1.0, keyword_confidence + exact_match_boost + partial_match_boost)
        return final_confidence
    
    def _analyze_general_context(self, query: str) -> ContextualQuery:
        """Analyze query for general contextual clues"""
        query_lower = query.lower().strip()
        
        # Extract potential objects from query
        query_words = query_lower.split()
        potential_objects = []
        
        # Check against object synonyms
        for category, objects in self.object_synonyms.items():
            for word in query_words:
                if word in objects or any(obj in word for obj in objects):
                    potential_objects.extend(objects[:3])  # Add first 3 objects from category
        
        # Determine general activity type
        activity_type = "general"
        if any(word in query_lower for word in ["party", "celebration", "birthday"]):
            activity_type = "social"
        elif any(word in query_lower for word in ["work", "office", "meeting"]):
            activity_type = "work"
        elif any(word in query_lower for word in ["outdoor", "nature", "hiking"]):
            activity_type = "outdoor"
        elif any(word in query_lower for word in ["home", "family", "house"]):
            activity_type = "domestic"
        
        # Estimate setting
        setting = "mixed"
        if any(word in query_lower for word in ["indoor", "inside", "home", "office", "kitchen"]):
            setting = "indoor"
        elif any(word in query_lower for word in ["outdoor", "outside", "nature", "beach", "park"]):
            setting = "outdoor"
        
        return ContextualQuery(
            original_query=query,
            scenario_type="general_context",
            context_keywords=query_words,
            expected_objects=list(set(potential_objects)),
            expected_people_count=None,
            expected_setting=setting,
            emotion_indicators=[],
            activity_type=activity_type,
            confidence=0.5
        )
    
    def generate_enhanced_search_terms(self, contextual_query: ContextualQuery) -> Dict[str, List[str]]:
        """Generate enhanced search terms based on contextual analysis"""
        
        enhanced_terms = {
            "primary_terms": [contextual_query.original_query],
            "object_terms": contextual_query.expected_objects,
            "context_terms": contextual_query.context_keywords,
            "emotion_terms": contextual_query.emotion_indicators,
            "semantic_queries": []
        }
        
        # Generate semantic search queries based on scenario
        if contextual_query.scenario_type != "general_context":
            # Create multiple semantic variations
            scenario_type = contextual_query.scenario_type.replace("_", " ")
            
            enhanced_terms["semantic_queries"] = [
                contextual_query.original_query,
                scenario_type,
                f"{scenario_type} with people",
                f"photos of {scenario_type}",
                " ".join(contextual_query.context_keywords[:3])  # Top 3 context keywords
            ]
        else:
            # General context - create simpler variations
            enhanced_terms["semantic_queries"] = [
                contextual_query.original_query,
                " ".join(contextual_query.context_keywords[:2])
            ]
        
        return enhanced_terms
    
    def get_scenario_info(self, query: str) -> Dict:
        """Get detailed scenario information for debugging/display"""
        contextual_query = self.analyze_query(query)
        enhanced_terms = self.generate_enhanced_search_terms(contextual_query)
        
        return {
            "original_query": query,
            "detected_scenario": contextual_query.scenario_type,
            "confidence": contextual_query.confidence,
            "expected_objects": contextual_query.expected_objects,
            "expected_setting": contextual_query.expected_setting,
            "activity_type": contextual_query.activity_type,
            "enhanced_search_terms": enhanced_terms,
            "search_strategy": self._get_search_strategy(contextual_query)
        }
    
    def _get_search_strategy(self, contextual_query: ContextualQuery) -> Dict:
        """Determine the best search strategy for this context"""
        strategy = {
            "primary_method": "semantic",
            "fallback_methods": ["object_tags", "general_semantic"],
            "weight_objects": 0.4,
            "weight_people": 0.3,
            "weight_semantic": 0.3
        }
        
        # Adjust strategy based on scenario type
        if contextual_query.scenario_type in ["birthday_party", "wedding", "family_gathering"]:
            strategy["weight_people"] = 0.5
            strategy["weight_objects"] = 0.3
            strategy["weight_semantic"] = 0.2
        elif contextual_query.scenario_type in ["outdoor_adventure", "sports_activity"]:
            strategy["weight_objects"] = 0.5
            strategy["weight_semantic"] = 0.4
            strategy["weight_people"] = 0.1
        elif contextual_query.scenario_type == "work_meeting":
            strategy["weight_objects"] = 0.6  # Focus on laptops, tables, etc.
            strategy["weight_people"] = 0.2
            strategy["weight_semantic"] = 0.2
        
        return strategy

# Global instance for easy access
contextual_engine = ContextualSearchEngine()

def analyze_contextual_query(query: str) -> Dict:
    """Convenience function to analyze a contextual query"""
    return contextual_engine.get_scenario_info(query)

def get_enhanced_search_terms(query: str) -> Dict[str, List[str]]:
    """Get enhanced search terms for a contextual query"""
    contextual_query = contextual_engine.analyze_query(query)
    return contextual_engine.generate_enhanced_search_terms(contextual_query)

if __name__ == "__main__":
    # Test the contextual search engine
    test_queries = [
        "birthday party",
        "family gathering",
        "outdoor adventure",
        "work meeting", 
        "romantic dinner",
        "kids playing",
        "beach vacation",
        "cooking at home",
        "christmas celebration",
        "graduation ceremony"
    ]
    
    print("🎭 Contextual Search Engine Test")
    print("=" * 50)
    
    for query in test_queries:
        info = analyze_contextual_query(query)
        print(f"\n🔍 Query: '{query}'")
        print(f"   📋 Scenario: {info['detected_scenario']} (confidence: {info['confidence']:.2f})")
        print(f"   🎯 Objects: {', '.join(info['expected_objects'][:5])}")
        print(f"   🏠 Setting: {info['expected_setting']}")
        print(f"   🎭 Activity: {info['activity_type']}")