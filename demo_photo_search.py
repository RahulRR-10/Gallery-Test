#!/usr/bin/env python3
"""
🌟 ULTIMATE PHOTO SEARCH DEMO
============================
Complete demonstration of the on-device AI photo search system!

This script showcases all features:
- CLIP embeddings for semantic search
- YOLO object detection
- Face detection and recognition
- Natural language queries
- Automatic indexing
- Visual results display

Run this to see the magic! ✨
"""

import os
import sys
from auto_photo_search import AutoPhotoIndexer

def demo_header():
    """Print the demo header"""
    print("=" * 60)
    print("🌟 ULTIMATE PHOTO SEARCH SYSTEM DEMO")
    print("=" * 60)
    print("🤖 AI-Powered On-Device Photo Search")
    print("🧠 CLIP + YOLO + Face Detection")
    print("🔍 Natural Language Queries")
    print("📱 100% Local Processing")
    print("=" * 60)
    print()

def run_demo_searches():
    """Run a series of demo searches"""
    indexer = AutoPhotoIndexer()
    
    # Demo searches
    demo_queries = [
        ("motorcycle racing sport", "🏍️ VEHICLES"),
        ("person wearing tie formal", "👔 PEOPLE"),
        ("beautiful flower garden", "🌸 FLOWERS"),
        ("delicious food dessert cake", "🍰 FOOD"),
        ("colorful nature outdoor", "🌈 NATURE")
    ]
    
    print("🔄 Auto-indexing all photos...")
    stats = indexer.auto_index_new_photos()
    print(f"✅ Database ready with {stats['total']} photos!\n")
    
    print("🎭 RUNNING DEMO SEARCHES")
    print("-" * 40)
    
    for query, category in demo_queries:
        print(f"\n{category}")
        print(f"🔍 Query: '{query}'")
        print("📊 Top 3 Results:")
        
        # Get searcher and perform search
        searcher = indexer.initialize_searcher()
        results = searcher.search_photos(query, limit=3, show_results=False)
        
        if results:
            for i, result in enumerate(results, 1):
                filename = os.path.basename(result['path'])
                similarity = result['similarity']
                
                # Show objects if available
                objects_info = ""
                if result.get('objects'):
                    try:
                        objects_str = result['objects']
                        if objects_str and objects_str != 'None':
                            if ',' in objects_str:
                                objects = objects_str.split(',')
                                objects_info = f" [Objects: {', '.join([obj.strip() for obj in objects])}]"
                            else:
                                objects_info = f" [Objects: {objects_str}]"
                    except:
                        pass
                
                # Show faces if available
                faces_info = ""
                if result.get('faces'):
                    try:
                        faces_str = result['faces']
                        if faces_str and faces_str != 'None' and faces_str.startswith('['):
                            faces = eval(faces_str)
                            if isinstance(faces, list) and faces:
                                faces_info = f" [👤 {len(faces)} faces]"
                    except:
                        pass
                
                print(f"   {i}. {filename} (similarity: {similarity:.3f}){objects_info}{faces_info}")
        else:
            print("   No results found")
    
    print("\n" + "=" * 60)

def show_system_capabilities():
    """Show what the system can do"""
    print("🚀 SYSTEM CAPABILITIES")
    print("-" * 30)
    print("✅ Semantic Image Search")
    print("   • 'red flower in garden'")
    print("   • 'person wearing blue shirt'")
    print("   • 'motorcycle on road'")
    print()
    print("✅ Object Detection (YOLO)")
    print("   • 80+ object categories")
    print("   • Cars, people, animals, food, etc.")
    print("   • Bounding box coordinates")
    print()
    print("✅ Face Detection")
    print("   • OpenCV + face_recognition")
    print("   • Face embeddings for similarity")
    print("   • Multiple faces per image")
    print()
    print("✅ Smart Indexing")
    print("   • Automatic duplicate detection")
    print("   • Batch processing")
    print("   • Progress tracking")
    print()
    print("✅ Natural Language Queries")
    print("   • 'Show me happy people'")
    print("   • 'Find vehicles and transportation'")
    print("   • 'Beautiful nature photos'")
    print()

def show_technical_details():
    """Show technical implementation details"""
    print("🔬 TECHNICAL IMPLEMENTATION")
    print("-" * 35)
    print("🧠 CLIP Model: OpenAI ViT-B/32")
    print("   • 151M parameters")
    print("   • 512-dimensional embeddings")
    print("   • Image + text understanding")
    print()
    print("🎯 YOLO Model: YOLOv8n")
    print("   • Real-time object detection")
    print("   • 80 COCO object classes")
    print("   • Confidence scoring")
    print()
    print("👤 Face Detection: OpenCV + face_recognition")
    print("   • Haar cascade detection")
    print("   • 128-dimensional face encodings")
    print("   • Face similarity comparison")
    print()
    print("💾 Database: SQLite")
    print("   • BLOB storage for embeddings")
    print("   • JSON for metadata")
    print("   • Efficient similarity search")
    print()
    print("🔍 Search: Cosine Similarity")
    print("   • Vector space retrieval")
    print("   • Ranked results")
    print("   • Multi-modal queries")
    print()

def interactive_demo():
    """Interactive demo mode"""
    indexer = AutoPhotoIndexer()
    
    print("🎮 INTERACTIVE DEMO MODE")
    print("-" * 30)
    print("Try some searches! Examples:")
    print("• 'motorcycle sport racing'")
    print("• 'person with formal dress'")
    print("• 'colorful flower garden'")
    print("• 'delicious food cake'")
    print("• 'beautiful nature scene'")
    print()
    print("Commands:")
    print("• 'stats' - Show database statistics")
    print("• 'help' - Show this help")
    print("• 'quit' - Exit demo")
    print()
    
    while True:
        try:
            query = input("🔍 Demo Search> ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("🎭 Demo complete! Thanks for trying the system!")
                break
            
            if query.lower() == 'help':
                print("\n🎯 Available commands:")
                print("• Any text query for semantic search")
                print("• 'stats' - Database statistics")
                print("• 'quit' - Exit demo")
                continue
            
            if query.lower() == 'stats':
                searcher = indexer.initialize_searcher()
                searcher.show_stats()
                continue
            
            if not query:
                continue
            
            # Perform search
            print(f"\n🔍 Searching for: '{query}'")
            searcher = indexer.initialize_searcher()
            results = searcher.search_photos(query, limit=5, show_results=False)
            
            if results:
                print(f"✅ Found {len(results)} matching photos:")
                for i, result in enumerate(results, 1):
                    filename = os.path.basename(result['path'])
                    similarity = result['similarity']
                    print(f"   {i}. {filename} (similarity: {similarity:.3f})")
            else:
                print("❌ No matching photos found")
            
            print()
            
        except KeyboardInterrupt:
            print("\n🎭 Demo complete! Thanks for trying the system!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main demo function"""
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        # Interactive mode only
        demo_header()
        interactive_demo()
    else:
        # Full demo
        demo_header()
        show_system_capabilities()
        print()
        show_technical_details()
        print()
        run_demo_searches()
        print()
        
        # Ask if user wants interactive mode
        try:
            response = input("🎮 Would you like to try interactive search? (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                print()
                interactive_demo()
            else:
                print("🎭 Demo complete! Thanks for exploring the system!")
        except KeyboardInterrupt:
            print("\n🎭 Demo complete! Thanks for exploring the system!")

if __name__ == "__main__":
    main()
