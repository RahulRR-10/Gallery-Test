#!/usr/bin/env python3
"""
🚀 AUTO-INDEXING PHOTO SEARCH SYSTEM
===================================
This script automatically detects and indexes new photos in the sample_photos folder.
It watches for changes and keeps your photo database up to date!

Usage:
  python auto_photo_search.py                    # Auto-index + Interactive mode
  python auto_photo_search.py --search "query"   # Direct search
  python auto_photo_search.py --watch           # Watch mode (continuous monitoring)
"""

import os
import sys
import time
import argparse
from pathlib import Path
from final_photo_search import UltimatePhotoSearcher

class AutoPhotoIndexer:
    """
    Automatically indexes new photos and provides easy search interface
    """
    
    def __init__(self, photos_dir: str = "sample_photos", db_path: str = "photos.db"):
        self.photos_dir = photos_dir
        self.db_path = db_path
        self.searcher = None
        self.last_scan_time = 0
        
        print("🤖 Auto-Indexing Photo Search System")
        print("=" * 50)
        
        # Create photos directory if it doesn't exist
        if not os.path.exists(photos_dir):
            os.makedirs(photos_dir)
            print(f"📁 Created photos directory: {photos_dir}")
            print("   👉 Add your photos here for automatic indexing!")
    
    def initialize_searcher(self):
        """Initialize the photo searcher (lazy loading)"""
        if self.searcher is None:
            print("🚀 Initializing AI photo search system...")
            self.searcher = UltimatePhotoSearcher(self.db_path)
            print("✅ System ready!")
        return self.searcher
    
    def check_for_new_photos(self) -> bool:
        """Check if there are new photos to index"""
        if not os.path.exists(self.photos_dir):
            return False
        
        # Get current modification time of directory
        dir_mtime = os.path.getmtime(self.photos_dir)
        
        # Check if directory has been modified since last scan
        if dir_mtime > self.last_scan_time:
            self.last_scan_time = time.time()
            return True
        
        return False
    
    def auto_index_new_photos(self) -> dict:
        """Automatically index any new photos in the directory"""
        if not os.path.exists(self.photos_dir):
            print(f"📁 Photos directory not found: {self.photos_dir}")
            return {"total": 0, "processed": 0, "skipped": 0, "errors": 0}
        
        # Count total images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = []
        
        for root, dirs, files in os.walk(self.photos_dir):
            for file in files:
                if Path(file).suffix.lower() in image_extensions:
                    image_files.append(os.path.join(root, file))
        
        if not image_files:
            print(f"📸 No images found in {self.photos_dir}")
            return {"total": 0, "processed": 0, "skipped": 0, "errors": 0}
        
        print(f"📸 Found {len(image_files)} images in {self.photos_dir}")
        
        # Initialize searcher and index photos
        searcher = self.initialize_searcher()
        stats = searcher.index_photos(self.photos_dir)
        
        if stats['processed'] > 0:
            print(f"✅ Successfully indexed {stats['processed']} new photos!")
        elif stats['skipped'] > 0:
            print(f"ℹ️ All {stats['skipped']} photos were already indexed")
        
        return stats
    
    def interactive_search(self):
        """Interactive search mode"""
        searcher = self.initialize_searcher()
        
        print("\n🔍 Interactive Photo Search Mode")
        print("=" * 40)
        print("Enter your search queries (or 'quit' to exit)")
        print("Examples:")
        print("  'red flower' - Find red flowers")
        print("  'person with tie' - Find people wearing ties")
        print("  'motorcycle' - Find vehicles")
        print("  'food cake dessert' - Find food items")
        print()
        
        while True:
            try:
                query = input("🔍 Search> ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not query:
                    continue
                
                if query.lower() == 'stats':
                    searcher.show_stats()
                    continue
                
                if query.lower().startswith('index'):
                    print("🔄 Re-indexing photos...")
                    self.auto_index_new_photos()
                    continue
                
                # Perform search
                print(f"\n🔍 Searching for: '{query}'")
                results = searcher.search_photos(query, limit=5, show_results=False)
                
                if not results:
                    print("❌ No matching photos found")
                else:
                    print(f"✅ Found {len(results)} matching photos!")
                
                print()
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def watch_mode(self):
        """Continuous monitoring mode"""
        print("👀 Watch mode: Monitoring for new photos...")
        print("   (Press Ctrl+C to stop)")
        
        try:
            while True:
                if self.check_for_new_photos():
                    print("\n📸 New photos detected! Auto-indexing...")
                    stats = self.auto_index_new_photos()
                    if stats['processed'] > 0:
                        print(f"✅ Indexed {stats['processed']} new photos")
                
                time.sleep(5)  # Check every 5 seconds
                
        except KeyboardInterrupt:
            print("\n👋 Watch mode stopped")
    
    def quick_search(self, query: str, limit: int = 5):
        """Quick search without interactive mode"""
        searcher = self.initialize_searcher()
        
        # Auto-index first
        print("🔄 Checking for new photos...")
        self.auto_index_new_photos()
        
        # Perform search
        results = searcher.search_photos(query, limit=limit)
        return results

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="🤖 Auto-Indexing Photo Search System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python auto_photo_search.py                        # Auto-index + Interactive mode
  python auto_photo_search.py --search "red flower"  # Quick search
  python auto_photo_search.py --watch                # Continuous monitoring
  python auto_photo_search.py --dir my_photos        # Use different directory
        """
    )
    
    parser.add_argument('--search', type=str, metavar='QUERY',
                       help='Search photos using natural language query')
    parser.add_argument('--watch', action='store_true',
                       help='Watch mode: continuously monitor for new photos')
    parser.add_argument('--dir', type=str, default='sample_photos', metavar='DIRECTORY',
                       help='Photos directory to monitor (default: sample_photos)')
    parser.add_argument('--limit', type=int, default=5, metavar='N',
                       help='Maximum number of search results (default: 5)')
    parser.add_argument('--db', type=str, default='photos.db', metavar='PATH',
                       help='Database file path (default: photos.db)')
    
    args = parser.parse_args()
    
    # Initialize auto-indexer
    indexer = AutoPhotoIndexer(args.dir, args.db)
    
    if args.search:
        # Quick search mode
        indexer.quick_search(args.search, args.limit)
    elif args.watch:
        # Watch mode
        indexer.watch_mode()
    else:
        # Auto-index + Interactive mode
        print("🔄 Auto-indexing photos...")
        indexer.auto_index_new_photos()
        indexer.interactive_search()

if __name__ == "__main__":
    main()
