#!/usr/bin/env python3
"""
Manual Photo Indexing Script
Use this when photos were added while the server was down
"""

import os
import sys
import requests
import time
from pathlib import Path

def check_server_status():
    """Check if the backend server is running"""
    try:
        response = requests.get('http://localhost:8000/api/status', timeout=5)
        return response.status_code == 200
    except:
        return False

def get_unindexed_photos():
    """Check how many photos are unindexed"""
    try:
        # Get total photos in folder
        photo_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        sample_photos_dir = Path('sample_photos')
        total_files = sum(1 for f in sample_photos_dir.glob('*') 
                         if f.suffix.lower() in photo_extensions)
        
        # Get indexed photos count
        response = requests.get('http://localhost:8000/api/stats')
        indexed_count = response.json().get('total_photos', 0)
        
        return total_files, indexed_count
    except Exception as e:
        print(f"Error checking photo counts: {e}")
        return 0, 0

def trigger_manual_indexing():
    """Trigger manual indexing of all photos"""
    try:
        response = requests.post('http://localhost:8000/api/index', 
                               json={'directory': 'sample_photos', 'recursive': True})
        if response.status_code == 200:
            task_info = response.json()
            print(f"✅ Indexing started! Task ID: {task_info['task_id']}")
            return task_info['task_id']
        else:
            print(f"❌ Failed to start indexing: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Error starting indexing: {e}")
        return None

def monitor_indexing_progress(task_id):
    """Monitor the indexing progress"""
    print(f"📊 Monitoring task {task_id}...")
    
    while True:
        try:
            response = requests.get(f'http://localhost:8000/api/tasks/{task_id}')
            if response.status_code == 200:
                task_status = response.json()
                status = task_status.get('status', 'unknown')
                progress = task_status.get('progress', 0)
                total = task_status.get('total', 0)
                
                if status == 'running':
                    print(f"🔄 Processing... {progress}/{total} files ({progress/total*100:.1f}%)")
                elif status == 'completed':
                    print(f"✅ Indexing completed! Processed {total} files.")
                    break
                elif status == 'failed':
                    error = task_status.get('error', 'Unknown error')
                    print(f"❌ Indexing failed: {error}")
                    break
                else:
                    print(f"📋 Status: {status}")
                    
            time.sleep(5)  # Check every 5 seconds
            
        except KeyboardInterrupt:
            print("\n⏸️ Monitoring stopped (indexing continues in background)")
            break
        except Exception as e:
            print(f"❌ Error monitoring progress: {e}")
            break

def restart_auto_indexing():
    """Restart the auto-indexing service"""
    try:
        # Stop auto-indexing
        response = requests.post('http://localhost:8000/api/auto-index/stop')
        print("🛑 Stopped auto-indexing service")
        
        time.sleep(2)
        
        # Start auto-indexing
        response = requests.post('http://localhost:8000/api/auto-index/start')
        if response.status_code == 200:
            print("✅ Auto-indexing service restarted")
        else:
            print(f"⚠️ Failed to restart auto-indexing: {response.text}")
            
    except Exception as e:
        print(f"❌ Error restarting auto-indexing: {e}")

def main():
    print("🔍 Manual Photo Indexing Tool")
    print("=" * 40)
    
    # Check server status
    if not check_server_status():
        print("❌ Backend server is not running!")
        print("   Start it with: python -m uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload")
        sys.exit(1)
    
    print("✅ Backend server is running")
    
    # Check photo counts
    total_files, indexed_count = get_unindexed_photos()
    unindexed = total_files - indexed_count
    
    print(f"📊 Photo Status:")
    print(f"   Total photos in folder: {total_files}")
    print(f"   Already indexed: {indexed_count}")
    print(f"   Unindexed photos: {unindexed}")
    
    if unindexed == 0:
        print("✅ All photos are already indexed!")
        return
    
    # Ask user what to do
    print(f"\n🤔 Found {unindexed} unindexed photos. What would you like to do?")
    print("1. Start manual indexing")
    print("2. Restart auto-indexing service only")
    print("3. Both (recommended)")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        task_id = trigger_manual_indexing()
        if task_id:
            monitor_indexing_progress(task_id)
            
    elif choice == "2":
        restart_auto_indexing()
        
    elif choice == "3":
        task_id = trigger_manual_indexing()
        if task_id:
            monitor_indexing_progress(task_id)
        restart_auto_indexing()
        
    elif choice == "4":
        print("👋 Goodbye!")
        
    else:
        print("❌ Invalid choice!")

if __name__ == "__main__":
    main()