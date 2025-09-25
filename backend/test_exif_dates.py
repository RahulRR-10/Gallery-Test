import requests
import datetime

try:
    response = requests.get('http://localhost:8000/api/photos?limit=10')
    data = response.json()
    
    print("🔍 Testing EXIF date sorting:")
    print(f"Total photos returned: {len(data['results'])}")
    print("\n📅 Photo dates (should be sorted by EXIF date, not file date):")
    
    for i, photo in enumerate(data['results'][:5]):
        filename = photo['path'].split('/')[-1] if '/' in photo['path'] else photo['path'].split('\\')[-1]
        photo_date = datetime.datetime.fromtimestamp(photo['timestamp'])
        print(f"{i+1}. {filename}")
        print(f"   📅 Date: {photo_date.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   🔢 Timestamp: {photo['timestamp']}")
        print()
        
except Exception as e:
    print(f"❌ Error testing API: {e}")