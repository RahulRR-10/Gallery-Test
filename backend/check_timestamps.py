import sqlite3
import datetime

try:
    conn = sqlite3.connect('photos.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT path, timestamp, exif_timestamp 
        FROM photos 
        ORDER BY COALESCE(exif_timestamp, timestamp) DESC
        LIMIT 10
    ''')
    
    print("🔍 Database EXIF vs File timestamps:")
    print("=" * 60)
    
    for row in cursor.fetchall():
        filename = row[0].split('/')[-1] if '/' in row[0] else row[0].split('\\')[-1]
        file_ts = row[1]
        exif_ts = row[2]
        
        print(f"📁 File: {filename}")
        
        if file_ts:
            file_date = datetime.datetime.fromtimestamp(file_ts)
            print(f"   📄 File timestamp: {file_date.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if exif_ts:
            exif_date = datetime.datetime.fromtimestamp(exif_ts)
            print(f"   📷 EXIF timestamp: {exif_date.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   ✅ Using EXIF date for sorting")
        else:
            print(f"   ⚠️  No EXIF date found, using file timestamp")
        
        print()
    
    conn.close()
    
except Exception as e:
    print(f"❌ Error: {e}")