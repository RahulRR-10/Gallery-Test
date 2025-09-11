import sqlite3

# Connect to database
conn = sqlite3.connect('photos.db')
cursor = conn.cursor()

# Get all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print("Tables:", [t[0] for t in tables])

# Get schema for each table
for table in tables:
    table_name = table[0]
    print(f"\n--- Schema for {table_name} ---")
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    for col in columns:
        print(f"  {col[1]} {col[2]}")

conn.close()
