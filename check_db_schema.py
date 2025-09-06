import sqlite3

conn = sqlite3.connect('docfoundry.db')
cursor = conn.cursor()

print('Chunks table schema:')
cursor.execute('PRAGMA table_info(chunks)')
for row in cursor.fetchall():
    print(row)

print('\nDocuments table schema:')
cursor.execute('PRAGMA table_info(documents)')
for row in cursor.fetchall():
    print(row)

conn.close()