import sqlite3
import pandas as pd
import time
import json
import os

connection = sqlite3.connect('for_database.db')
cursor = connection.cursor()


cursor.execute('''
    CREATE TABLE IF NOT EXISTS sensors_data_given (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        ec_microsiemens REAL,
        temperature_celsius REAL,
        sensor_id TEXT,
        location TEXT
    )
''')


cursor.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        predict_mineralization REAL,
        predict_temperature REAL,
        prediction_method TEXT
    )
''')

cursor.execute('''
    CREATE TABLE IF NOT EXISTS actual_info (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        actual_mineralization REAL,
        actual_temperature REAL
    )
''')

connection.commit()

data_read_file = pd.read_csv('kiyazevo_iot_realistic.csv')

cursor.execute("DELETE FROM sensors_data_given")
connection.commit()
cursor.execute("DELETE FROM sqlite_sequence WHERE name='sensors_data_given'")
connection.commit()

for index, row in data_read_file.iterrows():
    cursor.execute('''
        INSERT INTO sensors_data_given (timestamp, ec_microsiemens, temperature_celsius, sensor_id, location)
        VALUES (?, ?, ?, ?, ?)'''
    , (row['timestamp'], row['ec_microsiemens'], row['temperature_celsius'], row['sensor_id'], row['location']))

connection.commit()

cursor.execute("SELECT * FROM sensors_data_given LIMIT 5")
rows = cursor.fetchall()
print("\n📋 Первые 5 записей в таблице:")
for row in rows:
    print(row)

cursor.close()
connection.close()