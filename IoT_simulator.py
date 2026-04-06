import sqlite3 as sq
import json as js
import time
import paho.mqtt.client as mqtt

BROKER = "broker.emqx.io"
PORT = 1883
TOPIC = "rokot_kosmodroma/river"

connection = sq.connect('for_database.db')
cursor = connection.cursor()

cursor.execute("SELECT timestamp, ec_microsiemens, temperature_celsius, sensor_id, location FROM sensors_data_given ORDER BY timestamp")
rows = cursor.fetchall()

mqtt_client = mqtt.Client()
mqtt_client.connect(BROKER, PORT)

i = 1
for r in rows:
    data = {
        'packet_id': i,
        'timestamp': r[0],
        'ec_microsiemens': r[1],
        'temperature_celsius': r[2],
        'sensor_id': r[3],
        'location': r[4]
    }
    i+=1
    message = js.dumps(data, ensure_ascii=False)
    mqtt_client.publish(TOPIC, message)
    time.sleep(1)

