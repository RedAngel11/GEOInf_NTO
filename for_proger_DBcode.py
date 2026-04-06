import paho.mqtt.client as mqtt
import json

BROKER = "broker.emqx.io"
PORT = 1883
TOPIC = "rokot_kosmodroma/river"

def on_message(client, userdata, msg):
    info = json.loads(msg.payload.decode())
    
    timestamp = info['timestamp']
    ec = info['ec_microsiemens']
    temp = info['temperature_celsius']

    # i dalshe mozhno rabotat s dannimi

client = mqtt.Client()
client.on_message = on_message
client.connect(BROKER, PORT)
client.subscribe(TOPIC)

client.loop_forever()