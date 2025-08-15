import os
import time
import requests

os.system('start powershell -NoExit -Command "python -m app.server.server"')

print("⏳ Waiting for server to finish training the initial model...")

while not os.path.exists("model_ready.txt"):
    time.sleep(1)

print("✅ Initial model is ready. Proceeding to start clients...")

#clients = {f"client{i}": 6000 + i for i in range(1, 10)}   # 10 clients
clients = {f"client{i}": 6000 + i for i in range(1, 6)}   # 5  clients
#clients["client10"] = 5005 

for client, port in clients.items():
    os.system(f'start powershell -NoExit -Command "cd app\\clients\\{client}; docker compose up"')

print("🛡️ Starting attacker container...")
os.system('start powershell -NoExit -Command "cd attacker; docker compose up"')

time.sleep(30)

try:
    response = requests.post("http://localhost:5000/start_federated_learning")
    if response.status_code == 200:
        print("🚀 Federated Learning started successfully!")
    else:
        print("❌ Failed to start FL")
except Exception as e:
    print("⚠️ Make sure server is running, then manually POST to /start_federated_learning")
    print(f"Error: {e}")

if os.path.exists("model_ready.txt"):
    os.remove("model_ready.txt")
