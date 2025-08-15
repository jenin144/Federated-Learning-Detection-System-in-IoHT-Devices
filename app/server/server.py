#server.py
from flask import Flask, jsonify, request
import tensorflow as tf
import numpy as np
import requests
import os
import pandas as pd
import threading
import time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from app.model.model_definition import create_keras_model
from app.utils.oversampling import make_tf_dataset
from sklearn.metrics import precision_score, recall_score, f1_score
import tenseal as ts
import base64
import sqlite3
from datetime import datetime


tenseal_context = None

app = Flask(__name__)

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(APP_DIR, 'model', 'model.h5')
SERVER_TRAIN_DATA = os.path.join(APP_DIR, 'server', 'data', 'server_train_data.csv')
SERVER_TEST_DATA = os.path.join(APP_DIR, 'server', 'data', 'server_test_data.csv')

CLIENT_BASE_URL = "http://localhost"
#CLIENT_PORTS = [6001 , 6002, 6003, 6004, 6005, 6006, 6007, 6008, 6009, 5005]  # 10 clients
CLIENT_PORTS = [6001 , 6002, 6003, 6004, 6005]
CLIENT_URLS = [f"{CLIENT_BASE_URL}:{port}" for port in CLIENT_PORTS]
#CLIENT_IDS = ["client1", "client2", "client3", "client4", "client5", "client6", "client7", "client8", "client9", "client10"] # 10 clients
CLIENT_IDS = ["client1", "client2", "client3", "client4", "client5"] 

# نظام الجولات المحدث
training_enabled = False
model = None
lock = threading.Lock()
model_file_lock = threading.Lock()

NUM_CLIENTS = len(CLIENT_PORTS)
MIN_CLIENTS_FOR_AGGREGATION = 4
MIN_TESTED_SAMPLES_FOR_TRAINING = 1000  # الحد الأدنى من العينات المختبرة للتدريب
CHECK_INTERVAL = 15  # فحص حالة العملاء كل 15 ثانية

# متغيرات إدارة الجولات
current_round = 0
active_round = None
round_results = {}
client_status_cache = {}  # تخزين مؤقت لحالة العملاء
last_status_check = 0
######################################################################
def init_db():
    conn = sqlite3.connect("clients.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS clients (
            client_ip TEXT PRIMARY KEY,
            client_id TEXT,
            username TEXT,
            password TEXT,
            device_type TEXT,
            model_version TEXT,
            registered_at TEXT,
            wifi_connected INTEGER DEFAULT 1,
            is_charging INTEGER DEFAULT 1,
            is_idle INTEGER DEFAULT 1,
            can_start_training INTEGER DEFAULT 0,
            untrained_records INTEGER DEFAULT 0,
            total_tested_records INTEGER DEFAULT 0,
            total_trained_records INTEGER DEFAULT 0,
            num_of_rounds INTEGER DEFAULT 0
        )
    ''')
    # Get existing columns
    cursor.execute("PRAGMA table_info(clients)")
    existing_cols = [row[1] for row in cursor.fetchall()]
    # Add columns if they don't exist
    for col, default in [
        ("client_id", "'client1'"),
        ("can_start_training", "0"),
        ("untrained_records", "0"),
        ("total_tested_records", "0"),
        ("total_trained_records", "0")
    ]:
        if col not in existing_cols:
            cursor.execute(f'ALTER TABLE clients ADD COLUMN {col} INTEGER DEFAULT {default}')
    conn.commit()
    conn.close()

######################################################################
def initialize_ckks():
    global tenseal_context
    tenseal_context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    tenseal_context.global_scale = 2**40
    tenseal_context.generate_galois_keys()
    tenseal_context.generate_relin_keys()

@app.route('/get_ckks_context', methods=['GET'])
def get_ckks_context():
    global tenseal_context
    public_context = tenseal_context.copy()
    public_context.make_context_public()
    serialized_context = public_context.serialize()
    serialized_b64 = base64.b64encode(serialized_context).decode('utf-8')
    return jsonify({"context": serialized_b64}), 200

def initialize_model():
    global model
    if not os.path.exists(MODEL_PATH):
        print("🆕 Creating and training initial model...")
        train_df = pd.read_csv(SERVER_TRAIN_DATA)
        model = create_keras_model()
        batch_size = 64
        train_dataset = make_tf_dataset(train_df, batch_size=batch_size, balance=True)

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='categorical_accuracy',
            patience=3,
            restore_best_weights=True
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'categorical_accuracy']
        )

        model.fit(train_dataset, epochs=15, verbose=1, callbacks=[early_stop])
        model.save(MODEL_PATH)
        print("✅ Initial model trained and saved.")

        with open("model_ready.txt", "w") as f:
            f.write("ready")
    else:
        print("📦 Loading existing model...")
        model = tf.keras.models.load_model(MODEL_PATH)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'categorical_accuracy']
        )

        if not os.path.exists("model_ready.txt"):
            with open("model_ready.txt", "w") as f:
                f.write("ready")

def evaluate_model_properly():
    """تقييم صحيح للـ multiclass classification"""
    try:
        test_df = pd.read_csv(SERVER_TEST_DATA)
        batch_size = 128
        test_dataset = make_tf_dataset(test_df, batch_size=batch_size, balance=False)
        
        loss, accuracy, categorical_accuracy = model.evaluate(test_dataset, verbose=0)
        
        y_true_all = []
        y_pred_all = []
        
        for batch_x, batch_y in test_dataset:
            predictions = model.predict(batch_x, verbose=0)
            y_true_batch = np.argmax(batch_y, axis=1)
            y_pred_batch = np.argmax(predictions, axis=1)
            y_true_all.extend(y_true_batch)
            y_pred_all.extend(y_pred_batch)
        
        manual_accuracy = accuracy_score(y_true_all, y_pred_all)
        num_classes = len(np.unique(y_true_all))
        precision_macro = precision_score(y_true_all, y_pred_all, average='macro', zero_division=0)
        recall_macro = recall_score(y_true_all, y_pred_all, average='macro', zero_division=0)
        f1_macro = f1_score(y_true_all, y_pred_all, average='macro', zero_division=0)
        
        return {
            'loss': float(loss),
            'keras_accuracy': float(accuracy),
            'categorical_accuracy': float(categorical_accuracy),
            'manual_accuracy': float(manual_accuracy),
            'num_classes': int(num_classes),
            'total_samples': len(y_true_all),
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'f1_macro': float(f1_macro)
        }
    except Exception as e:
        print(f"❌ Error in detailed evaluation: {e}")
        return None
###################################################################################################################################3
@app.route('/notify_ready', methods=['POST'])
def notify_ready():
    try:
        data = request.get_json()
        client_id = data.get('client_id')
        untrained = data.get('untrained_records', 0)
        total_tested = data.get('total_tested_records', 0)
        total_trained = data.get('total_trained_records', 0)

        print(f"📬 Client {client_id} notified it's ready for training ({untrained} untrained records)")

        # Update cache
        client_status_cache[client_id] = {
            'untrained_records': untrained,
            'can_start_training': True,
            'total_tested_records': total_tested,
            'total_trained_records': total_trained
        }
        # Update DB
        conn = sqlite3.connect("clients.db")
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE clients SET
                can_start_training=1,
                untrained_records=?,
                total_tested_records=?,
                total_trained_records=?,
                num_of_rounds=?
            WHERE client_id=?
        """, (untrained, total_tested, total_trained, current_round, new_id))
        conn.commit()
        conn.close()

        return jsonify({"message": "Status received"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

#######################
@app.route('/update_status', methods=['POST'])
def update_status():
    data = request.get_json()
    client_id = data.get('client_id')
    can_start_training = data.get('can_start_training')
    untrained = data.get('untrained_records')
    total_tested = data.get('total_tested_records')
    total_trained = data.get('total_trained_records')
    if client_id not in client_status_cache:
        return jsonify({"error": "Client not registered"}), 404

    if untrained is not None:
        client_status_cache[client_id]['untrained_records'] = untrained
    if total_tested is not None:
        client_status_cache[client_id]['total_tested_records'] = total_tested
    if total_trained is not None:
        client_status_cache[client_id]['total_trained_records'] = total_trained
    if can_start_training is not None:
        client_status_cache[client_id]['can_start_training'] = bool(can_start_training)
    # Update DB
    conn = sqlite3.connect("clients.db")
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE clients SET
            can_start_training=?,
            untrained_records=?,
            total_tested_records=?,
            total_trained_records=?,
            num_of_rounds=?
        WHERE client_id=? 
    """, (
        1 if client_status_cache[client_id].get('can_start_training', False) else 0,
        client_status_cache[client_id].get('untrained_records', 0),
        client_status_cache[client_id].get('total_tested_records', 0),
        client_status_cache[client_id].get('total_trained_records', 0),
        current_round,
        client_id
    ))
    conn.commit()
    conn.close()
    print(f"✅ Client {client_id} status updated: {client_status_cache[client_id].get('untrained_records')} untrained")
    return jsonify({"message": "Status updated"}), 200

#################################################
def continuous_training_loop():
    """حلقة التدريب المستمرة مع نظام الـ batches"""
    global training_enabled, current_round
    
    current_round = 0
    print("🔄 Starting continuous batch-based training loop...")
    print(f"⚙️ Configuration:")
    print(f"   • Minimum clients for training: {MIN_CLIENTS_FOR_AGGREGATION}")
    print(f"   • Minimum tested samples per client: {MIN_TESTED_SAMPLES_FOR_TRAINING}")
    print(f"   • Client status check interval: {CHECK_INTERVAL} seconds")
    
    while training_enabled:
        try:
            ready_clients = []

            for client_id, status in client_status_cache.items():
                untrained = status.get('untrained_records', 0)
                can_train = status.get('can_start_training', False)

                if can_train and untrained >= MIN_TESTED_SAMPLES_FOR_TRAINING:
                    ready_clients.append({
                        'client_id': client_id,
                        'untrained_samples': untrained,
                        'total_tested': status.get('total_tested_records', 0),
                        'total_trained': status.get('total_trained_records', 0)
                    })

            
            if len(ready_clients) >= MIN_CLIENTS_FOR_AGGREGATION:
                current_round += 1
                print(f"\n🚀 Starting training round {current_round}")
                print(f"📊 Ready clients: {len(ready_clients)}")
                
                for client in ready_clients:
                    print(f"   • {client['client_id']}: {client['untrained_samples']} untrained samples")
                
                # تنفيذ جولة التدريب
                run_training_round(current_round, ready_clients)
                
            else:
                ready_count = len(ready_clients)
                print(f"⏳ Waiting for more clients... ({ready_count}/{MIN_CLIENTS_FOR_AGGREGATION} ready)")
                
                if ready_clients:
                    print("   Ready clients:")
                    for client in ready_clients:
                        print(f"   • {client['client_id']}: {client['untrained_samples']} untrained")
                
                # انتظار قبل الفحص التالي
                time.sleep(CHECK_INTERVAL)
                
        except Exception as e:
            print(f"❌ Error in training loop: {e}")
            time.sleep(CHECK_INTERVAL)
    
    print("🛑 Training loop stopped")
#############################################################################################################################
def run_training_round(round_num, ready_clients):
    """تنفيذ جولة تدريب مع العملاء الجاهزين"""
    global active_round, round_results, model
    
    active_round = round_num
    round_results[round_num] = []
    
    # تحضير أوزان النموذج
    with model_file_lock:
        model = tf.keras.models.load_model(MODEL_PATH)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'categorical_accuracy']
        )
        weights = model.get_weights()
        serialized_weights = [w.tolist() for w in weights]
    
    # إرسال أوامر التدريب للعملاء الجاهزين
    successful_starts = 0
    participating_clients = []
    
    for client in ready_clients:
        client_id = client['client_id']
        client_url = None
        
        # العثور على URL العميل
        for cid, curl in zip(CLIENT_IDS, CLIENT_URLS):
            if cid == client_id:
                client_url = curl
                break
        
        if not client_url:
            print(f"❌ URL not found for client {client_id}")
            continue
        
        try:
            response = requests.post(
                f"{client_url}/start_round",
                json={
                    "round": round_num,
                    "weights": serialized_weights
                },
                timeout=30
            )
            
            if response.status_code == 200:
                successful_starts += 1
                participating_clients.append(client_id)
                print(f"✅ {client_id} started training round {round_num}")
            else:
                print(f"❌ {client_id} failed to start training: {response.status_code}")
                try:
                    error_info = response.json()
                    print(f"   Error details: {error_info}")
                except:
                    print(f"   Response: {response.text}")
                    
        except Exception as e:
            print(f"❌ Error contacting {client_id}: {e}")
    
    if successful_starts == 0:
        print(f"❌ No clients successfully started training for round {round_num}")
        active_round = None
        return
    
    print(f"📊 Round {round_num}: {successful_starts} clients participating")
    
    # انتظار النتائج
    print("⏳ Waiting for training results...")
    timeout = 300  # 5 دقائق
    start_time = time.time()
    
    while time.time() - start_time < timeout and training_enabled:
        with lock:
            received_results = len(round_results[round_num])
            if received_results >= min(successful_starts, MIN_CLIENTS_FOR_AGGREGATION):
                print(f"✅ Received results from {received_results} clients")
                break
        
        print(f"⏳ Results received: {received_results}/{min(successful_starts, MIN_CLIENTS_FOR_AGGREGATION)}")
        time.sleep(10)
    
    # معالجة النتائج
    if training_enabled:
        with lock:
            if len(round_results[round_num]) >= MIN_CLIENTS_FOR_AGGREGATION:
                process_round_results(round_num)
            else:
                print(f"❌ Round {round_num} timed out - insufficient results")
    
    # تنظيف
    active_round = None
    if round_num in round_results:
        del round_results[round_num]
###########################################################################################################################
def average_encrypted_deltas(encrypted_deltas_list):
    """حساب متوسط الدلتات المشفرة"""
    aggregated = encrypted_deltas_list[0]
    for ct in encrypted_deltas_list[1:]:
        aggregated += ct
    n = len(encrypted_deltas_list)
    aggregated = aggregated * (1.0 / n)
    return aggregated

def process_round_results(round_num):
    """معالجة نتائج الجولة وتحديث النموذج"""
    global model, round_results, tenseal_context

    print(f"\n🔍 Processing encrypted results for round {round_num}")

    if round_num not in round_results or len(round_results[round_num]) < MIN_CLIENTS_FOR_AGGREGATION:
        print(f"⚠️ Not enough results for round {round_num}. Skipping aggregation.")
        return

    all_encrypted_deltas = [res['deltas'] for res in round_results[round_num]]
    client_info = [res['client_id'] for res in round_results[round_num]]
    
    print(f"📊 Aggregating encrypted deltas from {len(all_encrypted_deltas)} clients: {client_info}")

    # تجميع الدلتات المشفرة
    avg_encrypted_delta = average_encrypted_deltas(all_encrypted_deltas)

    # فك التشفير
    avg_delta_plain = avg_encrypted_delta.decrypt()
    avg_delta_np = np.array(avg_delta_plain)

    # تطبيق الدلتات على النموذج
    weights = model.get_weights()
    new_weights = []
    start = 0
    
    for w in weights:
        size = w.size
        shape = w.shape
        layer_delta = avg_delta_np[start:start+size].reshape(shape)
        new_weights.append(w + layer_delta)
        start += size

    model.set_weights(new_weights)

    # حفظ النموذج المحدث
    with model_file_lock:
        model.save(MODEL_PATH)
    print(f"💾 Model updated and saved for round {round_num}")

    # تقييم النموذج
    eval_results = evaluate_model_properly()
    if eval_results:
        print(f"\n📊 Round {round_num} Results:")
        print(f"   • Accuracy: {eval_results['manual_accuracy']:.4f} ({eval_results['manual_accuracy']*100:.2f}%)")
        print(f"   • Precision (Macro): {eval_results['precision_macro']:.4f}")
        print(f"   • Recall (Macro): {eval_results['recall_macro']:.4f}")
        print(f"   • F1-Score (Macro): {eval_results['f1_macro']:.4f}")
        print(f"   • Test Samples: {eval_results['total_samples']}")

@app.route('/receive_deltas', methods=['POST'])
def receive_deltas():
    """استقبال الدلتات المشفرة من العملاء"""
    global round_results, active_round, tenseal_context
    
    try:
        data = request.get_json(force=True)
        encrypted_deltas_b64 = data.get('deltas')
        client_round = data.get('round')
        client_id = data.get('client_id')
        training_info = data.get('training_info', {})

        if not all([encrypted_deltas_b64, client_round, client_id]):
            return jsonify({"error": "Missing parameters"}), 400

        if client_round != active_round:
            return jsonify({
                "status": "error",
                "message": f"Round {client_round} is not active (active round: {active_round})"
            }), 409

        # فك تشفير الدلتات
        encrypted_bytes = base64.b64decode(encrypted_deltas_b64)
        encrypted_delta = ts.ckks_vector_from(tenseal_context, encrypted_bytes)

        with lock:
            if client_round not in round_results:
                round_results[client_round] = []
            
            # تحقق من عدم التكرار
            for existing in round_results[client_round]:
                if existing['client_id'] == client_id:
                    return jsonify({
                        "status": "error",
                        "message": "Deltas already received from this client"
                    }), 409
            
            round_results[client_round].append({
                "client_id": client_id,
                "deltas": encrypted_delta,
                "training_info": training_info
            })

        print(f"📥 Received encrypted deltas from {client_id} for round {client_round}")
        if training_info:
            data_range = training_info.get('data_range', 'Unknown')
            print(f"   Training range: {data_range}")
        
        return jsonify({"status": "success"}), 200

    except Exception as e:
        print(f"❌ Error receiving deltas: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/start_federated_learning', methods=['POST'])
def start_federated_learning():
    """بدء عملية التدريب المتوزعة"""
    global training_enabled
    
    if training_enabled:
        return jsonify({
            "status": "error",
            "message": "Training already in progress"
        }), 409
    
    # تقييم النموذج الأولي
    print("📊 Initial Model Evaluation:")
    initial_eval = evaluate_model_properly()
    if initial_eval:
        print(f"   • Initial Accuracy: {initial_eval['manual_accuracy']:.4f}")
        print(f"   • Number of Classes: {initial_eval['num_classes']}")
        print(f"   • Test Samples: {initial_eval['total_samples']}")
    
    # بدء التدريب
    training_enabled = True
    threading.Thread(target=continuous_training_loop, daemon=True).start()
    
    return jsonify({
        "status": "started",
        "message": "Batch-based federated learning started",
        "configuration": {
            "min_clients_for_aggregation": MIN_CLIENTS_FOR_AGGREGATION,
            "min_tested_samples_per_client": MIN_TESTED_SAMPLES_FOR_TRAINING,
            "check_interval_seconds": CHECK_INTERVAL
        }
    }), 200

@app.route('/stop_federated_learning', methods=['POST'])
def stop_federated_learning():
    """إيقاف عملية التدريب المتوزعة"""
    global training_enabled
    
    if not training_enabled:
        return jsonify({
            "status": "error",
            "message": "Training is not running"
        }), 409
    
    training_enabled = False
    print("🛑 Training stop requested...")
    
    return jsonify({
        "status": "stopped",
        "message": "Training will stop after current round completes"
    }), 200


@app.route('/set_config', methods=['POST'])
def set_config():
    """تعديل إعدادات السيرفر"""
    global MIN_CLIENTS_FOR_AGGREGATION, MIN_TESTED_SAMPLES_FOR_TRAINING, CHECK_INTERVAL
    
    try:
        data = request.get_json()
        
        if 'min_clients' in data:
            new_min_clients = data['min_clients']
            if isinstance(new_min_clients, int) and new_min_clients > 0:
                MIN_CLIENTS_FOR_AGGREGATION = new_min_clients
        
        if 'min_samples' in data:
            new_min_samples = data['min_samples']
            if isinstance(new_min_samples, int) and new_min_samples > 0:
                MIN_TESTED_SAMPLES_FOR_TRAINING = new_min_samples
        
        if 'check_interval' in data:
            new_interval = data['check_interval']
            if isinstance(new_interval, (int, float)) and new_interval > 0:
                CHECK_INTERVAL = new_interval
        
        return jsonify({
            "status": "success",
            "current_config": {
                "min_clients_for_aggregation": MIN_CLIENTS_FOR_AGGREGATION,
                "min_tested_samples_per_client": MIN_TESTED_SAMPLES_FOR_TRAINING,
                "check_interval_seconds": CHECK_INTERVAL
            }
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

###############################################################
@app.route('/status', methods=['GET'])
def get_server_status():
    """الحصول على حالة السيرفر الحالية"""
    global training_enabled, current_round, active_round, client_status_cache, model

    status_info = {
        "server_status": "running",
        "training_enabled": training_enabled,
        "current_round": current_round,
        "active_round": active_round if active_round else "None",
        "client_info": {
            "configured_clients": NUM_CLIENTS,
            "connected_clients": len(client_status_cache),
            "client_details": client_status_cache
        },
        "training_configuration": {
            "min_clients_for_aggregation": MIN_CLIENTS_FOR_AGGREGATION,
            "min_tested_samples_per_client": MIN_TESTED_SAMPLES_FOR_TRAINING,
            "check_interval_seconds": CHECK_INTERVAL
        },
    }
    return jsonify(status_info), 200
######################################################################
@app.route('/register_client', methods=['POST'])
def register_client():
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        device_type = data.get('device_type')
        model_version = data.get('model_version')

        # IP address assignment (incremental)
        conn = sqlite3.connect("clients.db")
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM clients")
        client_count = cursor.fetchone()[0]
        new_ip = f"192.168.100.{11 + client_count}"
        new_id = f"client{client_count+1}"
        registered_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute('''
            INSERT INTO clients (
                client_ip, client_id, username, password, device_type,
                model_version, registered_at,
                wifi_connected, is_charging, is_idle, can_start_training,
                untrained_records, total_tested_records, total_trained_records
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, 1, 1, 1, 1, 0, 0, 0)
        ''', (new_ip, new_id, username, password, device_type, model_version, registered_at))
        

        conn.commit()
        conn.close()

        return jsonify({
            "status": "success",
            "client_ip": new_ip,
            "client_id": new_id,
            "message": "Client registered successfully"
        }), 201

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

######################################################################
@app.route('/login_client', methods=['POST'])
def login_client():
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        if not username or not password:
            return jsonify({"status": "error", "message": "Missing username or password"}), 400
        conn = sqlite3.connect("clients.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM clients WHERE username=? AND password=?", (username, password))
        user = cursor.fetchone()
        conn.close()
        if user:
            return jsonify({"status": "success", "message": "Login successful"}), 200
        else:
            return jsonify({"status": "error", "message": "Invalid credentials."}), 401
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

######################################################################
@app.route('/get_client_info', methods=['POST'])
def get_client_info():
    try:
        data = request.get_json()
        username = data.get('username')
        if not username:
            return jsonify({"status": "error", "message": "Missing username"}), 400
        conn = sqlite3.connect("clients.db")
        cursor = conn.cursor()
        cursor.execute("SELECT device_type, model_version, client_ip, registered_at FROM clients WHERE username=?", (username,))
        row = cursor.fetchone()
        conn.close()
        if row:
            device_type, model_version, client_ip, registered_at = row
            return jsonify({
                "status": "success",
                "device_type": device_type,
                "model_version": model_version,
                "client_ip": client_ip,
                "registered_at": registered_at
            }), 200
        else:
            return jsonify({"status": "error", "message": "User not found"}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

######################################################################


if __name__ == '__main__':
    print("🚀 Starting Batch-Based Federated Learning Server with CKKS")
    init_db()

    # ... (rest of the main block)
if __name__ == '__main__':
    print("🚀 Starting Batch-Based Federated Learning Server with CKKS")
    print("=" * 60)
    initialize_ckks()
    initialize_model()
    print(f"🎯 Server configured for {NUM_CLIENTS} clients")
    print(f"⚙️ Training Configuration:")
    print(f"   • Minimum clients for aggregation: {MIN_CLIENTS_FOR_AGGREGATION}")
    print(f"   • Minimum tested samples per client: {MIN_TESTED_SAMPLES_FOR_TRAINING}")
    print(f"   • Client status check interval: {CHECK_INTERVAL} seconds")
    print("=" * 60)
    print("🔄 Use /start_federated_learning to begin training")
    print("🛑 Use /stop_federated_learning to stop training")
    print("📊 Use /status to check detailed status")
    app.run(host='0.0.0.0', port=5000, use_reloader=False)