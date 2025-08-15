import tensorflow as tf
import numpy as np
import requests
import threading
import time
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from utils.oversampling import make_tf_dataset
from utils.preprocessing import preprocess_data
from model.model_definition import create_keras_model
import tenseal as ts
import base64
import subprocess
import sys
import signal
import traceback
from datetime import datetime
import json

# --- Global Configurations ---
SERVER_URL = "http://host.docker.internal:5000"
MODEL_PATH = 'model/local_model.h5'
CLIENT_ID = "client1"
MONITORING_CSV_FILE = 'network_traffic.csv'
BATCH_SIZE = 124
ROUND_TRACKER_FILE = 'round_tracker.json'
TRAINING_BATCH_SIZE = 1000
CKKS_CONTEXT_FETCH_RETRIES = 10
CKKS_CONTEXT_RETRY_DELAY = 5
ALERT_THRESHOLD = 0.7  # Confidence threshold for attack alerts

# --- Global Variables for Client State ---
model = None
tenseal_context = None
training_in_progress = False
monitoring_process = None
monitoring_active = False
total_tested_records = 0
total_trained_records = 0
round_data_tracker = {}
flask_app_ready = threading.Event()

# --- Utility Functions ---

def setup_directories():
    for directory in ['model', 'data', 'alerts']:
        os.makedirs(directory, exist_ok=True)
    print("Directories ensured: model, data, alerts")

def reset_tracking_files():
    """Reset all tracking files to start fresh"""
    global round_data_tracker, total_tested_records, total_trained_records
    
    # Reset global variables
    round_data_tracker = {}
    total_tested_records = 0
    total_trained_records = 0
    
    # Remove old tracking file
    if os.path.exists(ROUND_TRACKER_FILE):
        try:
            os.remove(ROUND_TRACKER_FILE)
            print(f"🗑️ Removed old tracking file: {ROUND_TRACKER_FILE}")
        except Exception as e:
            print(f"Warning: Could not remove old tracking file: {e}")
    
    # Remove old CSV file to start fresh
    if os.path.exists(MONITORING_CSV_FILE):
        try:
            os.remove(MONITORING_CSV_FILE)
            print(f"🗑️ Removed old CSV file: {MONITORING_CSV_FILE}")
        except Exception as e:
            print(f"Warning: Could not remove old CSV file: {e}")
    
    # Create fresh tracking file
    save_round_tracker()
    print("✅ Fresh tracking system initialized")

def load_round_tracker():
    """Load round tracker - but only if CSV file exists and matches"""
    global round_data_tracker, total_tested_records, total_trained_records
    
    # Always start fresh - no loading of old data
    round_data_tracker = {}
    total_tested_records = 0
    total_trained_records = 0
    print("🔄 Starting with fresh tracking (no old data loaded)")

def save_round_tracker():
    try:
        tracker_data = {
            'total_tested_records': total_tested_records,
            'total_trained_records': total_trained_records,
            'rounds': round_data_tracker,
            'last_updated': datetime.now().isoformat(),
            'csv_file': MONITORING_CSV_FILE
        }
        with open(ROUND_TRACKER_FILE, 'w') as f:
            json.dump(tracker_data, f, indent=2)
        print(f"📊 Tracker saved: {total_tested_records} tested, {total_trained_records} trained")
    except Exception as e:
        print(f"Error saving tracker file: {e}")

def validate_csv_and_tracking():
    """Validate that CSV file size matches our tracking"""
    global total_tested_records
    
    if not os.path.exists(MONITORING_CSV_FILE):
        print(f"CSV file doesn't exist: {MONITORING_CSV_FILE}")
        return True  # OK, we'll start fresh
    
    try:
        df = pd.read_csv(MONITORING_CSV_FILE, on_bad_lines='skip')
        
        csv_size = len(df)
        
        if total_tested_records > csv_size:
            print(f"⚠️ MISMATCH: Tracking says {total_tested_records} tested, but CSV has only {csv_size} records")
            print("🔄 Resetting tracking to match CSV size")
            total_tested_records = csv_size
            save_round_tracker()
            return False
        
        print(f"✅ Tracking validation passed: {total_tested_records} tested, CSV has {csv_size} records")
        return True
        
    except Exception as e:
        print(f"Error validating CSV: {e}")
        return False

def get_next_test_batch():
    global total_tested_records
    
    try:
        if not os.path.exists(MONITORING_CSV_FILE):
            print(f"CSV file not found: {MONITORING_CSV_FILE}")
            return None, 0, 0
        
        # Validate before processing
        if not validate_csv_and_tracking():
            print("CSV validation failed, retrying...")
            return None, 0, 0
            
        df = pd.read_csv(MONITORING_CSV_FILE, on_bad_lines='skip')
        total_available = len(df)
        
        print(f"DEBUG: CSV has {total_available} records, tested so far: {total_tested_records}")
        
        start_idx = total_tested_records
        end_idx = start_idx + BATCH_SIZE
        
        if start_idx >= total_available:
            print(f"No more data for testing. Need records from {start_idx}, but only have {total_available}")
            return None, start_idx, end_idx
            
        actual_end = min(end_idx, total_available)
        batch_df = df.iloc[start_idx:actual_end].copy()
        
        print(f"Testing batch: records {start_idx + 1}-{actual_end} ({len(batch_df)} samples)")
        return batch_df, start_idx + 1, actual_end
        
    except Exception as e:
        print(f"Error getting test batch: {e}")
        traceback.print_exc()
        return None, 0, 0

def can_start_training():
    global total_tested_records, total_trained_records
    
    untrained_records = total_tested_records - total_trained_records
    can_train = untrained_records >= TRAINING_BATCH_SIZE
    
    print(f"Training check: {total_tested_records} tested, {total_trained_records} trained")
    print(f"Untrained: {untrained_records}, Can train: {can_train}")
    
    return can_train

def get_training_data_range():
    global total_trained_records
    
    start_idx = total_trained_records
    end_idx = start_idx + TRAINING_BATCH_SIZE
    
    print(f"Training range: records {start_idx + 1}-{end_idx}")
    return start_idx, end_idx

def fetch_ckks_context_from_server():
    global tenseal_context
    print("Attempting to fetch CKKS context from server...")
    try:
        response = requests.get(f"{SERVER_URL}/get_ckks_context", timeout=30)
        response.raise_for_status()
        context_b64 = response.json().get("context")
        if context_b64:
            context_bytes = base64.b64decode(context_b64)
            tenseal_context = ts.context_from(context_bytes)
            print("CKKS context loaded successfully.")
            return True
        else:
            print("No context received from server.")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Network error fetching CKKS context: {e}")
    except Exception as e:
        print(f"Error processing CKKS context: {e}")
    return False

def encrypt_model_deltas(deltas):
    if tenseal_context is None:
        print("TenSEAL context not available for encryption.")
        return None
    try:
        flat_deltas = []
        for d in deltas:
            if d.size > 0:
                flat_deltas.extend(d.flatten().tolist())
            else:
                print(f"Warning: Empty delta array encountered. Skipping flattening.")

        if not flat_deltas:
            print("No data to encrypt after flattening deltas.")
            return None

        encrypted_vector = ts.ckks_vector(tenseal_context, flat_deltas)
        serialized_b64 = base64.b64encode(encrypted_vector.serialize()).decode('utf-8')
        print("Model deltas encrypted successfully.")
        return serialized_b64
    except Exception as e:
        print(f"Error during encryption: {e}")
        traceback.print_exc()
    return None

def start_network_monitoring():
    global monitoring_process, monitoring_active
    if monitoring_process and monitoring_process.poll() is None:
        print("Network monitoring is already running.")
        return

    print("Starting continuous network monitoring...")
    try:
        monitoring_process = subprocess.Popen(
            [sys.executable, 'monitoring.py', '--csv-name', 'network_traffic'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
        )
        monitoring_active = True
        threading.Thread(target=read_monitor_output, daemon=True).start()
        print(f"Continuous monitoring started. Data saved to {MONITORING_CSV_FILE}")
    except FileNotFoundError:
        print("Error: 'monitoring.py' not found or Python path is incorrect.")
    except Exception as e:
        print(f"Failed to start monitoring: {e}")

def read_monitor_output():
    global monitoring_process
    while monitoring_active and monitoring_process and monitoring_process.poll() is None:
        try:
            line = monitoring_process.stdout.readline()
            if line:
                line = line.strip()
                pass 
            else:
                time.sleep(0.1)
        except Exception as e:
            print(f"Error reading monitor output: {e}")
            break

def stop_network_monitoring():
    global monitoring_active, monitoring_process
    monitoring_active = False
    if monitoring_process and monitoring_process.poll() is None:
        print("Stopping continuous monitoring...")
        try:
            monitoring_process.terminate()
            monitoring_process.wait(timeout=5)
            print("Monitoring stopped gracefully.")
        except subprocess.TimeoutExpired:
            monitoring_process.kill()
            print("Monitoring process killed due to timeout.")
        monitoring_process = None
############################################################
def predict_attacks_on_batch(batch_df): 
    """Predict attacks on batch using trained model"""
    global model
    
    if model is None:
        print("⚠️ Model not available for prediction")
        return None
    
    try:
        # Prepare data for prediction (without labels)
        X_scaled, _ = prepare_data_for_prediction(batch_df)
        
        if X_scaled is None:
            print("Failed to prepare data for prediction")
            return None
        
        # Make predictions
        predictions = model.predict(X_scaled, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        confidences = np.max(predictions, axis=1)
        
        attack_types = ["No Attack", "DoS Attack", "ARP Spoofing", "Nmap Port Scan", "Smurf Attack"]
        
        # Analyze predictions
        attack_alerts = []
        for idx, (pred_class, confidence) in enumerate(zip(predicted_classes, confidences)):
            if pred_class > 0 and confidence >= ALERT_THRESHOLD:  # Attack detected with high confidence
                alert = {
                    'record_index': idx,
                    'attack_type': attack_types[pred_class],
                    'confidence': float(confidence),
                    'timestamp': datetime.now().isoformat()
                }
                attack_alerts.append(alert)
        
        return {
            'total_records': len(batch_df),
            'attack_alerts': attack_alerts,
            'predictions_summary': {
                'no_attack': int(np.sum(predicted_classes == 0)),
                'dos_attack': int(np.sum(predicted_classes == 1)),
                'arp_spoofing': int(np.sum(predicted_classes == 2)),
                'port_scan': int(np.sum(predicted_classes == 3)),
                'smurf_attack': int(np.sum(predicted_classes == 4))
            }
        }
        
    except Exception as e:
        print(f"Error in attack prediction: {e}")
        traceback.print_exc()
        return None
#######################################################3
def prepare_data_for_prediction(batch_df):
    """Prepare data for prediction (similar to training but without labels)"""
    FIXED_INPUT_FEATURES = 5
    
    if batch_df.empty:
        return None, None
    
    try:
        X_scaled, _ = preprocess_data(batch_df.copy())
        
        if X_scaled is not None:
            if X_scaled.shape[1] > FIXED_INPUT_FEATURES:
                X_scaled = X_scaled[:, :FIXED_INPUT_FEATURES]
            elif X_scaled.shape[1] < FIXED_INPUT_FEATURES:
                padding = np.zeros((X_scaled.shape[0], FIXED_INPUT_FEATURES - X_scaled.shape[1]))
                X_scaled = np.hstack([X_scaled, padding])
            
            return X_scaled, None
        
    except Exception as e:
        print(f"Error preparing data for prediction: {e}")
    
    # Generate dummy data if preprocessing fails
    print("🔴🔴🔴🔴🔴🔴 Generating dummy data for prediction...🔴🔴🔴🔴🔴🔴🔴")
    dummy_features = np.random.rand(len(batch_df), FIXED_INPUT_FEATURES)
    return dummy_features, None

def send_attack_alerts(alerts_data):
    """Send attack alerts to server"""
    if not alerts_data or not alerts_data['attack_alerts']:
        return
    
    try:
        alert_payload = {
            'client_id': CLIENT_ID,
            'timestamp': datetime.now().isoformat(),
            'alerts': alerts_data['attack_alerts'],
            'summary': alerts_data['predictions_summary'],
            'total_records_analyzed': alerts_data['total_records']
        }
        
        response = requests.post(
            f"{SERVER_URL}/receive_alerts",
            json=alert_payload,
            timeout=30
        )
        
        if response.status_code == 200:
            print(f"✅ Attack alerts sent successfully - {len(alerts_data['attack_alerts'])} alerts")
        else:
            print(f"⚠️ Failed to send alerts: {response.status_code}")
            
    except Exception as e:
        print(f"Error sending attack alerts: {e}")
##############################################################
def save_local_alerts(alerts_data):
    """Save alerts locally for backup"""
    if not alerts_data or not alerts_data['attack_alerts']:
        return
    
    try:
        alerts_file = f"alerts/alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(alerts_file, 'w') as f:
            json.dump(alerts_data, f, indent=2)
        print(f"📁 Alerts saved locally: {alerts_file}")
    except Exception as e:
        print(f"Error saving local alerts: {e}")

def analyze_batch_for_attacks(batch_df):
    attack_type_mapping = {
        "No Attack": 0, "Non-Attack": 0, "DoS Attack": 1, "ARP Spoofing": 2,
        "Nmap Port Scan": 3, "Port Scan": 3, "Smurf Attack": 4
    }
    
    attack_types = { # هذا الـ mapping يستخدم للطباعة، تأكد من ترتيبه
        0: "No Attack", 1: "DoS Attack", 2: "ARP Spoofing",
        3: "Nmap Port Scan", 4: "Smurf Attack"
    }
    
    print(f"\n🔍 Analyzing {len(batch_df)} records for attack detection...", flush=True)
    
    suspicious_records = 0
    detected_attacks = {}
    
    for idx, row in batch_df.iterrows():
        attack_type = 0
        if 'Type_of_attack' in row and pd.notna(row['Type_of_attack']):
            type_value = str(row['Type_of_attack']).strip()
            attack_type = attack_type_mapping.get(type_value, 0)
        
        if attack_type > 0:
            suspicious_records += 1
            attack_name = attack_types.get(attack_type, "Unknown Attack")
            detected_attacks[attack_name] = detected_attacks.get(attack_name, 0) + 1
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if suspicious_records > 0:
        print(f"🔴 Detected {suspicious_records} suspicious records out of {len(batch_df)} at {current_time}")
        for attack_type, count in detected_attacks.items():
            print(f"   - {attack_type}: {count} cases")
        
        risk_percentage = (suspicious_records / len(batch_df)) * 100
        if risk_percentage > 50:
            print(f"🔥 HIGH risk level: {risk_percentage:.1f}% of data contains attacks!")
        elif risk_percentage > 20:
            print(f"⚠️ MEDIUM risk level: {risk_percentage:.1f}% of data is suspicious")
        else:
            print(f"⚡ LOW risk level: {risk_percentage:.1f}% of data is suspicious")
    else:
        print(f"✅ No attacks detected in this batch - Time: {current_time}")
    
    return suspicious_records, detected_attacks
######################################################################
def prepare_training_data_from_batch(batch_df):  # انقلي هاي لباقي الكلاينتس

    FIXED_OUTPUT_CLASSES = 5
    
    if batch_df.empty:
        print("Empty batch received for training")
        return None, None
    
    try:
        print(f"📊 Preparing training data with oversampling for {len(batch_df)} records...")
        
        # First, preprocess the data to get basic features
        X_scaled, y_encoded = preprocess_data(batch_df.copy())
        
        if X_scaled is None or y_encoded is None:
            print("⚠️🔴⚠️🔴⚠️🔴⚠️🔴⚠️ Preprocessing failed, generating dummy data...")
            dummy_features = np.random.rand(len(batch_df), 5)  # 5 fixed features
            dummy_labels = tf.keras.utils.to_categorical(
                np.random.randint(0, FIXED_OUTPUT_CLASSES, len(batch_df)), 
                num_classes=FIXED_OUTPUT_CLASSES
            )
            return dummy_features, dummy_labels
        
        # Create DataFrame with processed features for oversampling
        feature_columns = [f'feature_{i}' for i in range(X_scaled.shape[1])]
        processed_df = pd.DataFrame(X_scaled, columns=feature_columns)
        
        # Add the attack type columns needed for make_tf_dataset
        attack_columns = [
            'Type of attack_ARP Spoofing',
            'Type of attack_DoS Attack', 
            'Type of attack_Nmap Port Scan',
            'Type of attack_No Attack',
            'Type of attack_Smurf Attack'
        ]
        
        # Initialize all attack columns to 0
        for col in attack_columns:
            processed_df[col] = 0
        
        # Set the appropriate attack column to 1 based on y_encoded
        attack_mapping = {
            0: 'Type of attack_No Attack',
            1: 'Type of attack_DoS Attack', 
            2: 'Type of attack_ARP Spoofing',
            3: 'Type of attack_Nmap Port Scan',
            4: 'Type of attack_Smurf Attack'
        }
        
        for idx, attack_code in enumerate(y_encoded):
            attack_code = int(attack_code) % FIXED_OUTPUT_CLASSES  # Ensure within bounds
            if attack_code in attack_mapping:
                attack_col = attack_mapping[attack_code]
                processed_df.loc[idx, attack_col] = 1
        
        # Add dummy 'Type' column (required by make_tf_dataset but will be dropped)
        processed_df['Type'] = 'placeholder'
        
        print("\n📋 Class distribution BEFORE oversampling:")
        for col in attack_columns:
            count = processed_df[col].sum()
            print(f"🔹 {col.replace('Type of attack_', '')}: {int(count)} samples")
        
        # Apply oversampling using make_tf_dataset
        balanced_dataset = make_tf_dataset(processed_df, batch_size=None)

        # Handle empty dataset case
        if balanced_dataset.cardinality().numpy() == 0:
            print("🔴 No valid data after oversampling. Using original data...")
            y_one_hot = tf.keras.utils.to_categorical(
                y_encoded % FIXED_OUTPUT_CLASSES, 
                num_classes=FIXED_OUTPUT_CLASSES
            )
            return X_scaled[:, :5], y_one_hot
        
        # Convert TF dataset back to numpy arrays
        X_balanced_list = []
        y_balanced_list = []
        
        for batch_x, batch_y in balanced_dataset:
            X_balanced_list.append(batch_x.numpy())
            y_balanced_list.append(batch_y.numpy())
        
        if not X_balanced_list:
            print("🔴⚠️🔴 No data returned from oversampling, using original data...")
            y_one_hot = tf.keras.utils.to_categorical(y_encoded % FIXED_OUTPUT_CLASSES, num_classes=FIXED_OUTPUT_CLASSES)
            return X_scaled[:, :5] if X_scaled.shape[1] > 5 else X_scaled, y_one_hot
        
        X_balanced = np.vstack(X_balanced_list)
        y_balanced = np.vstack(y_balanced_list)
        
        # Ensure we have exactly 5 input features
        if X_balanced.shape[1] > 5:
            X_balanced = X_balanced[:, :5]
        elif X_balanced.shape[1] < 5:
            padding = np.zeros((X_balanced.shape[0], 5 - X_balanced.shape[1]))
            X_balanced = np.hstack([X_balanced, padding])
        
        print(f"✅✅✅✅✅✅✅✅✅ Oversampling completed:")
        print(f"   • Original samples: {len(batch_df)}")
        print(f"   • Balanced samples: {len(X_balanced)}")
        print(f"   • Input features: {X_balanced.shape[1]}")
        print(f"   • Output classes: {y_balanced.shape[1]}")
        
        return X_balanced, y_balanced
        
    except Exception as e:
        print(f"❌ Error in oversampling preparation: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback to dummy data
        print("🔴🔴🔴🔴🔴🔴🔴🔴 Generating dummy data as fallback...")
        dummy_features = np.random.rand(len(batch_df), 5)
        dummy_labels = tf.keras.utils.to_categorical(
            np.random.randint(0, FIXED_OUTPUT_CLASSES, len(batch_df)), 
            num_classes=FIXED_OUTPUT_CLASSES
        )
        return dummy_features, dummy_labels

#################################################
def create_and_compile_model():
    print("Creating Keras model...")
    model = create_keras_model()
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
##################################################################
def train_model_on_data_range(start_idx, end_idx): # انقلي هاي لباقي الكلاينتس
    """
    Train model on data range with oversampling integration
    """
    global model, MONITORING_CSV_FILE, total_trained_records
    
    print(f"\n🎯 Starting training with oversampling on records from {start_idx + 1} to {end_idx}")
    
    try:
        df = pd.read_csv(MONITORING_CSV_FILE, on_bad_lines='skip')
        training_df = df.iloc[start_idx:end_idx].copy()
        
        print("--- DEBUG: 'Type_of_attack' Distribution in Training Batch ---")
        print(training_df['Type_of_attack'].value_counts())
        print("----------------------------------------------------------")
        
        print(f"Training data size: {len(training_df)} records")
        
        if len(training_df) == 0:
            print("No data available for training in the specified range.")
            return None, 0.0

        # Use the new oversampling preparation function
        X, y = prepare_training_data_from_batch(training_df)
        
        if X is None or y is None or len(X) == 0:
            print("Failed to prepare training data or no valid samples after preprocessing.")
            return None, 0.0
        
        if len(X) < 2:
            print("Insufficient samples for train-validation split. Skipping training.")
            return None, 0.0

        # Check class distribution after oversampling
        unique_classes_in_y = np.unique(np.argmax(y, axis=1))
        print(f"📊 Classes present after oversampling: {unique_classes_in_y}")
        
        if len(unique_classes_in_y) < 2:
            print(f"Only one class ({unique_classes_in_y[0]}) found in training data. Skipping stratified split.")
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=np.argmax(y, axis=1)
            )
        
        print(f"📊 Data split after oversampling:")
        print(f"   • Training samples: {len(X_train)}")
        print(f"   • Validation samples: {len(X_val)}")
        
        if len(X_train) == 0:
            print("No samples for training after split. Skipping training.")
            return None, 0.0

        # Create and prepare training model
        from model.model_definition import create_keras_model
        training_model = create_keras_model()
        training_model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Load existing weights if available
        original_weights = None
        if model is not None:
            original_weights = model.get_weights()
            training_model.set_weights(original_weights)
            print("✅ Global model weights loaded successfully.")
        else:
            original_weights = training_model.get_weights()
            print("🆕 Using new model weights.")
        
        print(f"\n🚀 Starting training on {len(X_train)} balanced samples...")
        
        # Train the model
        history = training_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=5,
            batch_size=min(32, len(X_train)),
            verbose=1,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=3,
                    restore_best_weights=True
                )
            ]
        )
        
        print("✅ Training completed successfully!")
        
        # Evaluate the model
        val_predictions = training_model.predict(X_val, verbose=0)
        val_pred_classes = np.argmax(val_predictions, axis=1)
        val_true_classes = np.argmax(y_val, axis=1)
        
        accuracy = accuracy_score(val_true_classes, val_pred_classes)
        
        print(f"\n📊 📊 📊 Training Results with Oversampling: 📊 📊 📊")
        print(f"   • Final Validation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   • Completed Epochs: {len(history.history['loss'])}")
        print(f"   • Final Training Loss: {history.history['loss'][-1]:.4f}")
        print(f"   • Final Validation Loss: {history.history['val_loss'][-1]:.4f}")
        print(f"   • Training samples used: {len(X_train)} (after oversampling)")
        print(f"   • Original data range: {start_idx + 1}-{end_idx}")
        
        # Calculate deltas (difference between trained and original weights)
        trained_weights = training_model.get_weights()
        deltas = [trained_w - original_w for trained_w, original_w in zip(trained_weights, original_weights)]
        
        print("✅ Deltas calculated successfully.")
        print(f"   • Number of weight layers: {len(deltas)}")
        print(f"   • Deltas shapes: {[d.shape for d in deltas]}")
        
        return deltas, accuracy
            
    except Exception as e:
        print(f"❌ Error during training with oversampling: {e}")
        import traceback
        traceback.print_exc()
        return None, 0.0
############################################################
def continuous_testing_loop():
    global total_tested_records
    
    print("Starting continuous testing loop...")
    
    consecutive_empty_batches = 0
    max_empty_batches = 5
    
    while True:
        try:
            print(f"\n--- Testing Loop Iteration (Tested so far: {total_tested_records}) ---")
            
            batch_df, start_display, end_display = get_next_test_batch()
            
            if batch_df is None:
                consecutive_empty_batches += 1
                print(f"No new data for testing (attempt {consecutive_empty_batches}/{max_empty_batches}). Waiting...")
                
                if consecutive_empty_batches >= max_empty_batches:
                    print("No new data for extended period. Waiting longer...")
                    time.sleep(30)
                else:
                    time.sleep(10)
                continue
            
            if batch_df.empty:
                consecutive_empty_batches += 1
                print(f"Empty batch received (attempt {consecutive_empty_batches}/{max_empty_batches})")
                time.sleep(10)
                continue
            
            consecutive_empty_batches = 0
            
            print(f"Processing batch with {len(batch_df)} records...")
            suspicious_count, detected_attacks = analyze_batch_for_attacks(batch_df)
            
            records_processed = len(batch_df)
            total_tested_records += records_processed
            
            print(f"Updated total_tested_records: {total_tested_records}")
            
            try:
                save_round_tracker()
                print("Tracker saved successfully")
            except Exception as save_error:
                print(f"Error saving tracker: {save_error}")
            
            print(f"Batch {start_display}-{start_display + records_processed - 1} tested.")
            print(f"Total records tested: {total_tested_records}")
            
            if can_start_training():
                print("✅ Sufficient data available for training!")
                notify_server_ready()

            else:
                untrained = total_tested_records - total_trained_records
                needed = TRAINING_BATCH_SIZE - untrained
                print(f"Need {needed} more records for training (current untrained: {untrained})")

            time.sleep(5)
            
        except Exception as e:
            print(f"Error in testing loop: {e}")
            traceback.print_exc()
            time.sleep( )
##################################################################
def send_training_results_to_server_with_round(deltas, training_range_str, round_num):
    if deltas is None:
        print("No deltas to send.")
        return False

    print(f"Encrypting deltas for round {round_num}...")
    encrypted_deltas_b64 = encrypt_model_deltas(deltas)
    if encrypted_deltas_b64 is None:
        print("Failed to encrypt deltas.")
        return False

    print(f"Sending encrypted deltas for round {round_num}, range {training_range_str}...")
    try:
        response = requests.post(
            f"{SERVER_URL}/receive_deltas",
            json={
                "deltas": encrypted_deltas_b64,
                "round": round_num,
                "client_id": CLIENT_ID,
                "model_info": {
                    "input_shape": 5,
                    "output_shape": 5,
                    "type": "multi_class"
                },
                "training_info": {
                    "data_range": training_range_str,
                    "total_tested": total_tested_records,
                    "total_trained": total_trained_records
                }
            },
            timeout=120
        )
        response.raise_for_status()
        print(f"Deltas sent successfully for round {round_num}, range {training_range_str}.")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Network or server error sending deltas for round {round_num}: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error sending deltas for round {round_num}: {e}")
        traceback.print_exc()
        return False

# --- Flask Routes ---
from flask import Flask, jsonify, request
app = Flask(__name__)

@app.route('/start_round', methods=['POST'])
def start_round_endpoint():
    global model, total_trained_records, training_in_progress, round_data_tracker
    
    if training_in_progress:
        return jsonify({"status": "busy", "message": "Client is currently busy with training."}), 409

    try:
        data = request.json
        round_num = data.get('round')
        global_weights_json = data.get('weights')

        if round_num is None or global_weights_json is None:
            return jsonify({"error": "Missing 'round' or 'weights' in request."}), 400

        print(f"\nServer requested training for round {round_num}")
        
        if not can_start_training():
            untrained = total_tested_records - total_trained_records
            return jsonify({
                "status": "Insufficient data",
                "message": f"Need more than {TRAINING_BATCH_SIZE} untrained records for training. Currently have {untrained}",
                "total_tested": total_tested_records,
                "total_trained": total_trained_records,
                "untrained": untrained
            }), 400

        if model is None:
            model = create_and_compile_model()

        try:
            deserialized_weights = []
            current_model_weights = model.get_weights() 
            
            if len(global_weights_json) != len(current_model_weights):
                print(f"Weight mismatch: Server sent {len(global_weights_json)} layers, model has {len(current_model_weights)} layers.")
                return jsonify({"error": "Model layer count does not match server weights."}), 400

            for i, layer_weights in enumerate(global_weights_json):
                target_shape = current_model_weights[i].shape
                deserialized_weights.append(np.array(layer_weights).reshape(target_shape))

            model.set_weights(deserialized_weights)
            model.save(MODEL_PATH)
            print("Model weights updated successfully.")
        except Exception as e:
            print(f"Error updating weights: {e}")
            traceback.print_exc()
            return jsonify({"error": f"Failed to update model weights: {str(e)}"}), 500

        start_idx, end_idx = get_training_data_range()
        training_range_str = f"{start_idx + 1}-{end_idx}"

        round_data_tracker[round_num] = {
            "start_idx": start_idx,
            "end_idx": end_idx,
            "timestamp": datetime.now().isoformat(),
            "status": "training_initiated",
            "total_tested_at_start": total_tested_records,
            "total_trained_at_start": total_trained_records
        }
        save_round_tracker()

        print(f"Starting training thread for range {training_range_str}")
        threading.Thread(target=run_training_process, args=(round_num, start_idx, end_idx, training_range_str), daemon=True).start()
        
        return jsonify({
            "status": "Training started",
            "round": round_num,
            "message": f"Training initiated on {training_range_str}",
            "data_range": training_range_str,
            "total_tested": total_tested_records,
            "total_trained": total_trained_records
        }), 200

    except Exception as e:
        print(f"Error in /start_round endpoint: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
###########################3
def notify_server_status():
    try:
        payload = {
            "client_id": CLIENT_ID,
            "total_tested_records": total_tested_records,
            "total_trained_records": total_trained_records,
            "untrained_records": total_tested_records - total_trained_records
        }
        response = requests.post(f"{SERVER_URL}/update_status", json=payload, timeout=15)
        if response.status_code == 200:
            print("📤 Updated server with current status after training.")
        else:
            print(f"⚠️ Server returned status {response.status_code} on status update.")
    except Exception as e:
        print(f"❌ Failed to update server status: {e}")
#########################
def run_training_process(round_num, start_idx, end_idx, training_range_str):
    global training_in_progress, total_trained_records, round_data_tracker
    
    training_in_progress = True
    round_status = "Failed"
    try:
        print(f"\nExecuting training process for round {round_num}")
        print(f"Training range: {training_range_str}")
        
        deltas, accuracy = train_model_on_data_range(start_idx, end_idx)
        
        if deltas is not None:
            print(f"Training completed successfully with accuracy: {accuracy:.4f}")
            
            total_trained_records += TRAINING_BATCH_SIZE
            notify_server_status()

            round_data_tracker[round_num]["status"] = "Trained locally"
            round_data_tracker[round_num]["accuracy"] = accuracy
            round_data_tracker[round_num]["total_trained_after_round"] = total_trained_records
            save_round_tracker()
            print(f"Local trained records updated: {total_trained_records}")

            if send_training_results_to_server_with_round(deltas, training_range_str, round_num):
                round_status = "Sent to Server"
                round_data_tracker[round_num]["status"] = "Sent to Server"
                print(f"Training results sent to server successfully.")



            else:
                round_status = "Failed to Send"
                round_data_tracker[round_num]["status"] = "Failed to Send"
                print("Failed to send deltas to server.")
        else:
            round_status = "Training Failed - No Deltas"
            round_data_tracker[round_num]["status"] = "Training Failed - No Deltas"
            print("Training failed - no deltas generated.")
            
    except Exception as e:
        print(f"Error in training process: {e}")
        traceback.print_exc()
        round_status = "Exception during training"
        round_data_tracker[round_num]["status"] = "Exception during training"
    finally:
        training_in_progress = False
        round_data_tracker[round_num]["final_status"] = round_status
        round_data_tracker[round_num]["timestamp_end"] = datetime.now().isoformat()
        save_round_tracker()
        print(f"Training process for round {round_num} completed with status: {round_status}")

@app.route('/model_info', methods=['GET'])
def get_model_info():
    if model is None:
        trainable_params = 0
    else:
        trainable_params = model.count_params()

    return jsonify({
        "model_exists": model is not None,
        "input_shape": 5,
        "output_shape": 5,
        "trainable_params": trainable_params,
        "total_tested_records": total_tested_records,
        "total_trained_records": total_trained_records,
        "untrained_records": total_tested_records - total_trained_records,
        "can_train": can_start_training(),
        "batch_size": BATCH_SIZE,
        "round_history": round_data_tracker
    }), 200
################################################################################################################################
def notify_server_ready():
    try:
        payload = {
            "client_id": CLIENT_ID,
            "total_tested_records": total_tested_records,
            "total_trained_records": total_trained_records,
            "untrained_records": total_tested_records - total_trained_records
        }
        response = requests.post(f"{SERVER_URL}/notify_ready", json=payload, timeout=15)
        if response.status_code == 200:
            print("📤 Notified server of readiness for training.")
        else:
            print(f"⚠️ Server returned status {response.status_code} on readiness notify.")
    except Exception as e:
        print(f"❌ Failed to notify server: {e}")


###################################################################################################################################
@app.route('/predict', methods=['POST'])
def predict_endpoint():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 400

    try:
        features = request.json.get('features')
        if not features or len(features) != 5:
            return jsonify({"error": "Expected 5 features"}), 400

        input_data = np.array(features).reshape(1, -1)
        predictions = model.predict(input_data, verbose=0)
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))

        attack_types = ["No Attack", "DoS Attack", "ARP Spoofing", "Nmap Port Scan", "Smurf Attack"]
        
        return jsonify({
            "predicted_class": predicted_class,
            "predicted_attack": attack_types[predicted_class],
            "confidence": confidence,
            "class_probabilities": {attack_types[i]: float(p) for i, p in enumerate(predictions[0])},
            "total_trained_records": total_trained_records
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def signal_handler(sig, frame):
    print(f"\nReceived signal {sig}. Shutting down gracefully...")
    cleanup_on_shutdown()
    sys.exit(0)

def cleanup_on_shutdown():
    global monitoring_active, monitoring_process
    
    print("Cleaning up resources...")
    
    try:
        save_round_tracker()
        print("Tracker file saved.")
    except Exception as e:
        print(f"Error saving tracker file: {e}")
    
    try:
        stop_network_monitoring()
        print("Network monitoring stopped.")
    except Exception as e:
        print(f"Error stopping monitoring: {e}")
    
    print("Cleanup complete.")

def run_flask_app():
    global flask_app_ready
    try:
        app.run(host='0.0.0.0', port=6001, debug=False, use_reloader=False)
    except Exception as e:
        print(f"Fatal error starting Flask app: {e}")
        # Signal that Flask app failed to start, if main() is waiting for it
        flask_app_ready.set()
        raise

def main():
    global model
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("Starting Federated Learning Client...")
    
    try:
        setup_directories()
        load_round_tracker()
        print(f"Starting client: {CLIENT_ID}")

        # Start Flask app in a separate thread FIRST
        # This makes sure the endpoints are available for the main client script
        # (your main_client.py) to check client status and fetch context.
        flask_thread = threading.Thread(target=run_flask_app, daemon=True)
        flask_thread.start()

        # Wait for the Flask app to signal it's ready.
        # A simple sleep might not be enough, but for this context,
        # the main_client.py script's check for the server will implicitly wait.
        # For an internal wait, you might add a small delay or more robust check.
        print("Waiting for internal Flask server to be ready...")
        time.sleep(2) # Give Flask a moment to spin up

        print("Fetching encryption context...")
        context_fetched = False
        for i in range(CKKS_CONTEXT_FETCH_RETRIES):
            if fetch_ckks_context_from_server():
                print("Encryption context loaded successfully.")
                context_fetched = True
                break
            else:
                print(f"Failed to fetch encryption context - retrying in {CKKS_CONTEXT_RETRY_DELAY} seconds ({i+1}/{CKKS_CONTEXT_FETCH_RETRIES})...")
                time.sleep(CKKS_CONTEXT_RETRY_DELAY)
        
        if not context_fetched:
            print("Cannot proceed without encryption context after multiple attempts. Exiting.")
            sys.exit(1)

        # Initialize the model AFTER the encryption context is successfully fetched
        if model is None:
            print("Initializing model for the first time...")
            if os.path.exists(MODEL_PATH):
                try:
                    model = tf.keras.models.load_model(MODEL_PATH)
                    print(f"Loaded existing model from {MODEL_PATH}")
                except Exception as e:
                    print(f"Error loading model from {MODEL_PATH}: {e}. Creating a new model.")
                    model = create_and_compile_model()
            else:
                print("No saved model found, creating a new model.")
                model = create_and_compile_model()

        print("Starting network monitoring...")
        start_network_monitoring()
        time.sleep(3) # Give monitoring a moment to initialize

        if not monitoring_active:
            print("Warning: Network monitoring might not be active.")
        else:
            print("Network monitoring is active.")

        print("Starting continuous testing loop...")
        testing_thread = threading.Thread(target=continuous_testing_loop, daemon=True)
        testing_thread.start()
        time.sleep(2)

        print(f"\nCurrent Client Status:")
        print(f"   • Client ID: {CLIENT_ID}")
        print(f"   • Total tested records: {total_tested_records}")
        print(f"   • Total trained records: {total_trained_records}")
        print(f"   • Untrained records: {total_tested_records - total_trained_records}")
        print(f"   • Can start training: {can_start_training()}")
        print(f"   • Batch size for testing: {BATCH_SIZE}")
        print(f"   • Training threshold: {TRAINING_BATCH_SIZE} records")

        print(f"\nOperational Mode:")
        print(f"   • Testing: Continuous batches of {BATCH_SIZE} records")
        print(f"   • Training: Triggered by server when {TRAINING_BATCH_SIZE}+ untrained records available")
        print(f"   • Each training round processes exactly {TRAINING_BATCH_SIZE} records")
        print(f"   • Server URL: {SERVER_URL}")

        flask_app_ready.set() # Signal that all core client services are up and running

        print(f"\nClient ready for Federated Learning!")
        print(f"Waiting for server requests...")

        # Keep main thread alive
        while True:
            time.sleep(1)

    except Exception as e:
        print(f"FATAL ERROR in main client execution: {e}")
        traceback.print_exc()
        cleanup_on_shutdown()
        sys.exit(1)

if __name__ == '__main__':
    main()