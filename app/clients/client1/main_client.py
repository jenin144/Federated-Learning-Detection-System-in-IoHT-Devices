#!/usr/bin/env python3
"""
Main Client Script - Updated Version
يدير تشغيل عميل التعلم المتوزع مع المونيترنج المستمر
"""

import subprocess
import sys
import os
import time
import signal
import threading
import requests
from pathlib import Path
import json
from flask import Flask, render_template, request
from multiprocessing import Process
SERVER_URL = "http://host.docker.internal:5000"
app = Flask(__name__, template_folder="templates", static_folder="static")


#--------------------web-------------
from flask import session
app.secret_key = 'my_secret_key' 


@app.route('/')
def home():
    return render_template("index.html", message="")

@app.route('/about.html')
def about():
    return render_template("about.html")

@app.route('/signup', methods=['POST'])
def signup():
    username = request.form.get('username')
    password = request.form.get('password')
    device_type = request.form.get('device_type')
    model_version = request.form.get('model_version')

    # If any required field is missing, redirect to /register
    if not username or not password or not device_type or not model_version:
        return render_template("index.html,", message="Please fill in all fields.")

    payload = {
        "username": username,
        "password": password,
        "device_type": device_type,
        "model_version": model_version
    }
    try:
        res = requests.post(f"{SERVER_URL}/register_client", json=payload)
        res_json = res.json()

        if res.status_code in [200, 201]:
            return render_template("index.html", message=f"✅ {res_json.get('message', 'Registered successfully')}")
        else:
            return render_template("index.html", message=f"❌ {res_json.get('message', res_json.get('error', 'Registration failed.'))}")
    except Exception as e:
        return render_template("index.html", message=f"❌ Error: {e}")



@app.route('/signin', methods=['POST'])
def signin():
    username = request.form['username']
    password = request.form['password']

    try:
        res = requests.post(f"{SERVER_URL}/login_client", json={"username": username, "password": password})
        if res.status_code == 200:
            # Fetch client info
            info_res = requests.post(f"{SERVER_URL}/get_client_info", json={"username": username})
            if info_res.status_code == 200:
                info = info_res.json()
                # حفظ البيانات في session
                session['username'] = username
                session['device_type'] = info.get("device_type", "")
                session['model_version'] = info.get("model_version", "")
                session['client_ip'] = info.get("client_ip", "")
                session['registered_at'] = info.get("registered_at", "")
                return render_template(
                    "welcome.html",
                    username=username,
                    device_type=session['device_type'],
                    model_version=session['model_version'],
                    client_ip=session['client_ip'],
                    registered_at=session['registered_at']
                )
            else:
                return render_template("welcome.html", username=username, device_type="", model_version="", client_ip="", registered_at="")
        else:
            return render_template("index.html", message="❌ Invalid credentials.")
    except Exception as e:
        return render_template("index.html", message=f"❌ Error: {e}")

@app.route('/welcome')
def welcome():
    return render_template(
        "welcome.html",
        username=session.get("username", ""),
        device_type=session.get("device_type", ""),
        model_version=session.get("model_version", ""),
        client_ip=session.get("client_ip", ""),
        registered_at=session.get("registered_at", "")
    )

@app.route('/notifications')
def notifications():
    notifications_list = []
    try:
        if os.path.exists("notifications.log"):
            with open("notifications.log", "r") as f:
                notifications_list = f.readlines()
    except Exception as e:
        notifications_list = [f"⚠️ Error reading notifications: {e}"]

    return render_template(
        "notifications.html",
        username=session.get("username", ""),
        notifications=notifications_list
    )

@app.route('/notification_count')
def notification_count():
    try:
        if os.path.exists("notifications.log"):
            with open("notifications.log", "r") as f:
                lines = f.readlines()
                return {"count": len(lines)}
    except:
        return {"count": 0}
    return {"count": 0}


#---------------end of web-----------------------

def start_web_interface():
    app.run(host="0.0.0.0", port=6101, debug=False, use_reloader=False)
class FederatedLearningClient:
    def __init__(self, client_id="client1"):
        self.client_id = client_id
        self.local_train_process = None
        self.running = True
        self.server_url = "http://host.docker.internal:5000"
        
        
    def setup_environment(self):
        """إعداد البيئة والمجلدات المطلوبة"""
        print("🔧 Setting up client environment...")
        
        # إنشاء المجلدات المطلوبة
        directories = ['model', 'data']
        for dir_name in directories:
            Path(dir_name).mkdir(exist_ok=True)
            print(f"📁 Created/verified directory: {dir_name}")
        
        # التحقق من وجود الملفات المطلوبة
        required_files = [
            'local_train.py',
            'utils/oversampling.py',
            'model/model_definition.py'
        ]
        
        missing_files = []
        for file_path in required_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        if missing_files:
            print("❌ Missing required files:")
            for file_path in missing_files:
                print(f"   - {file_path}")
            print("\nPlease ensure all required files are present before running the client.")
            return False
        
        print("✅ Environment setup completed")
        return True
    
    def wait_for_server(self, max_retries=30, delay=10):
        """انتظار السيرفر حتى يصبح جاهزاً"""
        print("⏳ Waiting for federated learning server to be ready...")
        
        for attempt in range(max_retries):
            try:
                # محاولة الاتصال بالسيرفر
                response = requests.get(f"{self.server_url}/status", timeout=5)
                if response.status_code == 200:
                    print("✅ Federated learning server is ready!")
                    return True
                else:
                    print(f"🔄 Server responded with status: {response.status_code}")
                    
            except requests.exceptions.ConnectionError:
                print(f"🔄 Attempt {attempt + 1}/{max_retries}: Server not reachable")
            except requests.exceptions.Timeout:
                print(f"🔄 Attempt {attempt + 1}/{max_retries}: Server timeout")
            except Exception as e:
                print(f"🔄 Attempt {attempt + 1}/{max_retries}: Error - {e}")
            
            if attempt < max_retries - 1:
                print(f"   Retrying in {delay} seconds...")
                time.sleep(delay)
        
        print("❌ Server failed to become ready within timeout")
        print("   Please ensure the federated learning server is running")
        return False
    
    def start_local_training_service(self):
        """بدء خدمة التدريب المحلي"""
        try:
            print("🚀 Starting local training service...")
            print("   This service will:")
            print("   • Start continuous network monitoring")
            print("   • Wait for training requests from server")
            print("   • Train on collected network data")
            print("   • Send encrypted model updates back to server")
            
            # بدء local_train.py كخدمة
            self.local_train_process = subprocess.Popen(
                [sys.executable, "local_train.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # تشغيل thread لقراءة ومعالجة output
            threading.Thread(
                target=self._handle_service_output,
                args=(self.local_train_process,),
                daemon=True
            ).start()
            
            # التحقق من أن الخدمة بدأت بنجاح
            time.sleep(3)
            if self.local_train_process.poll() is not None:
                print("❌ Local training service failed to start")
                return False
            
            print("✅ Local training service started successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start local training service: {e}")
            return False
    
    def _handle_service_output(self, process):
        """التعامل مع output الخدمة"""
        try:
            while self.running and process.poll() is None:
                line = process.stdout.readline()
                if line:
                    # تنسيق الرسائل
                    line = line.rstrip()
                    # إذا كان فيه ديتكت أتاك، نحفظها كإشعار
                    if "Detected" in line and "suspicious" in line:
                        self.save_notification(line)

                    if any(keyword in line for keyword in ['🚨', '❌', 'ERROR', 'CRITICAL']):
                        print(f"🔴 {line}")
                    elif any(keyword in line for keyword in ['✅', '🎯', '📊']):
                        print(f"🟢 {line}")
                    elif any(keyword in line for keyword in ['⚠️', 'WARNING']):
                        print(f"🟡 {line}")
                    else:
                        print(f"ℹ️  {line}")
                        
        except Exception as e:
            print(f"❌ Error handling service output: {e}")


    def save_notification(self, message):
        """حفظ إشعار جديد في ملف notifications"""
        try:
            with open("notifications.log", "a") as f:
                f.write(message + "\n")
        except Exception as e:
            print(f"⚠️ Failed to write notification: {e}")

    def monitor_service_health(self):
        """مراقبة صحة الخدمة"""
        print("👁️ Starting service health monitor...")
        
        while self.running:
            try:
                # فحص حالة عملية التدريب المحلي
                if self.local_train_process:
                    if self.local_train_process.poll() is not None:
                        return_code = self.local_train_process.returncode
                        if return_code != 0:
                            print(f"❌ Local training service crashed with code: {return_code}")
                            print("🔄 Attempting to restart service...")
                            if self.restart_service():
                                print("✅ Service restarted successfully")
                            else:
                                print("❌ Failed to restart service")
                                break
                        else:
                            print("ℹ️ Local training service ended normally")
                            break
                
                # فحص دوري كل 10 ثوان
                time.sleep(10)
                
            except Exception as e:
                print(f"❌ Error in service health monitoring: {e}")
                time.sleep(30)  # انتظار أطول عند الأخطاء
    
    def restart_service(self):
        """إعادة تشغيل الخدمة"""
        try:
            print("🔄 Attempting to restart local training service...")
            
            # تنظيف العملية القديمة
            if self.local_train_process:
                self.cleanup_process(self.local_train_process)
            
            # انتظار قصير
            time.sleep(5)
            
            # إعادة تشغيل الخدمة
            return self.start_local_training_service()
            
        except Exception as e:
            print(f"❌ Error restarting service: {e}")
            return False
    
    def cleanup_process(self, process):
        """تنظيف عملية معينة"""
        if process and process.poll() is None:
            try:
                print("🧹 Cleaning up process...")
                process.terminate()
                process.wait(timeout=15)
                print("✅ Process terminated gracefully")
            except subprocess.TimeoutExpired:
                print("⚠️ Process didn't terminate gracefully, forcing...")
                process.kill()
                process.wait()
                print("✅ Process killed")
            except Exception as e:
                print(f"❌ Error cleaning up process: {e}")
    
    def cleanup_all(self):
        """تنظيف جميع الموارد"""
        print("🧹 Cleaning up all resources...")
        self.running = False
        
        if self.local_train_process:
            self.cleanup_process(self.local_train_process)
            print("✅ Local training service cleaned up")
    
    def signal_handler(self, signum, frame):
        """معالج إشارات النظام"""
        print(f"\n🛑 Received signal {signum}, shutting down client...")
        self.cleanup_all()
        print("👋 Client shutdown complete")
        sys.exit(0)
    
    def display_startup_info(self):
        """عرض معلومات بدء التشغيل"""
        print("🎯 Federated Learning Client")
        print("=" * 50)
        print(f"Client ID: {self.client_id}")
        print(f"Server URL: {self.server_url}")
        print(f"Working Directory: {os.getcwd()}")
        print("=" * 50)
    
    def display_status_info(self):
        """عرض معلومات الحالة"""
        print("\n🎉 Client is now running successfully!")
        print("📋 Current Status:")
        print("   🔍 Network monitoring: ACTIVE (continuous)")
        print("   📡 Server connection: CONNECTED")
        print("   🤖 ML training: WAITING for server requests")
        print("   🔒 Encryption: ENABLED (CKKS)")
        print("\n💡 The client will automatically:")
        print("   • Monitor network traffic in real-time")
        print("   • Detect various types of network attacks")
        print("   • Train ML models when requested by server")
        print("   • Use only NEW data for each training round")
        print("   • Send encrypted model updates to server")
        print("   • Participate in federated learning process")
        print("\n🛑 Press Ctrl+C to stop the client")
        print("-" * 50)
    
    def run(self):
        """تشغيل العميل الرئيسي"""
        threading.Thread(target=start_web_interface, daemon=True).start()
        try:
            self.display_startup_info()
            
            # إعداد معالجات الإشارات
            signal.signal(signal.SIGINT, self.signal_handler)
            signal.signal(signal.SIGTERM, self.signal_handler)
            
            # إعداد البيئة
            if not self.setup_environment():
                return False
            
            # انتظار السيرفر
            if not self.wait_for_server():
                print("💡 Tip: Make sure the federated learning server is running")
                return False
            
            # بدء خدمة التدريب المحلي
            if not self.start_local_training_service():
                return False
            
            # عرض معلومات الحالة
            self.display_status_info()
            
            # بدء مراقبة صحة الخدمة
            self.monitor_service_health()
            
            return True
            
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
            return True
        except Exception as e:
            print(f"❌ Unexpected error in client: {e}")
            return False
        finally:
            self.cleanup_all()

def print_usage():
    """طباعة معلومات الاستخدام"""
    print("🔧 Federated Learning Client")
    print("Usage: python main_client.py [client_id]")
    print("\nArguments:")
    print("  client_id    Optional client identifier (default: client1)")
    print("\nExamples:")
    print("  python main_client.py")
    print("  python main_client.py client2")
    print("\nThis client will:")
    print("• Monitor network traffic continuously")
    print("• Participate in federated learning rounds")
    print("• Send encrypted model updates to server")

def main():
    """الدالة الرئيسية"""
    # تحليل المعاملات
    client_id = "client1"
    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help']:
            print_usage()
            sys.exit(0)
        client_id = sys.argv[1]
    
    # إنشاء وتشغيل العميل
    client = FederatedLearningClient(client_id)
    
    try:
        success = client.run()
        exit_code = 0 if success else 1
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        exit_code = 1
    finally:
        client.cleanup_all()
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main()