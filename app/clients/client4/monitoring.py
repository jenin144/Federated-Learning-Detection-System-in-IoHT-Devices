#!/usr/bin/env python3
"""
Continuous Network Monitor - Standalone Version
يعمل بشكل مستمر ويحفظ البيانات في ملف CSV واحد
"""

from scapy.all import sniff, IP, TCP, UDP, ARP, ICMP, Ether
import time
from collections import defaultdict
import os
import argparse
import threading
import signal
import sys
from datetime import datetime

class ContinuousNetworkMonitor:
    def __init__(self, csv_file='network_traffic.csv'):
        self.csv_file = csv_file
        self.monitoring_active = False
        self.packet_count = 0
        self.attack_count = 0
        self.start_time = time.time()
        
        # Attack detection variables
        self.arp_spoofing_detected = {}
        self.dos_flood_counter = defaultdict(list)
        self.syn_scan_tracker = defaultdict(set)
        self.protocol_map = {1: 'ICMP', 6: 'TCP', 17: 'UDP'}
        
        # Thread lock للكتابة الآمنة
        self.write_lock = threading.Lock()
        
        # إعداد ملف CSV
        self.setup_csv()
        
    def setup_csv(self):
        """إعداد ملف CSV مع headers"""
        try:
            # إنشاء ملف جديد أو محو الموجود
            with open(self.csv_file, 'w') as f:
                f.write('No.,Time,Source,Destination,Protocol,Length,Info,Type,Type_of_attack\n')
            
            print(f"📄 CSV file initialized: {os.path.abspath(self.csv_file)}")
            
        except Exception as e:
            print(f"❌ Error setting up CSV: {e}")
            sys.exit(1)

    def detect_attacks(self, packet):
        """كشف الهجمات من الحزم"""
        attacks = []
        current_time = time.time()

        # ARP Spoofing Detection
        if packet.haslayer(ARP) and packet[ARP].op == 2:
            attacks.append("ARP Spoofing")

        # DoS Detection (TCP Flood)
        if packet.haslayer(TCP) and packet.haslayer(IP):
            src = packet[IP].src
            self.dos_flood_counter[src].append(current_time)
            self.dos_flood_counter[src] = [t for t in self.dos_flood_counter[src] if current_time - t <= 5]
            if len(self.dos_flood_counter[src]) > 5:
                attacks.append("DoS (TCP Flood)")
                self.dos_flood_counter[src].clear()

        # Smurf Attack Detection
        if packet.haslayer(ICMP) and packet.haslayer(IP):
            if packet[IP].dst.endswith('.255') or packet[IP].dst == '255.255.255.255':
                attacks.append("Smurf Attack")

        # Port Scan Detection (SYN Scan)
        if packet.haslayer(TCP) and packet[TCP].flags == 'S' and packet.haslayer(IP):
            src = packet[IP].src
            dst = packet[IP].dst
            port = packet[TCP].dport
            self.syn_scan_tracker[(src, dst)].add(port)
            if len(self.syn_scan_tracker[(src, dst)]) > 3:
                attacks.append("Port Scan (Nmap)")
                self.syn_scan_tracker[(src, dst)].clear()

        return ", ".join(attacks) if attacks else "No Attack"

    def process_packet(self, packet):
        """معالجة الحزمة وكتابتها في CSV"""
        try:
            self.packet_count += 1
            
            # استخراج معلومات البروتوكول والعناوين
            if packet.haslayer(IP):
                protocol = self.protocol_map.get(packet[IP].proto, 'Other')
                src = packet[IP].src
                dst = packet[IP].dst
            elif packet.haslayer(ARP):
                protocol = 'ARP'
                src = packet[ARP].psrc
                dst = packet[ARP].pdst
            else:
                protocol = 'Other'
                src = packet[Ether].src if packet.haslayer(Ether) else 'Unknown'
                dst = packet[Ether].dst if packet.haslayer(Ether) else 'Unknown'

            # كشف الهجمات
            attack_info = self.detect_attacks(packet)
            status = "Attack" if attack_info != "No Attack" else "Non-Attack"
            
            if status == "Attack":
                self.attack_count += 1
                print(f"🚨 Attack detected: {attack_info} from {src} to {dst}" )

            # تحضير بيانات الحزمة
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            packet_length = len(packet)
            packet_info = packet.summary().replace('"', "'").replace(',', ';')  # تجنب مشاكل CSV
            
            # كتابة البيانات في CSV
            with self.write_lock:
                with open(self.csv_file, 'a', buffering=1) as f:
                    f.write(f'{self.packet_count},{timestamp},{src},{dst},{protocol},'
                           f'{packet_length},"{packet_info}",{status},{attack_info}\n')

            # طباعة إحصائيات دورية
            if self.packet_count % 100 == 0:
                print(f"Captured {self.packet_count} packets so far...", flush=True)
                self.print_stats()

        except Exception as e:
            print(f"❌ Error processing packet: {e}")

    def print_stats(self):
        """طباعة إحصائيات المونيترنج"""
        duration = time.time() - self.start_time
        packets_per_sec = self.packet_count / duration if duration > 0 else 0
        
        print(f"📊 Stats: {self.packet_count} packets processed "
              f"({self.attack_count} attacks) - "
              f"{packets_per_sec:.1f} packets/sec")

    def start_monitoring(self):
        """بدء المونيترنج المستمر"""
        if self.monitoring_active:
            print("ℹ️ Monitoring is already running")
            return
        
        self.monitoring_active = True
        print("🔍 Starting continuous network monitoring...")
        print(f"📄 Data will be saved to: {self.csv_file}")
        print("🚨 Only attacks will be displayed in real-time")
        print("📊 Packet statistics will be shown every 100 packets")
        
        try:
            while self.monitoring_active:
                try:
                    # بدء جلسة التقاط لمدة 30 ثانية
                    sniff(prn=self.process_packet, 
                         timeout=30, 
                         stop_filter=lambda x: not self.monitoring_active)
                    
                    if self.monitoring_active:
                        print("🔄 Restarting packet capture session...")
                        time.sleep(1)
                        
                except Exception as e:
                    print(f"⚠️ Error in packet capture session: {e}")
                    if self.monitoring_active:
                        time.sleep(5)
                        
        except Exception as e:
            print(f"❌ Fatal error in monitoring: {e}")
        finally:
            print("🛑 Packet capture stopped")
            self.print_final_stats()

    def print_final_stats(self):
        """طباعة الإحصائيات النهائية"""
        duration = time.time() - self.start_time
        packets_per_sec = self.packet_count / duration if duration > 0 else 0
        
        print(f"\n📊 Final Monitoring Statistics:")
        print(f"   Duration: {duration:.1f} seconds")
        print(f"   Total packets: {self.packet_count}")
        print(f"   Attack packets: {self.attack_count}")
        print(f"   Normal packets: {self.packet_count - self.attack_count}")
        print(f"   Average packets per second: {packets_per_sec:.1f}")
        print(f"   Data saved to: {self.csv_file}")

    def stop_monitoring(self):
        """إيقاف المونيترنج"""
        print("🛑 Stopping monitoring...")
        self.monitoring_active = False

class MonitorManager:
    """مدير المونيترنج للتحكم في العمليات"""
    
    def __init__(self, csv_file):
        self.monitor = ContinuousNetworkMonitor(csv_file)
        self.running = True
    
    def signal_handler(self, signum, frame):
        """معالج إشارات النظام"""
        print(f"\n🛑 Received signal {signum}, stopping monitoring...")
        self.running = False
        self.monitor.stop_monitoring()
        sys.exit(0)
    
    def run_continuous(self):
        """تشغيل مونيترنج مستمر"""
        # إعداد معالجات الإشارات
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        print("🚀 Starting continuous network monitoring...")
        print("Press Ctrl+C to stop")
        
        try:
            self.monitor.start_monitoring()
            
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        finally:
            self.monitor.stop_monitoring()
            print("✅ Monitoring stopped")

def main():
    """الدالة الرئيسية"""
    parser = argparse.ArgumentParser(description="Continuous Network Monitor")
    parser.add_argument('--csv-name', type=str, default='network_traffic',
                       help="Base name for CSV file (default: network_traffic)")
    
    args = parser.parse_args()
    
    # إنشاء اسم ملف CSV
    csv_file = f"{args.csv_name}.csv"
    
    try:
        manager = MonitorManager(csv_file)
        manager.run_continuous()
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()