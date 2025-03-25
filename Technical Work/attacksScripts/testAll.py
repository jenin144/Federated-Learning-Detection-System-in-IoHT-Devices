from scapy.all import sniff, IP, TCP, UDP, ARP, ICMP, Ether
import pandas as pd
import time
from collections import defaultdict

packet_list = []
arp_spoofing_detected = {}
dos_flood_counter = defaultdict(list)
syn_scan_tracker = defaultdict(set)
protocol_map = {1: 'ICMP', 6: 'TCP', 17: 'UDP'}

def detect_attacks(packet):
    attacks = []
    current_time = time.time()
    if packet.haslayer(ARP) and packet[ARP].op == 2:
        src_ip = packet[ARP].psrc
        src_mac = packet[ARP].hwsrc
        if src_ip in arp_spoofing_detected:
            stored_mac, timestamp = arp_spoofing_detected[src_ip]
            if stored_mac != src_mac and (current_time - timestamp) < 2:
                attacks.append("ARP Spoofing")
        arp_spoofing_detected[src_ip] = (src_mac, current_time)
    if packet.haslayer(TCP) and packet.haslayer(IP):
        src = packet[IP].src
        dst = packet[IP].dst
        dos_flood_counter[src].append(current_time)
        dos_flood_counter[src] = [t for t in dos_flood_counter[src] if current_time - t <= 1]
        if len(dos_flood_counter[src]) > 100:
            attacks.append("DoS (TCP Flood)")
            dos_flood_counter[src].clear()
    if packet.haslayer(ICMP) and packet.haslayer(IP):
        if packet[IP].dst.endswith('.255') or packet[IP].dst == '255.255.255.255':
            attacks.append("Smurf Attack")
    if packet.haslayer(TCP) and packet[TCP].flags == 'S' and packet.haslayer(IP):
        src = packet[IP].src
        dst = packet[IP].dst
        port = packet[TCP].dport
        syn_scan_tracker[(src, dst)].add(port)
        if len(syn_scan_tracker[(src, dst)]) > 5:
            attacks.append("Port Scan (Nmap)")
            syn_scan_tracker[(src, dst)].clear()
    return ", ".join(attacks) if attacks else "No Attack"

def process_packet(packet):
    if packet.haslayer(IP):
        protocol = protocol_map.get(packet[IP].proto, 'Other')
    elif packet.haslayer(ARP):
        protocol = 'ARP'
    else:
        protocol = 'Other'
    if packet.haslayer(IP):
        src = packet[IP].src
        dst = packet[IP].dst
    elif packet.haslayer(ARP):
        src = packet[ARP].psrc
        dst = packet[ARP].pdst
    else:
        src = packet[Ether].src
        dst = packet[Ether].dst
    attack_info = detect_attacks(packet)
    status = "Attack" if attack_info != "No Attack" else "Non-Attack"
    packet_data = {
        'No.': len(packet_list) + 1,
        'Time': int(time.time()),
        'Source': src,
        'Destination': dst,
        'Protocol': protocol,
        'Length': len(packet),
        'Info': packet.summary(),
        'Type': status,
        'Type of attack': attack_info
    }
    packet_list.append(packet_data)

print("🚀 Starting network monitoring...")
sniff(prn=process_packet, timeout=20)
df = pd.DataFrame(packet_list)
df.to_csv('network_traffic.csv', index=False)
print("\n🔍 Detection Summary:")
print(f"Total packets analyzed: {len(df)}")
print("Top alerts:")
print(df[df['Type'] == 'Attack']['Type of attack'].value_counts().head(5))
print("\n✅ Analysis complete. Data saved to network_traffic.csv")