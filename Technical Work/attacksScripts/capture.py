#!/usr/bin/env python3

from scapy.all import sniff, IP, TCP, UDP, ARP, ICMP, Ether
import pandas as pd
import time

# List to store captured packets
packet_list = []

# Function to map protocol numbers to their names
protocol_map = {
    1: 'ICMP',
    6: 'TCP',
    17: 'UDP',
    2054: 'ARP',  # ARP has a specific protocol number in Ethernet
}

# Function to process each packet and extract important features
def process_packet(packet):
    protocol = packet[IP].proto if packet.haslayer(IP) else ('ARP' if packet.haslayer(ARP) else 'Other')
    protocol_name = protocol_map.get(protocol, 'Other') if isinstance(protocol, int) else protocol
    
    features = {
        'No.': len(packet_list) + 1,
        'Time': time.time(),  # Timestamp
        'Source': packet[IP].src if packet.haslayer(IP) else packet[Ether].src,
        'Destination': packet[IP].dst if packet.haslayer(IP) else packet[Ether].dst,
        'Protocol': protocol_name,
        'Length': len(packet),
        'Info': packet.summary(),
        'Type': 'Normal',
        'Type of attack': 'No Attack',
    }
    packet_list.append(features)

# Capture packets for 30 seconds
duration = 30
print(f"📡 Starting packet capture for {duration} seconds...")
sniff(prn=process_packet, timeout=duration)

# Convert data to DataFrame
packet_df = pd.DataFrame(packet_list)

# Save data to CSV file for later analysis
packet_df.to_csv("captured_packets.csv", index=False)

print("✅ Packets saved to 'captured_packets.csv'")
print(packet_df.head())
