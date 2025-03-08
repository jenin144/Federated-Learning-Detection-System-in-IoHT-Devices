# Nmap Port Scanning Attack Script

import os

def nmap_scan(target_ip):
    os.system(f"nmap -sS -p- {target_ip}")

nmap_scan("192.168.88.9")