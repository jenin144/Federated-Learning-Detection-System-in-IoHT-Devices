# Nmap Port Scanning Attack Script

import os

def nmap_scan(target_ip):
    os.system(f"nmap -sS -p- {target_ip}")

nmap_scan("172.20.10.4")
