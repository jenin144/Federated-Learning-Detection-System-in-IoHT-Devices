# AI-Powered Network Attack Simulation Environment

## 📋 Environment Overview

This project simulates a network attack scenario using an AI model. The setup consists of the following components:

- **Attacker:** Kali Linux  
- **Victim:** Ubuntu Linux  
- **AI Model:** Runs on VS Code in Windows

---

## 📁 File Structure

- `attacks.py`: Python script executed by the attacker (Kali Linux)

---

## ⚙️ Configuration Instructions

Before running the code, make sure to update the IP addresses in `attacks.py` based on your victim's IP.  
For example, if the victim IP is `192.168.x.y`, modify the following variables:

```python
target_ip = "192.168.x.y"
broadcast_ip = "192.168.x.255"
spoofed_ip = "192.168.x.1"
```

---

## 🔄 Shared Folder Requirement

Before executing the scripts, you **must set up a shared folder** between the **Windows machine (which contains the `test_dataset` folder)** and the **Ubuntu (Victim) machine**.

This shared folder is necessary for accessing the dataset during runtime.

> 💡 If you're unsure how to configure a shared folder between Windows and Ubuntu, feel free to ask ChatGPT for step-by-step instructions.

---

## ▶️ Execution Steps

To run the entire environment successfully, follow these steps in order:

1. **Open VS Code** on your Windows machine and ensure the AI model is ready.
2. **Run the victim code** on the Ubuntu machine.
3. **Run the attacker code** (`attacks.py`) on the Kali Linux machine.

Make sure that:
- All machines are on the same local network.
- The shared folder is correctly mounted.
- IP addresses in the attack script are properly configured.

---

## 📌 Notes

- Ensure proper permissions are set for shared folder access.
- Always double-check IP configurations before starting the attack simulation.

---

Happy testing and experimenting! 🚀