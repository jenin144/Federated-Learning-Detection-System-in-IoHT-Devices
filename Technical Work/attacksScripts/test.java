import org.jnetpcap.Pcap;
import org.jnetpcap.packet.JPacket;
import org.jnetpcap.packet.JPacketHandler;
import org.jnetpcap.protocol.network.Ip4;
import org.jnetpcap.protocol.tcpip.Tcp;
import org.jnetpcap.protocol.tcpip.Udp;
import org.jnetpcap.protocol.lan.Ethernet;
import org.jnetpcap.protocol.network.Arp;
import org.jnetpcap.protocol.network.Icmp;

import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

public class NetworkMonitor {

    private static final Map<Integer, String> PROTOCOL_MAP = Map.of(
            1, "ICMP",
            6, "TCP",
            17, "UDP"
    );

    private static final List<Map<String, Object>> packetList = new ArrayList<>();
    private static final Map<String, Map.Entry<String, Long>> arpSpoofingDetected = new ConcurrentHashMap<>();
    private static final Map<String, List<Long>> dosFloodCounter = new ConcurrentHashMap<>();
    private static final Map<String, Set<Integer>> synScanTracker = new ConcurrentHashMap<>();
    private static final AtomicInteger packetCounter = new AtomicInteger(0);

    public static void main(String[] args) {
        System.out.println("🚀 Starting network monitoring...");

        // Find all network devices
        List<PcapIf> devices = new ArrayList<>();
        StringBuilder errbuf = new StringBuilder();
        
        int status = Pcap.findAllDevs(devices, errbuf);
        if (status != Pcap.OK || devices.isEmpty()) {
            System.err.println("Error finding devices: " + errbuf);
            return;
        }

        // Open the first device for capture
        int snaplen = 64 * 1024; // Capture all packets
        int flags = Pcap.MODE_PROMISCUOUS; // Capture all packets
        int timeout = 10 * 1000; // 10 seconds in millis
        Pcap pcap = Pcap.openLive(devices.get(0).getName(), snaplen, flags, timeout, errbuf);

        if (pcap == null) {
            System.err.println("Error opening device: " + errbuf);
            return;
        }

        // Packet handler
        JPacketHandler<String> handler = (packet, user) -> {
            processPacket(packet);
        };

        // Start capturing packets for 20 seconds
        pcap.loop(20 * 1000, handler, "NetworkMonitor");
        pcap.close();

        // Save to CSV
        saveToCSV();

        System.out.println("\n✅ Analysis complete. Data saved to network_traffic.csv");
    }

    private static void processPacket(JPacket packet) {
        String protocol = "Other";
        String src = "";
        String dst = "";
        int length = packet.getTotalSize();

        // Check protocol layers
        if (packet.hasHeader(new Ip4())) {
            Ip4 ip = packet.getHeader(new Ip4());
            protocol = PROTOCOL_MAP.getOrDefault(ip.type(), "Other");
            src = ip.sourceToDotFormat();
            dst = ip.destinationToDotFormat();
        } else if (packet.hasHeader(new Arp())) {
            Arp arp = packet.getHeader(new Arp());
            protocol = "ARP";
            src = arp.spaToDotFormat();
            dst = arp.tpaToDotFormat();
        } else if (packet.hasHeader(new Ethernet())) {
            Ethernet eth = packet.getHeader(new Ethernet());
            src = String.format("%02X:%02X:%02X:%02X:%02X:%02X", 
                eth.source(0), eth.source(1), eth.source(2), 
                eth.source(3), eth.source(4), eth.source(5));
            dst = String.format("%02X:%02X:%02X:%02X:%02X:%02X", 
                eth.destination(0), eth.destination(1), eth.destination(2), 
                eth.destination(3), eth.destination(4), eth.destination(5));
        }

        String attackInfo = detectAttacks(packet);
        String status = !attackInfo.equals("No Attack") ? "Attack" : "Non-Attack";

        Map<String, Object> packetData = new LinkedHashMap<>();
        packetData.put("No.", packetCounter.incrementAndGet());
        packetData.put("Time", System.currentTimeMillis() / 1000);
        packetData.put("Source", src);
        packetData.put("Destination", dst);
        packetData.put("Protocol", protocol);
        packetData.put("Length", length);
        packetData.put("Info", packet.toString());
        packetData.put("Type", status);
        packetData.put("Type of attack", attackInfo);

        packetList.add(packetData);
    }

    private static String detectAttacks(JPacket packet) {
        List<String> attacks = new ArrayList<>();
        long currentTime = System.currentTimeMillis();

        // ARP Spoofing detection
        if (packet.hasHeader(new Arp())) {
            Arp arp = packet.getHeader(new Arp());
            if (arp.operation() == Arp.ARP_REPLY) {
                String srcIp = arp.spaToDotFormat();
                String srcMac = String.format("%02X:%02X:%02X:%02X:%02X:%02X", 
                    arp.sha(0), arp.sha(1), arp.sha(2), arp.sha(3), arp.sha(4), arp.sha(5));

                if (arpSpoofingDetected.containsKey(srcIp)) {
                    Map.Entry<String, Long> entry = arpSpoofingDetected.get(srcIp);
                    String storedMac = entry.getKey();
                    long timestamp = entry.getValue();

                    if (!storedMac.equals(srcMac) && (currentTime - timestamp) < 2000) {
                        attacks.add("ARP Spoofing");
                    }
                }
                arpSpoofingDetected.put(srcIp, new AbstractMap.SimpleEntry<>(srcMac, currentTime));
            }
        }

        // TCP Flood detection
        if (packet.hasHeader(new Ip4()) && packet.hasHeader(new Tcp())) {
            Ip4 ip = packet.getHeader(new Ip4());
            String src = ip.sourceToDotFormat();

            dosFloodCounter.putIfAbsent(src, new ArrayList<>());
            List<Long> timestamps = dosFloodCounter.get(src);

            // Remove timestamps older than 1 second
            timestamps.removeIf(t -> currentTime - t > 1000);
            timestamps.add(currentTime);

            if (timestamps.size() > 100) {
                attacks.add("DoS (TCP Flood)");
                timestamps.clear();
            }
        }

        // Smurf Attack detection
        if (packet.hasHeader(new Ip4()) && packet.hasHeader(new Icmp())) {
            Ip4 ip = packet.getHeader(new Ip4());
            String dst = ip.destinationToDotFormat();
            if (dst.endsWith(".255") || dst.equals("255.255.255.255")) {
                attacks.add("Smurf Attack");
            }
        }

        // Port Scan detection
        if (packet.hasHeader(new Ip4()) && packet.hasHeader(new Tcp())) {
            Ip4 ip = packet.getHeader(new Ip4());
            Tcp tcp = packet.getHeader(new Tcp());
            
            if (tcp.flags_SYN()) {
                String src = ip.sourceToDotFormat();
                String dst = ip.destinationToDotFormat();
                int port = tcp.destination();
                
                String key = src + "->" + dst;
                synScanTracker.putIfAbsent(key, new HashSet<>());
                Set<Integer> ports = synScanTracker.get(key);
                ports.add(port);
                
                if (ports.size() > 5) {
                    attacks.add("Port Scan (Nmap)");
                    ports.clear();
                }
            }
        }

        return attacks.isEmpty() ? "No Attack" : String.join(", ", attacks);
    }

    private static void saveToCSV() {
        try (FileWriter writer = new FileWriter("network_traffic.csv")) {
            // Write header
            if (!packetList.isEmpty()) {
                writer.write(String.join(",", packetList.get(0).keySet()) + "\n");
            }

            // Write data
            for (Map<String, Object> packet : packetList) {
                List<String> values = new ArrayList<>();
                for (Object value : packet.values()) {
                    values.add(value.toString().replace(",", ";"));
                }
                writer.write(String.join(",", values) + "\n");
            }

            // Print summary
            System.out.println("\n🔍 Detection Summary:");
            System.out.println("Total packets analyzed: " + packetList.size());
            
            Map<String, Integer> attackCounts = new HashMap<>();
            for (Map<String, Object> packet : packetList) {
                if (packet.get("Type").equals("Attack")) {
                    String attackType = (String) packet.get("Type of attack");
                    attackCounts.put(attackType, attackCounts.getOrDefault(attackType, 0) + 1);
                }
            }
            
            System.out.println("Top alerts:");
            attackCounts.entrySet().stream()
                .sorted(Map.Entry.<String, Integer>comparingByValue().reversed())
                .limit(5)
                .forEach(entry -> System.out.println(entry.getKey() + ": " + entry.getValue()));
                
        } catch (IOException e) {
            System.err.println("Error writing to CSV: " + e.getMessage());
        }
    }
}
