"""
Phase 10 Stage A (A3): Label-to-Technique Mapping Script.
Generates artifacts/rag/label_technique_map_v1.json.

Strict Methodological Rules:
- Explicit assumptions, not ground truth.
- PortScan: T1595 (primary pre-compromise external reconnaissance from Kali); T1046 (secondary post-compromise).
- DoS: Differentiated by attack mechanics (Hulk & GoldenEye = T1499.003 HTTP request floods;
  slowloris & Slowhttptest = T1499.003 slow HTTP exhaustion; DDoS = T1498.001 network flood).
- Infiltration: Grounded strictly in Part A observables (36 flows, 192.168.10.8 -> 205.174.165.73:444,
  outbound C2/payload transfer). T1210 formally REJECTED. Mapped to T1071.001 / T1095 / T1105.
- Heartbleed: T1190 (+ T1212 / CVE-2014-0160).
- Bot: T1071.001 (Ares HTTP C2).
- Patator (FTP/SSH): T1110.001 (Password Guessing).
- Web Attack - Brute Force: T1110.001 (+ OWASP A07).
- Web Attack - XSS: T1059.007 / CWE-79.
- Web Attack - SQL Injection: T1190 / CWE-89 (+ OWASP A03).
- RULE_ONLY:R104: T1110.001 marked strictly UNCONFIRMED / SUSPECTED SPURIOUS.
- Fallback: NIST SP 800-61r3 CSF Community Profile (Incident Handling).
"""

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path

def build_mapping():
    timestamp_utc = datetime.now(timezone.utc).isoformat()
    
    mappings = {
        "PortScan": {
            "label": "PortScan",
            "primary_technique_id": "T1595",
            "primary_technique_name": "Active Scanning",
            "sub_techniques": ["T1595.001 (Scanning IP Blocks)", "T1595.002 (Vulnerability Scanning)"],
            "secondary_technique_ids": ["T1046"],
            "secondary_technique_names": ["Network Service Discovery"],
            "confidence_rating": "High (Analyst Assumption)",
            "rationale": (
                "Empirical PCAP traffic originates from external adversary machine (Kali Linux, 172.16.0.1) "
                "probing ports across internal victim subnet (192.168.10.50). In MITRE ATT&CK Enterprise Matrix, "
                "pre-compromise external scanning is categorized under Reconnaissance (T1595 Active Scanning). "
                "T1046 (Network Service Discovery) is retained as a secondary mapping for enterprise internal discovery."
            ),
            "primary_mitigations": [
                "M1056 (Pre-compromise / Network Intrusion Prevention)",
                "M1037 (Filter Network Traffic: restrict exposed ports at border firewall)"
            ],
            "detection_analytics": [
                "Multiple sequential SYN packets to closed or non-listening ports over short time windows.",
                "High ratio of SYN to SYN-ACK/ACK packets from external IP addresses."
            ],
            "authoritative_sources": ["MITRE ATT&CK v20.0"]
        },
        "Bot": {
            "label": "Bot",
            "primary_technique_id": "T1071.001",
            "primary_technique_name": "Application Layer Protocol: Web Protocols",
            "sub_techniques": ["T1071.001 (Web Protocols: HTTP/HTTPS)"],
            "secondary_technique_ids": ["T1573", "T1105"],
            "secondary_technique_names": ["Encrypted Channel", "Ingress Tool Transfer"],
            "confidence_rating": "High",
            "rationale": (
                "CIC-IDS2017 documentation specifies execution of the Ares Python-based botnet on victim workstations. "
                "Traffic exhibits periodic outbound HTTP POST requests beaconing to remote C2 server (205.174.165.73) "
                "polling for commands."
            ),
            "primary_mitigations": [
                "M1031 (Network Intrusion Prevention: inspect outbound HTTP headers and URIs)",
                "M1021 (Restricted Web-Based Content: proxy enforcement and C2 domain blacklisting)"
            ],
            "detection_analytics": [
                "Regular beaconing interval periodicity in outbound HTTP connection timestamps.",
                "Unusual User-Agent strings or uniform payload lengths in repetitive POST bodies."
            ],
            "authoritative_sources": ["MITRE ATT&CK v20.0"]
        },
        "Infiltration": {
            "label": "Infiltration",
            "primary_technique_id": "T1071.001",
            "primary_technique_name": "Application Layer Protocol: Web Protocols",
            "sub_techniques": ["T1071.001 (Web Protocols)", "T1095 (Non-Application Layer Protocol)"],
            "secondary_technique_ids": ["T1105", "T1041"],
            "secondary_technique_names": ["Ingress Tool Transfer", "Exfiltration Over C2 Channel"],
            "confidence_rating": "Medium (Analyst Assumption grounded in Part A flow observables)",
            "rationale": (
                "CRITICAL OBSERVATIONAL GROUNDING: Part A empirical flow analysis revealed exactly 36 flows "
                "exclusively directed from internal host 192.168.10.8 to external adversary host 205.174.165.73 "
                "on non-standard destination port 444. Zero inbound remote exploit traffic exists in the dataset. "
                "Mapping to T1210 (Exploitation of Remote Services) is formally REJECTED. The observed session "
                "represents an established reverse TCP/C2 connection facilitating staged tool ingress and data transfer."
            ),
            "primary_mitigations": [
                "M1037 (Filter Network Traffic: strict egress firewall filtering blocking non-standard outbound ports like 444)",
                "M1031 (Network Intrusion Prevention: inspect anomalous outbound TCP sessions)"
            ],
            "detection_analytics": [
                "Outbound connections from internal endpoints to external non-standard ports (e.g. TCP 444).",
                "High asymmetric data transfer ratio over outbound long-lived TCP sessions."
            ],
            "authoritative_sources": ["MITRE ATT&CK v20.0", "Part A Observational Telemetry"]
        },
        "Heartbleed": {
            "label": "Heartbleed",
            "primary_technique_id": "T1190",
            "primary_technique_name": "Exploit Public-Facing Application",
            "sub_techniques": [],
            "secondary_technique_ids": ["T1212"],
            "secondary_technique_names": ["Exploitation for Credential Access"],
            "confidence_rating": "High",
            "rationale": (
                "Heartbleed (CVE-2014-0160) exploits an OpenSSL TLS Heartbeat buffer over-read vulnerability "
                "on port 443, enabling remote unauthenticated memory disclosure of 64 KB per request without log traces."
            ),
            "primary_mitigations": [
                "M1051 (Update Software: upgrade OpenSSL to >= 1.0.1g or compile with -DOPENSSL_NO_HEARTBEATS)",
                "M1031 (Network Intrusion Prevention: inspect TLS Client Hello packets for malformed heartbeat length fields)"
            ],
            "detection_analytics": [
                "TLS Heartbeat request payload specifying length field exceeding actual transmitted payload size.",
                "Multiple rapid TLS heartbeat queries originating from a single remote host."
            ],
            "authoritative_sources": ["MITRE ATT&CK v20.0", "NVD API 2.0 (CVE-2014-0160)"]
        },
        "FTP-Patator": {
            "label": "FTP-Patator",
            "primary_technique_id": "T1110.001",
            "primary_technique_name": "Brute Force: Password Guessing",
            "sub_techniques": ["T1110.001 (Password Guessing)"],
            "secondary_technique_ids": ["T1078"],
            "secondary_technique_names": ["Valid Accounts"],
            "confidence_rating": "High",
            "rationale": (
                "Adversary executes multi-threaded dictionary credential guessing against vsftpd server "
                "on TCP port 21 from 172.16.0.1, generating sequential authentication failures."
            ),
            "primary_mitigations": [
                "M1036 (Account Use Policies: rate limiting and exponential backoff on failed FTP logins)",
                "M1032 (Multi-Factor Authentication)",
                "M1037 (Filter Network Traffic: restrict FTP access to authorized administrative management subnets)"
            ],
            "detection_analytics": [
                "High volume of FTP response code 530 (Login incorrect) within short time window.",
                "High rate of new TCP connections to port 21 from a single source IP."
            ],
            "authoritative_sources": ["MITRE ATT&CK v20.0"]
        },
        "SSH-Patator": {
            "label": "SSH-Patator",
            "primary_technique_id": "T1110.001",
            "primary_technique_name": "Brute Force: Password Guessing",
            "sub_techniques": ["T1110.001 (Password Guessing)"],
            "secondary_technique_ids": ["T1078"],
            "secondary_technique_names": ["Valid Accounts"],
            "confidence_rating": "High",
            "rationale": (
                "Systematic SSH password guessing against OpenSSH server on TCP port 22 from Kali adversary host, "
                "generating rapid encrypted handshake sequences."
            ),
            "primary_mitigations": [
                "M1036 (Account Use Policies: lockouts, fail2ban / SSH rate-limiting)",
                "M1032 (Multi-Factor Authentication / Disable password authentication in favor of ed25519 public keys)",
                "M1037 (Filter Network Traffic: restrict SSH management ports to VPN bastions)"
            ],
            "detection_analytics": [
                "High frequency of short-lived TCP sessions to port 22 terminating with connection reset.",
                "Multiple failed SSH authentication log entries within sliding window (e.g. >10 in 60s)."
            ],
            "authoritative_sources": ["MITRE ATT&CK v20.0"]
        },
        "DoS Hulk": {
            "label": "DoS Hulk",
            "primary_technique_id": "T1499.003",
            "primary_technique_name": "Endpoint Denial of Service: Application Exhaustion Flood",
            "sub_techniques": ["T1499.003 (Application Exhaustion Flood)"],
            "secondary_technique_ids": ["T1499.002"],
            "secondary_technique_names": ["Service Exhaustion"],
            "confidence_rating": "High",
            "rationale": (
                "HULK (HTTP Unbearable Load King) floods web server port 80 with high volumes of unique, "
                "dynamically generated GET requests with randomized query strings and forged Referer headers "
                "to bypass web server caching mechanisms and exhaust backend worker threads."
            ),
            "primary_mitigations": [
                "M1037 (Filter Network Traffic: reverse proxy rate limiting, WAF request validation)",
                "M1050 (Exploit Protection: configure web server connection limits per source IP)"
            ],
            "detection_analytics": [
                "Sudden spike in HTTP GET requests with unique randomized URI parameters from single or limited source IPs.",
                "High ratio of forward packets to backward packets over short intervals."
            ],
            "authoritative_sources": ["MITRE ATT&CK v20.0"]
        },
        "DoS GoldenEye": {
            "label": "DoS GoldenEye",
            "primary_technique_id": "T1499.003",
            "primary_technique_name": "Endpoint Denial of Service: Application Exhaustion Flood",
            "sub_techniques": ["T1499.003 (Application Exhaustion Flood)"],
            "secondary_technique_ids": ["T1499.002"],
            "secondary_technique_names": ["Service Exhaustion"],
            "confidence_rating": "High",
            "rationale": (
                "GoldenEye executes an application layer DoS by keeping HTTP connections persistently open using "
                "Keep-Alive headers while issuing randomized Cache-Control directives (no-cache) to force the web server "
                "into persistent socket allocation."
            ),
            "primary_mitigations": [
                "M1037 (Filter Network Traffic: enforce aggressive HTTP keep-alive timeouts and request body limits)",
                "M1050 (Exploit Protection: tune web server MaxKeepAliveRequests and KeepAliveTimeout)"
            ],
            "detection_analytics": [
                "Abnormal concentration of HTTP sessions maintaining maximum keep-alive requests.",
                "Rapid socket starvation on port 80 despite low aggregate network bandwidth."
            ],
            "authoritative_sources": ["MITRE ATT&CK v20.0"]
        },
        "DoS slowloris": {
            "label": "DoS slowloris",
            "primary_technique_id": "T1499.003",
            "primary_technique_name": "Endpoint Denial of Service: Application Exhaustion Flood",
            "sub_techniques": ["T1499.003 (Application Exhaustion Flood - Slow HTTP)"],
            "secondary_technique_ids": ["T1499.002"],
            "secondary_technique_names": ["Service Exhaustion"],
            "confidence_rating": "High",
            "rationale": (
                "Slowloris opens numerous concurrent HTTP connections to port 80 and transmits incomplete HTTP headers "
                "at periodic intervals (e.g. 'X-a: b\\r\\n' every 15s), preventing server thread release while consuming "
                "negligible network bandwidth."
            ),
            "primary_mitigations": [
                "M1037 (Filter Network Traffic: deploy reverse proxy such as NGINX or HAProxy with strict minimum data rate timeouts)",
                "M1050 (Exploit Protection: configure mod_reqtimeout in Apache to enforce minimum header receive rates)"
            ],
            "detection_analytics": [
                "High number of concurrent open TCP sessions with abnormally low data throughput per flow (<100 B/s).",
                "Connections remaining open for hundreds of seconds without emitting terminating double-CRLF."
            ],
            "authoritative_sources": ["MITRE ATT&CK v20.0"]
        },
        "DoS Slowhttptest": {
            "label": "DoS Slowhttptest",
            "primary_technique_id": "T1499.003",
            "primary_technique_name": "Endpoint Denial of Service: Application Exhaustion Flood",
            "sub_techniques": ["T1499.003 (Application Exhaustion Flood - Slow HTTP/Slow Read)"],
            "secondary_technique_ids": ["T1499.002"],
            "secondary_technique_names": ["Service Exhaustion"],
            "confidence_rating": "High",
            "rationale": (
                "Slowhttptest tool executes slow-POST and slow-read attacks, advertising small TCP window sizes "
                "or transmitting HTTP POST payload bodies one byte at a time to tie up server read buffers."
            ),
            "primary_mitigations": [
                "M1037 (Filter Network Traffic: drop connections with abnormally small TCP window sizes or prolonged write buffers)",
                "M1050 (Exploit Protection: enforce RequestReadTimeout for request bodies)"
            ],
            "detection_analytics": [
                "Long duration TCP sessions with tiny flow byte rates and fragmented TCP window updates.",
                "Asymmetric flow statistics: long inter-arrival times for client payload segments."
            ],
            "authoritative_sources": ["MITRE ATT&CK v20.0"]
        },
        "DDoS": {
            "label": "DDoS",
            "primary_technique_id": "T1498.001",
            "primary_technique_name": "Network Denial of Service: Direct Network Flood",
            "sub_techniques": ["T1498.001 (Direct Network Flood)"],
            "secondary_technique_ids": ["T1499.001"],
            "secondary_technique_names": ["OS Exhaustion Flood"],
            "confidence_rating": "High",
            "rationale": (
                "Adversary utilizes LOIC (Low Orbit Ion Cannon) to launch coordinated high-volume UDP/TCP volumetric "
                "floods against the target infrastructure on Friday afternoon, causing severe bandwidth and buffer exhaustion."
            ),
            "primary_mitigations": [
                "M1037 (Filter Network Traffic: upstream ISP DDoS mitigation, BGP Anycast routing, flow rate-limiting)",
                "M1031 (Network Intrusion Prevention: drop unestablished UDP/TCP flood traffic at perimeter border)"
            ],
            "detection_analytics": [
                "Massive surge in aggregate flow byte rate and packet arrival rate exceeding normal baseline by orders of magnitude.",
                "Floods of repeated UDP datagrams or TCP SYN/ACK packets saturating interface queue depths."
            ],
            "authoritative_sources": ["MITRE ATT&CK v20.0"]
        },
        "Web Attack - Brute Force": {
            "label": "Web Attack - Brute Force",
            "primary_technique_id": "T1110.001",
            "primary_technique_name": "Brute Force: Password Guessing",
            "sub_techniques": ["T1110.001 (Password Guessing)"],
            "secondary_technique_ids": ["OWASP-2021-A07"],
            "secondary_technique_names": ["Identification and Authentication Failures"],
            "confidence_rating": "High",
            "rationale": (
                "Adversary issues high-frequency HTTP POST requests to web application authentication endpoints "
                "(e.g. login.php) testing password lists against user accounts."
            ),
            "primary_mitigations": [
                "M1036 (Account Use Policies: CAPTCHA, account lockouts, rate limiting on HTTP login endpoints)",
                "M1032 (Multi-Factor Authentication for web portal access)"
            ],
            "detection_analytics": [
                "Excessive HTTP POST requests directed to login URI from single IP or user-agent.",
                "High ratio of HTTP 401/403 or redirect response statuses."
            ],
            "authoritative_sources": ["MITRE ATT&CK v20.0", "OWASP Top 10 (2021)"]
        },
        "Web Attack - XSS": {
            "label": "Web Attack - XSS",
            "primary_technique_id": "T1059.007",
            "primary_technique_name": "Command and Scripting Interpreter: JavaScript",
            "sub_techniques": [],
            "secondary_technique_ids": ["CWE-79", "OWASP-2021-A03"],
            "secondary_technique_names": ["Improper Neutralization of Input During Web Page Generation", "Injection"],
            "confidence_rating": "High",
            "rationale": (
                "Adversary transmits reflected or stored Cross-Site Scripting payloads containing HTML and JavaScript "
                "tags (<script>, <svg onload=>) within HTTP parameters to execute arbitrary code in victim browsers."
            ),
            "primary_mitigations": [
                "M1050 (Exploit Protection: context-aware HTML entity encoding, Content Security Policy headers)",
                "M1031 (Network Intrusion Prevention: WAF regex filtering for HTML script constructs)"
            ],
            "detection_analytics": [
                "Presence of HTML script tags, event handlers (onload, onerror), or JavaScript URI schemes in HTTP GET/POST parameters.",
                "WAF alert signatures matching OWASP Core Rule Set XSS rules."
            ],
            "authoritative_sources": ["MITRE ATT&CK v20.0", "OWASP Top 10 (2021)"]
        },
        "Web Attack - Sql Injection": {
            "label": "Web Attack - Sql Injection",
            "primary_technique_id": "T1190",
            "primary_technique_name": "Exploit Public-Facing Application",
            "sub_techniques": [],
            "secondary_technique_ids": ["CWE-89", "OWASP-2021-A03"],
            "secondary_technique_names": ["Improper Neutralization of Special Elements used in an SQL Command", "Injection"],
            "confidence_rating": "High",
            "rationale": (
                "Adversary injects malicious SQL syntax (UNION SELECT, OR 1=1, comment delimiters) into web form parameters "
                "to manipulate backend relational database queries and extract unauthorized data."
            ),
            "primary_mitigations": [
                "M1050 (Exploit Protection: parameterized queries / prepared statements, ORM abstraction)",
                "M1037 (Filter Network Traffic: Web Application Firewall SQL injection inspection rules)",
                "M1026 (Privileged Account Management: database accounts configured with least privilege)"
            ],
            "detection_analytics": [
                "SQL keywords (UNION, SELECT, INSERT, DROP, SLEEP) and SQL escape characters (' -- /*) in HTTP URI or body.",
                "Database error strings (e.g. 'syntax error in SQL statement') in web application HTTP responses."
            ],
            "authoritative_sources": ["MITRE ATT&CK v20.0", "OWASP Top 10 (2021)"]
        },
        "RULE_ONLY:R104": {
            "label": "RULE_ONLY:R104",
            "primary_technique_id": "T1110.001",
            "primary_technique_name": "Brute Force: Password Guessing (UNCONFIRMED / SPURIOUS CANDIDATE)",
            "sub_techniques": ["T1110.001 (Password Guessing)"],
            "secondary_technique_ids": ["T1046"],
            "secondary_technique_names": ["Network Service Discovery"],
            "confidence_rating": "Analyst Assumption (Flagged for Spurious False-Positive Validation)",
            "rationale": (
                "OPERATIONAL GUIDANCE MANDATE: Alerts flagged strictly by deterministic rule R104 without ML corroboration "
                "must be treated as unconfirmed candidates. In Phase 8 empirical evaluation, R104 produced 682 false incidents "
                "and zero unique attack flows, primarily triggered by benign multi-threaded web browser sessions. Guidance "
                "should emphasize corroborating application logs for actual authentication failures before taking disruptive action."
            ),
            "primary_mitigations": [
                "M1036 (Account Use Policies: verify authentication failure logs before issuing host containment)",
                "M1037 (Filter Network Traffic: tune firewall connection rate thresholds)"
            ],
            "detection_analytics": [
                "Rule R104 threshold exceeded (>50 connections to port 21/22/80) without corresponding ML classifier threat score."
            ],
            "authoritative_sources": ["Phase 8 Empirical Evaluation Report", "MITRE ATT&CK v20.0"]
        },
        "GENERIC_FALLBACK": {
            "label": "GENERIC_FALLBACK",
            "primary_technique_id": "NIST-CSF-RS",
            "primary_technique_name": "NIST SP 800-61r3 CSF Community Profile: Incident Response",
            "sub_techniques": ["RS.AN (Incident Analysis)", "RS.MI (Incident Mitigation)"],
            "secondary_technique_ids": ["NIST-CSF-DE"],
            "secondary_technique_names": ["Detect: Anomalous Activity Analysis"],
            "confidence_rating": "Standard Baseline",
            "rationale": (
                "Default baseline guidance applied when an incident exhibits low ML confidence, anomalous flow characteristics "
                "without a definitive attack class match, or unknown hybrid threat patterns. Follows NIST SP 800-61 Revision 3 "
                "continuous CSF 2.0 lifecycle: triage, isolate affected asset, acquire memory/flow telemetry, and conduct root cause analysis."
            ),
            "primary_mitigations": [
                "CSF 2.0 RS.MI-01: Incidents are contained to mitigate impact.",
                "CSF 2.0 RS.AN-03: Analysis is performed to determine what has taken place and the root causes."
            ],
            "detection_analytics": [
                "Composite risk score elevation without definitive attack plurality class match."
            ],
            "authoritative_sources": ["NIST SP 800-61 Revision 3 (CSF 2.0 Community Profile)"]
        }
    }
    
    out_obj = {
        "schema_version": "1.0",
        "title": "CIC-IDS2017 Plurality Threat to MITRE ATT&CK / NIST / OWASP Grounding Map",
        "phase": "Phase 10 Stage A (A3)",
        "created_utc": timestamp_utc,
        "entry_count": len(mappings),
        "methodological_disclaimer": (
            "NOTICE: This mapping represents explicit, expert analyst-assigned associations reflecting "
            "empirical flow characteristics and domain knowledge. It is not an intrinsic mathematical property "
            "of the dataset. All associations are auditable and subject to pre-registered evaluation criteria."
        ),
        "mappings": mappings
    }
    
    out_path = Path("artifacts/rag/label_technique_map_v1.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2)
        
    h = hashlib.sha256()
    with open(out_path, "rb") as f:
        h.update(f.read())
        
    print(f"[Map] Written {len(mappings)} mappings to {out_path}")
    print(f"[Map] SHA-256: {h.hexdigest()}")

if __name__ == "__main__":
    build_mapping()

