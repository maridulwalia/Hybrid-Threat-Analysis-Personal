"""
Phase 10 Stage A (A4): RAG Pre-Registration Document and Evaluation Suite Generator.
Generates:
  1. artifacts/rag/preregistration_rag_v1.md
  2. artifacts/rag/rag_evaluation_queries_v1.json

Includes:
- >=70 positive threat queries (schema-derived, free-text analyst observables, wrong-label/low-confidence, R104)
- >=30 negative queries (including hard negatives)
- Stratified Dev / Test split (Seed 42) + 15 reserved blind queries
- Explicit Gold Mapping criteria at (technique_id, section_type) level
- Mathematical formulas for Recall@k, MRR, Wilson 95% CIs, ROC-AUC, FAR
- Upfront chunking grid specification (256/50, 512/100, 1024/200)
"""

import json
import hashlib
import random
from datetime import datetime, timezone
from pathlib import Path

def generate_queries():
    # 1. Threat queries per label (5 per label across 14 categories = 70 + 4 R104 = 74)
    threat_templates = [
        # PortScan (T1595)
        {"id": "Q_PORTSCAN_SCH1", "label": "PortScan", "type": "schema", "text": "Incident on port 80 and multiple destination ports, high connection count, PortScan reconnaissance detected from 172.16.0.1.", "gold": ["T1595", "T1595.001", "T1595.002", "T1046"]},
        {"id": "Q_PORTSCAN_SCH2", "label": "PortScan", "type": "schema", "text": "High volume of short TCP SYN flows across ports 21, 22, 80, 443 with zero data bytes, labeled PortScan.", "gold": ["T1595", "T1595.001", "T1046"]},
        {"id": "Q_PORTSCAN_OBS1", "label": "PortScan", "type": "observable", "text": "External IP address 172.16.0.1 sending rapid SYN packets across 1,000 sequential destination ports on internal host 192.168.10.50 within 60 seconds.", "gold": ["T1595", "T1595.002", "T1046"]},
        {"id": "Q_PORTSCAN_OBS2", "label": "PortScan", "type": "observable", "text": "Multiple connection attempts to unassigned TCP ports resulting in RST-ACK responses from perimeter server.", "gold": ["T1595", "T1046"]},
        {"id": "Q_PORTSCAN_WRG1", "label": "PortScan", "type": "wrong_label", "text": "Classifier predicted benign web browsing but flow telemetry reveals 500 unique destination ports contacted in 3 seconds.", "gold": ["T1595", "T1046"]},

        # Bot (T1071.001)
        {"id": "Q_BOT_SCH1", "label": "Bot", "type": "schema", "text": "Incident involving internal workstation beaconing to external IP 205.174.165.73 over HTTP, labeled Bot.", "gold": ["T1071.001", "T1573", "T1105"]},
        {"id": "Q_BOT_SCH2", "label": "Bot", "type": "schema", "text": "Repeated periodic HTTP POST flows with identical byte lengths, labeled Botnet C2.", "gold": ["T1071.001", "T1573"]},
        {"id": "Q_BOT_OBS1", "label": "Bot", "type": "observable", "text": "Internal host 192.168.10.15 makes persistent outbound HTTP connections every 30 seconds to an external server with POST request length 128 bytes.", "gold": ["T1071.001", "T1573"]},
        {"id": "Q_BOT_OBS2", "label": "Bot", "type": "observable", "text": "Suspicious python script connecting out to remote server pulling downstream executable chunks.", "gold": ["T1071.001", "T1105"]},
        {"id": "Q_BOT_WRG1", "label": "Bot", "type": "wrong_label", "text": "Low-confidence benign classification on an endpoint that shows rigid 60-second beacon intervals to an unknown external IP.", "gold": ["T1071.001", "T1573"]},

        # Infiltration (T1071.001 / T1095 / T1105)
        {"id": "Q_INFIL_SCH1", "label": "Infiltration", "type": "schema", "text": "Incident with 36 flows from internal host 192.168.10.8 to external destination 205.174.165.73 port 444, labeled Infiltration.", "gold": ["T1071.001", "T1095", "T1105", "T1041"]},
        {"id": "Q_INFIL_SCH2", "label": "Infiltration", "type": "schema", "text": "Outbound non-standard port 444 connection from internal client workstation, labeled Infiltration.", "gold": ["T1071.001", "T1095", "T1105"]},
        {"id": "Q_INFIL_OBS1", "label": "Infiltration", "type": "observable", "text": "Exactly 36 outbound TCP flows established from victim 192.168.10.8 to external IP 205.174.165.73 on port 444 with high outbound byte transfer.", "gold": ["T1071.001", "T1095", "T1105", "T1041"]},
        {"id": "Q_INFIL_OBS2", "label": "Infiltration", "type": "observable", "text": "Internal workstation establishing reverse shell connection out to remote host on TCP port 444.", "gold": ["T1071.001", "T1095", "T1105"]},
        {"id": "Q_INFIL_WRG1", "label": "Infiltration", "type": "wrong_label", "text": "Model flagged as generic benign traffic but telemetry shows persistent outbound connection to external port 444.", "gold": ["T1071.001", "T1095", "T1105"]},

        # Heartbleed (T1190 / CVE-2014-0160)
        {"id": "Q_HBLD_SCH1", "label": "Heartbleed", "type": "schema", "text": "Incident targeting port 443 with TLS heartbeat payload anomalies, labeled Heartbleed.", "gold": ["T1190", "T1212", "CVE-2014-0160"]},
        {"id": "Q_HBLD_SCH2", "label": "Heartbleed", "type": "schema", "text": "OpenSSL memory leak exploit targeting public HTTPS server, labeled Heartbleed.", "gold": ["T1190", "T1212", "CVE-2014-0160"]},
        {"id": "Q_HBLD_OBS1", "label": "Heartbleed", "type": "observable", "text": "Client sending TLS heartbeat requests on port 443 where length field claims 65,535 bytes but transmitted payload is 1 byte.", "gold": ["T1190", "T1212", "CVE-2014-0160"]},
        {"id": "Q_HBLD_OBS2", "label": "Heartbleed", "type": "observable", "text": "Server process memory dump requested repeatedly over encrypted TLS channel without standard session termination.", "gold": ["T1190", "T1212", "CVE-2014-0160"]},
        {"id": "Q_HBLD_WRG1", "label": "Heartbleed", "type": "wrong_label", "text": "Classified as benign HTTPS traffic but request payload contains OpenSSL heartbeat length over-read signature.", "gold": ["T1190", "T1212", "CVE-2014-0160"]},

        # FTP-Patator (T1110.001)
        {"id": "Q_FTP_SCH1", "label": "FTP-Patator", "type": "schema", "text": "Incident on port 21, high volume of flows, labeled FTP-Patator password guessing.", "gold": ["T1110.001", "T1078"]},
        {"id": "Q_FTP_SCH2", "label": "FTP-Patator", "type": "schema", "text": "Multiple failed authentication attempts against vsftpd server, labeled FTP-Patator.", "gold": ["T1110.001", "T1078"]},
        {"id": "Q_FTP_OBS1", "label": "FTP-Patator", "type": "observable", "text": "Over 1,000 sequential TCP connections to port 21 from single host 172.16.0.1 with rapid USER and PASS commands.", "gold": ["T1110.001", "T1078"]},
        {"id": "Q_FTP_OBS2", "label": "FTP-Patator", "type": "observable", "text": "FTP server returning code 530 Login incorrect at 20 attempts per second from external IP.", "gold": ["T1110.001", "T1078"]},
        {"id": "Q_FTP_WRG1", "label": "FTP-Patator", "type": "wrong_label", "text": "Classifier predicts benign file transfer but authentication logs show rapid dictionary attack against root and admin.", "gold": ["T1110.001", "T1078"]},

        # SSH-Patator (T1110.001)
        {"id": "Q_SSH_SCH1", "label": "SSH-Patator", "type": "schema", "text": "Incident on port 22, high volume of encrypted handshakes, labeled SSH-Patator brute force.", "gold": ["T1110.001", "T1078"]},
        {"id": "Q_SSH_SCH2", "label": "SSH-Patator", "type": "schema", "text": "Automated password guessing against OpenSSH service, labeled SSH-Patator.", "gold": ["T1110.001", "T1078"]},
        {"id": "Q_SSH_OBS1", "label": "SSH-Patator", "type": "observable", "text": "Continuous sequence of TCP connections to port 22 each lasting under 1 second with immediate disconnect from 172.16.0.1.", "gold": ["T1110.001", "T1078"]},
        {"id": "Q_SSH_OBS2", "label": "SSH-Patator", "type": "observable", "text": "SSH daemon reporting repeated invalid user authentication failures from single external IP.", "gold": ["T1110.001", "T1078"]},
        {"id": "Q_SSH_WRG1", "label": "SSH-Patator", "type": "wrong_label", "text": "Low ML confidence on port 22 traffic while server authentication logs show hundreds of failed password attempts per minute.", "gold": ["T1110.001", "T1078"]},

        # DoS Hulk (T1499.003)
        {"id": "Q_HULK_SCH1", "label": "DoS Hulk", "type": "schema", "text": "Incident on port 80, massive request rate with randomized URIs, labeled DoS Hulk.", "gold": ["T1499.003", "T1499.002"]},
        {"id": "Q_HULK_SCH2", "label": "DoS Hulk", "type": "schema", "text": "High volume HTTP GET flood exhausting web server workers, labeled DoS Hulk.", "gold": ["T1499.003", "T1499.002"]},
        {"id": "Q_HULK_OBS1", "label": "DoS Hulk", "type": "observable", "text": "Thousands of HTTP GET requests per second on port 80 each containing randomized query parameters and forged User-Agents.", "gold": ["T1499.003", "T1499.002"]},
        {"id": "Q_HULK_OBS2", "label": "DoS Hulk", "type": "observable", "text": "Apache web server thread pool starvation caused by flood of non-cacheable unique GET requests.", "gold": ["T1499.003", "T1499.002"]},
        {"id": "Q_HULK_WRG1", "label": "DoS Hulk", "type": "wrong_label", "text": "Classified as benign flash crowd but URI query strings consist of completely randomized strings bypassing cache.", "gold": ["T1499.003", "T1499.002"]},

        # DoS GoldenEye (T1499.003)
        {"id": "Q_GOLD_SCH1", "label": "DoS GoldenEye", "type": "schema", "text": "Incident on port 80, HTTP keep-alive exhaustion with no-cache headers, labeled DoS GoldenEye.", "gold": ["T1499.003", "T1499.002"]},
        {"id": "Q_GOLD_SCH2", "label": "DoS GoldenEye", "type": "schema", "text": "Denial of service exploiting persistent HTTP connections, labeled DoS GoldenEye.", "gold": ["T1499.003", "T1499.002"]},
        {"id": "Q_GOLD_OBS1", "label": "DoS GoldenEye", "type": "observable", "text": "HTTP GET requests sent with Cache-Control: no-cache and Keep-Alive: 300 to exhaust server socket pool.", "gold": ["T1499.003", "T1499.002"]},
        {"id": "Q_GOLD_OBS2", "label": "DoS GoldenEye", "type": "observable", "text": "Web server reaching MaxClients limit while total inbound bandwidth remains below 500 KB/s.", "gold": ["T1499.003", "T1499.002"]},
        {"id": "Q_GOLD_WRG1", "label": "DoS GoldenEye", "type": "wrong_label", "text": "Traffic marked as normal web sessions but connections refuse to close and consume all available server threads.", "gold": ["T1499.003", "T1499.002"]},

        # DoS slowloris (T1499.003)
        {"id": "Q_SLOW_SCH1", "label": "DoS slowloris", "type": "schema", "text": "Incident on port 80, low bandwidth incomplete HTTP header attack, labeled DoS slowloris.", "gold": ["T1499.003", "T1499.002"]},
        {"id": "Q_SLOW_SCH2", "label": "DoS slowloris", "type": "schema", "text": "Slow connection starvation attack holding HTTP sockets open, labeled DoS slowloris.", "gold": ["T1499.003", "T1499.002"]},
        {"id": "Q_SLOW_OBS1", "label": "DoS slowloris", "type": "observable", "text": "Hundreds of open TCP connections to port 80 periodically sending partial headers every 15 seconds without final CRLF.", "gold": ["T1499.003", "T1499.002"]},
        {"id": "Q_SLOW_OBS2", "label": "DoS slowloris", "type": "observable", "text": "Web server unresponsive to new clients while network interface utilization is less than 1%.", "gold": ["T1499.003", "T1499.002"]},
        {"id": "Q_SLOW_WRG1", "label": "DoS slowloris", "type": "wrong_label", "text": "Predicted as benign idle sessions but sockets are deliberately held open by incremental header transmission.", "gold": ["T1499.003", "T1499.002"]},

        # DoS Slowhttptest (T1499.003)
        {"id": "Q_SHTT_SCH1", "label": "DoS Slowhttptest", "type": "schema", "text": "Incident on port 80, slow HTTP POST body transmission, labeled DoS Slowhttptest.", "gold": ["T1499.003", "T1499.002"]},
        {"id": "Q_SHTT_SCH2", "label": "DoS Slowhttptest", "type": "schema", "text": "Application exhaustion attack using slow read and slow POST mechanics, labeled DoS Slowhttptest.", "gold": ["T1499.003", "T1499.002"]},
        {"id": "Q_SHTT_OBS1", "label": "DoS Slowhttptest", "type": "observable", "text": "HTTP POST requests advertising large Content-Length but transmitting message body at 1 byte every 10 seconds.", "gold": ["T1499.003", "T1499.002"]},
        {"id": "Q_SHTT_OBS2", "label": "DoS Slowhttptest", "type": "observable", "text": "Client advertising zero or minimal TCP window size to freeze server response buffers.", "gold": ["T1499.003", "T1499.002"]},
        {"id": "Q_SHTT_WRG1", "label": "DoS Slowhttptest", "type": "wrong_label", "text": "Classifier assumes benign slow client but hundreds of simultaneous connections are sending 1-byte POST chunks.", "gold": ["T1499.003", "T1499.002"]},

        # DDoS (T1498.001)
        {"id": "Q_DDOS_SCH1", "label": "DDoS", "type": "schema", "text": "Incident on port 80, massive multi-source packet volume saturating bandwidth, labeled DDoS.", "gold": ["T1498.001", "T1499.001"]},
        {"id": "Q_DDOS_SCH2", "label": "DDoS", "type": "schema", "text": "Volumetric direct network flood causing severe packet loss, labeled DDoS.", "gold": ["T1498.001", "T1499.001"]},
        {"id": "Q_DDOS_OBS1", "label": "DDoS", "type": "observable", "text": "Severe network traffic surge reaching interface saturation with high rate of UDP and TCP flood packets to port 80.", "gold": ["T1498.001", "T1499.001"]},
        {"id": "Q_DDOS_OBS2", "label": "DDoS", "type": "observable", "text": "Router queue drops exceeding 90% due to volumetric LOIC flood targeting enterprise web host.", "gold": ["T1498.001", "T1499.001"]},
        {"id": "Q_DDOS_WRG1", "label": "DDoS", "type": "wrong_label", "text": "Alert labeled generic port scan during network outage when telemetry confirms volumetric pipe saturation.", "gold": ["T1498.001", "T1499.001"]},

        # Web Attack - Brute Force (T1110.001 / OWASP A07)
        {"id": "Q_WBF_SCH1", "label": "Web Attack - Brute Force", "type": "schema", "text": "Incident on port 80, HTTP POST requests to login form, labeled Web Attack - Brute Force.", "gold": ["T1110.001", "OWASP-2021-A07"]},
        {"id": "Q_WBF_SCH2", "label": "Web Attack - Brute Force", "type": "schema", "text": "Web application credential attack targeting user login page, labeled Web Attack - Brute Force.", "gold": ["T1110.001", "OWASP-2021-A07"]},
        {"id": "Q_WBF_OBS1", "label": "Web Attack - Brute Force", "type": "observable", "text": "Multiple HTTP POST requests targeting login.php testing passwords against admin user resulting in 401 statuses.", "gold": ["T1110.001", "OWASP-2021-A07"]},
        {"id": "Q_WBF_OBS2", "label": "Web Attack - Brute Force", "type": "observable", "text": "Automated web credential stuffing against customer account portal with hundreds of submissions per minute.", "gold": ["T1110.001", "OWASP-2021-A07"]},
        {"id": "Q_WBF_WRG1", "label": "Web Attack - Brute Force", "type": "wrong_label", "text": "Flagged as benign user logins but request rate is 50 attempts per second from single external IP.", "gold": ["T1110.001", "OWASP-2021-A07"]},

        # Web Attack - XSS (T1059.007 / CWE-79)
        {"id": "Q_WXSS_SCH1", "label": "Web Attack - XSS", "type": "schema", "text": "Incident on port 80, JavaScript script injection in HTTP parameters, labeled Web Attack - XSS.", "gold": ["T1059.007", "CWE-79", "OWASP-2021-A03"]},
        {"id": "Q_WXSS_SCH2", "label": "Web Attack - XSS", "type": "schema", "text": "Cross-site scripting payload detected in web request, labeled Web Attack - XSS.", "gold": ["T1059.007", "CWE-79", "OWASP-2021-A03"]},
        {"id": "Q_WXSS_OBS1", "label": "Web Attack - XSS", "type": "observable", "text": "HTTP GET request parameter contains script alert tags and JavaScript document cookie theft code.", "gold": ["T1059.007", "CWE-79", "OWASP-2021-A03"]},
        {"id": "Q_WXSS_OBS2", "label": "Web Attack - XSS", "type": "observable", "text": "Web application search field reflecting unencoded HTML event handler payload <img src=x onerror=alert(1)>.", "gold": ["T1059.007", "CWE-79", "OWASP-2021-A03"]},
        {"id": "Q_WXSS_WRG1", "label": "Web Attack - XSS", "type": "wrong_label", "text": "Classified as benign search query but URL decoded parameter contains JavaScript execution payload.", "gold": ["T1059.007", "CWE-79", "OWASP-2021-A03"]},

        # Web Attack - Sql Injection (T1190 / CWE-89)
        {"id": "Q_WSQL_SCH1", "label": "Web Attack - Sql Injection", "type": "schema", "text": "Incident on port 80, SQL metacharacters injected into HTTP query, labeled Web Attack - Sql Injection.", "gold": ["T1190", "CWE-89", "OWASP-2021-A03"]},
        {"id": "Q_WSQL_SCH2", "label": "Web Attack - Sql Injection", "type": "schema", "text": "Database injection attack attempting unauthorized data extraction, labeled Web Attack - Sql Injection.", "gold": ["T1190", "CWE-89", "OWASP-2021-A03"]},
        {"id": "Q_WSQL_OBS1", "label": "Web Attack - Sql Injection", "type": "observable", "text": "HTTP parameter containing UNION SELECT username, password FROM users and comment syntax --.", "gold": ["T1190", "CWE-89", "OWASP-2021-A03"]},
        {"id": "Q_WSQL_OBS2", "label": "Web Attack - Sql Injection", "type": "observable", "text": "Adversary probing web endpoint with boolean SQL injection payloads OR 1=1 to bypass login logic.", "gold": ["T1190", "CWE-89", "OWASP-2021-A03"]},
        {"id": "Q_WSQL_WRG1", "label": "Web Attack - Sql Injection", "type": "wrong_label", "text": "ML classifies request as benign form submit while parameter carries SQL sleep delay injection syntax.", "gold": ["T1190", "CWE-89", "OWASP-2021-A03"]},

        # RULE_ONLY:R104 (4 dedicated queries)
        {"id": "Q_R104_1", "label": "RULE_ONLY:R104", "type": "r104_only", "text": "Incident flagged exclusively by rule R104 on port 80 with 60 connections in 60s, ML confidence 0.00.", "gold": ["T1110.001", "T1046"]},
        {"id": "Q_R104_2", "label": "RULE_ONLY:R104", "type": "r104_only", "text": "Rule R104 threshold exceeded on port 22 but zero ML threat probability; suspected spurious false alarm.", "gold": ["T1110.001", "T1046"]},
        {"id": "Q_R104_3", "label": "RULE_ONLY:R104", "type": "r104_only", "text": "Uncorroborated brute force alert triggered by multi-threaded browser opening parallel connections to intranet server.", "gold": ["T1110.001", "T1046"]},
        {"id": "Q_R104_4", "label": "RULE_ONLY:R104", "type": "r104_only", "text": "Single host triggering R104 connection rate rule on port 21 during routine scheduled data sync.", "gold": ["T1110.001", "T1046"]}
    ]

    # 2. Negative queries (35 total: 25 routine/IT + 10 hard negatives)
    negatives = [
        {"id": "NEG_01", "label": "BENIGN", "type": "negative", "text": "How do I configure DHCP scope options and lease duration on Windows Server 2022?", "gold": []},
        {"id": "NEG_02", "label": "BENIGN", "type": "negative", "text": "What is the recommended procedure for onboarding a new remote employee laptop into Active Directory?", "gold": []},
        {"id": "NEG_03", "label": "BENIGN", "type": "negative", "text": "Employee submitted a ticket requesting a password reset after forgetting credentials during annual leave.", "gold": []},
        {"id": "NEG_04", "label": "BENIGN", "type": "negative", "text": "How do I upgrade the RAM and replace the CMOS battery on a Dell PowerEdge server?", "gold": []},
        {"id": "NEG_05", "label": "BENIGN", "type": "negative", "text": "Where can I find the network printer driver for HP LaserJet Enterprise in the IT share?", "gold": []},
        {"id": "NEG_06", "label": "BENIGN", "type": "negative", "text": "Standard steps to renew an internal TLS certificate issued by enterprise Active Directory Certificate Services.", "gold": []},
        {"id": "NEG_07", "label": "BENIGN", "type": "negative", "text": "Configuring Outlook email client with IMAP and SMTP settings for corporate mail.", "gold": []},
        {"id": "NEG_08", "label": "BENIGN", "type": "negative", "text": "Request to install Microsoft Teams and Zoom desktop applications on marketing department laptops.", "gold": []},
        {"id": "NEG_09", "label": "BENIGN", "type": "negative", "text": "Troubleshooting intermittent Wi-Fi connectivity in the second floor conference room.", "gold": []},
        {"id": "NEG_10", "label": "BENIGN", "type": "negative", "text": "How to map a shared network drive letter Z: using group policy preferences?", "gold": []},
        {"id": "NEG_11", "label": "BENIGN", "type": "negative", "text": "Configuring dual monitor display settings and resolution on Windows 11 workstation.", "gold": []},
        {"id": "NEG_12", "label": "BENIGN", "type": "negative", "text": "Standard patch deployment schedule for monthly Windows cumulative quality updates.", "gold": []},
        {"id": "NEG_13", "label": "BENIGN", "type": "negative", "text": "Setting up an out-of-office automated email response for departing staff member.", "gold": []},
        {"id": "NEG_14", "label": "BENIGN", "type": "negative", "text": "Requesting a temporary guest Wi-Fi voucher for visiting auditors next Tuesday.", "gold": []},
        {"id": "NEG_15", "label": "BENIGN", "type": "negative", "text": "What are the power management and sleep timeout policies enforced on desktop PCs?", "gold": []},
        {"id": "NEG_16", "label": "BENIGN", "type": "negative", "text": "How do I export personal Outlook contacts to a CSV file for backup purposes?", "gold": []},
        {"id": "NEG_17", "label": "BENIGN", "type": "negative", "text": "Cleaning up temporary files and clearing browser cache to resolve web portal rendering glitches.", "gold": []},
        {"id": "NEG_18", "label": "BENIGN", "type": "negative", "text": "Requesting additional cloud storage quota for graphic design project assets.", "gold": []},
        {"id": "NEG_19", "label": "BENIGN", "type": "negative", "text": "Configuring VPN client split-tunneling settings for remote development contractors.", "gold": []},
        {"id": "NEG_20", "label": "BENIGN", "type": "negative", "text": "Procedure for safely disposing of decommissioned desktop workstations and hard drives.", "gold": []},
        {"id": "NEG_21", "label": "BENIGN", "type": "negative", "text": "How to set up automatic screen lock after 5 minutes of user inactivity.", "gold": []},
        {"id": "NEG_22", "label": "BENIGN", "type": "negative", "text": "Enabling dark mode theme in developer IDE and terminal shell.", "gold": []},
        {"id": "NEG_23", "label": "BENIGN", "type": "negative", "text": "Updating firmware on conference room VoIP telephone handset.", "gold": []},
        {"id": "NEG_24", "label": "BENIGN", "type": "negative", "text": "How do I request an ergonomic keyboard and mouse from IT facilities?", "gold": []},
        {"id": "NEG_25", "label": "BENIGN", "type": "negative", "text": "Checking hard disk SMART health status and bad sector counts on office file server.", "gold": []},
        
        # Hard Negatives (overlapping technical keywords, but completely benign operations)
        {"id": "NEG_HARD_01", "label": "BENIGN", "type": "hard_negative", "text": "Running authorized vulnerability scan using Nessus against staging server subnet during approved weekend maintenance window.", "gold": []},
        {"id": "NEG_HARD_02", "label": "BENIGN", "type": "hard_negative", "text": "High volume scheduled database backup streaming over SMB port 445 to disaster recovery NAS appliance.", "gold": []},
        {"id": "NEG_HARD_03", "label": "BENIGN", "type": "hard_negative", "text": "System administrator testing SSH public key authentication and updating sshd_config on port 22.", "gold": []},
        {"id": "NEG_HARD_04", "label": "BENIGN", "type": "hard_negative", "text": "Executing legitimate SQL migration scripts to add foreign key constraints on production database.", "gold": []},
        {"id": "NEG_HARD_05", "label": "BENIGN", "type": "hard_negative", "text": "Legitimate stress test of e-commerce checkout page using Apache JMeter simulating 500 concurrent shoppers.", "gold": []},
        {"id": "NEG_HARD_06", "label": "BENIGN", "type": "hard_negative", "text": "Automated batch script uploading end-of-day sales data to external corporate SFTP server on port 22.", "gold": []},
        {"id": "NEG_HARD_07", "label": "BENIGN", "type": "hard_negative", "text": "Developer testing frontend JavaScript input validation for special characters on user registration form.", "gold": []},
        {"id": "NEG_HARD_08", "label": "BENIGN", "type": "hard_negative", "text": "Corporate proxy server opening multiple parallel outbound TCP connections to CDN endpoints on port 443.", "gold": []},
        {"id": "NEG_HARD_09", "label": "BENIGN", "type": "hard_negative", "text": "Internal DNS server handling query spikes during morning office startup.", "gold": []},
        {"id": "NEG_HARD_10", "label": "BENIGN", "type": "hard_negative", "text": "Network engineer issuing ICMP ping sweeps and traceroute commands to diagnose WAN routing latency.", "gold": []}
    ]

    all_queries = threat_templates + negatives
    
    # Stratified split with fixed seed 42
    rng = random.Random(42)
    
    # 15 reserved blind queries: select 10 from threats + 5 from negatives
    threats_copy = list(threat_templates)
    negs_copy = list(negatives)
    rng.shuffle(threats_copy)
    rng.shuffle(negs_copy)
    
    blind_threats = threats_copy[:10]
    eval_threats = threats_copy[10:]
    
    blind_negs = negs_copy[:5]
    eval_negs = negs_copy[5:]
    
    blind_ids = set(q["id"] for q in blind_threats + blind_negs)
    
    # Stratify remaining eval threats 50/50 Dev/Test by label and type
    dev_ids = set()
    test_ids = set()
    
    threats_by_label = {}
    for q in eval_threats:
        threats_by_label.setdefault(q["label"], []).append(q)
        
    for lbl, q_list in threats_by_label.items():
        rng.shuffle(q_list)
        n_half = len(q_list) // 2
        for q in q_list[:n_half]:
            dev_ids.add(q["id"])
        for q in q_list[n_half:]:
            test_ids.add(q["id"])
            
    negs_by_type = {}
    for q in eval_negs:
        negs_by_type.setdefault(q["type"], []).append(q)
        
    for ntype, q_list in negs_by_type.items():
        rng.shuffle(q_list)
        n_half = len(q_list) // 2
        for q in q_list[:n_half]:
            dev_ids.add(q["id"])
        for q in q_list[n_half:]:
            test_ids.add(q["id"])
            
    # Assign splits
    for q in all_queries:
        if q["id"] in blind_ids:
            q["split"] = "blind_reserved"
        elif q["id"] in dev_ids:
            q["split"] = "dev"
        elif q["id"] in test_ids:
            q["split"] = "test"
        else:
            q["split"] = "test"
            
    return all_queries

def write_preregistration_artifacts():
    queries = generate_queries()
    timestamp_utc = datetime.now(timezone.utc).isoformat()
    
    # Save JSON query suite
    json_path = Path("artifacts/rag/rag_evaluation_queries_v1.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "schema_version": "1.0",
            "phase": "Phase 10 Stage A (A4)",
            "created_utc": timestamp_utc,
            "total_queries": len(queries),
            "threat_queries_count": sum(1 for q in queries if q["label"] != "BENIGN"),
            "negative_queries_count": sum(1 for q in queries if q["label"] == "BENIGN"),
            "blind_reserved_count": sum(1 for q in queries if q["split"] == "blind_reserved"),
            "dev_count": sum(1 for q in queries if q["split"] == "dev"),
            "test_count": sum(1 for q in queries if q["split"] == "test"),
            "queries": queries
        }, f, indent=2)
        
    print(f"[Pre-Reg] Saved {len(queries)} queries to {json_path}")
    
    dev_c = sum(1 for q in queries if q['split'] == 'dev')
    test_c = sum(1 for q in queries if q['split'] == 'test')
    blind_c = sum(1 for q in queries if q['split'] == 'blind_reserved')
    tot_c = len(queries)
    
    md_content = f"""# Phase 10: RAG Knowledge Base Pre-Registration Specification (v1.0)

**Document Status:** Formal Pre-Registration (Locked prior to Stage B Chunking, Embedding, or Indexing)  
**Creation Date:** {timestamp_utc}  
**Governing Standard:** AGENTS.md Phase 10 Pre-Registration Mandate  
**Corpus Grounding:** MITRE ATT&CK v20.0, NIST SP 800-61r3, NVD API 2.0 (CVE-2014-0160), OWASP Top 10 (2021)

---

## 1. Methodological Foundations & Assumptions

1. **Secondary Decision-Support Mandate:** The RAG retrieval pipeline functions strictly as a decision-support and evidence-grounding layer. Retrieval queries are formulated exclusively from upstream operational telemetry (incident schema properties, extracted flow observables, and plurality predicted threat classes). Zero ground-truth test labels or test-set supervisory telemetry are ever ingested by the retrieval engine.
2. **Gold Label Criteria:** For every query q, the gold relevant corpus targets are pre-registered at the (technique_id, section_type) level:
   - A retrieved chunk is scored as **Relevant** if its primary metadata matches any pre-registered gold ATT&CK technique ID (e.g. T1595, T1110.001, T1190, T1071.001, T1498.001, T1499.003), specific CVE identifier (CVE-2014-0160), or OWASP risk category (OWASP-2021-A03, OWASP-2021-A07).
   - For negative queries, the gold target set is strictly empty. Any retrieval score exceeding similarity threshold tau is counted as a False Positive / False Accept.
3. **Strict Zero-LLM Corpus Rule:** The knowledge corpus is formed exclusively of authoritative, immutable text snapshots directly downloaded from primary standards bodies. Zero synthetic or LLM-generated documents exist in the corpus.

---

## 2. Query Suite Structure & Stratification

The pre-registered query suite contains **{tot_c} total queries** partitioned across query paradigms:

| Category | Description | Query Count | Purpose |
|---|---|---|---|
| **Schema Queries** | Natural-language translation of operational incident schema fields (port, flow count, duration, plurality label) | 28 | Tests retrieval when standard operational telemetry is populated |
| **Free-Text Observables** | Flow observables written strictly from Part A empirical telemetry without naming the attack class | 28 | Tests semantic retrieval when analyst describes raw network anomalies |
| **Wrong-Label / Low-Confidence** | Telemetry describes attack mechanics while classifier predicted benign or incorrect label | 14 | Evaluates whether RAG can surface correct grounding despite upstream ML confusion |
| **R104-Only Queries** | Incidents flagged strictly by rule R104 without ML corroboration | 4 | Tests guidance marking candidate alerts as unconfirmed/spurious |
| **Routine IT Negatives** | Benign administrative tasks, helpdesk inquiries, standard IT procedures | 25 | Evaluates rejection of non-attack queries |
| **Hard Negatives** | Benign queries containing overlapping technical keywords (SSH, SQL, backup on port 445, authorized scan) | 10 | Evaluates discriminative power against keyword matching traps |
| **Total Query Suite** | Full set of structured pre-registered queries | **{tot_c}** | Full benchmark coverage |

### Partitioning & Split Protocol
- **Randomization Seed:** Fixed at `seed = 42`.
- **Partitioning Strategy:**
  - **Dev Partition ({dev_c} queries):** Used exclusively for tuning the similarity threshold tau and exploring chunking parameters.
  - **Test Partition ({test_c} queries):** Held completely untouched until Stage B final evaluation.
  - **Blind Reserved Partition ({blind_c} queries):** Reserved for blind human analyst annotation during Phase 14 testing.

---

## 3. Pre-Registered Evaluation Metrics

### 3.1 Ranking & Retrieval Metrics (Evaluated on Threat Queries)
For each query q with pre-registered gold technique set G_q:

1. **Recall@k (k in {{1, 3, 5}}):**
   Proportion of queries where at least one chunk among the top k retrieved matches G_q.

2. **Mean Reciprocal Rank (MRR):**
   Average of reciprocal ranks (1 / rank) of the first relevant chunk matching G_q across all queries.

3. **Wilson Score 95% Confidence Intervals:**
   Calculated with z = 1.96 for all recall proportions to guarantee exact statistical reporting.

4. **Macro-Averaging:**
   All recall and ranking metrics will be reported both as micro-averages across all queries and macro-averaged across the 14 distinct attack classes to prevent dominant classes from biasing aggregate performance.

### 3.2 Negative Rejection & Calibration Metrics (Evaluated on Negative Queries)
1. **ROC-AUC on Top-1 Cosine Similarity Score:**
   Evaluates how cleanly the top-1 retrieval similarity score separates threat queries from negative queries across all possible decision thresholds.
2. **False Accept Rate (FAR) at Threshold tau:**
   Proportion of negative queries where top-1 similarity exceeds tau.

---

## 4. Frozen Threshold Decision Rule

- The minimum similarity acceptance threshold tau will be optimized strictly on the **Dev partition** to maximize F1(tau).
- Once determined on Dev, tau is **permanently frozen** and applied to the Test partition. Queries with top-1 similarity < tau trigger the `GENERIC_FALLBACK` triage protocol.

---

## 5. Comparative Baselines (Upfront Commitment)

The semantic RAG retrieval engine (`all-MiniLM-L6-v2` + cosine distance in ChromaDB) will be compared directly against:
1. **Naive Label-to-Technique Metadata Lookup:** Deterministic dictionary lookup mapping the plurality threat label directly to its primary ATT&CK ID without performing semantic vector search.
2. **Random Retrieval Baseline:** Uniformly random sampling of k chunks from the vector database.

---

## 6. Stage B Chunking Grid Specification

In Stage B, documents will be segmented and evaluated across the following pre-registered parameter grid:

| Grid ID | Chunk Size (tokens) | Chunk Overlap (tokens) | Primary Rationale |
|---|---|---|---|
| **C256** | 256 | 50 | High granularity; targets individual mitigation statements and detection rules |
| **C512 (Baseline)** | 512 | 100 | Balanced context window; fits full ATT&CK technique descriptions and NIST sections |
| **C1024** | 1024 | 200 | Broad context; captures multiple mitigation relationships and full vulnerability analyses |

*Note: The chunking grid is pre-registered here. No chunking or embedding execution will occur until Stage B.*

---

## 7. Blind Human Annotation Protocol

15 queries ({blind_c} queries) are segregated into the `blind_reserved` split. During evaluation, these queries will be presented in randomized, unlabelled order for human analyst validation to evaluate inter-annotator agreement against the pre-registered gold labels.
"""

    md_path = Path("artifacts/rag/preregistration_rag_v1.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
        
    h = hashlib.sha256()
    with open(md_path, "rb") as f:
        h.update(f.read())
    sha_md = h.hexdigest()
    
    print(f"[Pre-Reg] Written pre-registration markdown to {md_path}")
    print(f"[Pre-Reg] SHA-256: {sha_md}")
    return sha_md

if __name__ == "__main__":
    write_preregistration_artifacts()
