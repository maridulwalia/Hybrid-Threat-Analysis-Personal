# Reference Table & Step 0 Audit Scripts

Scripts used to audit, inspect, and verify the `GeneratedLabelledFlows/TrafficLabelling_` export and `artifacts/reference/flow_reference_v1.parquet`:

1. `01_audit_timing_ips_chronology.py` — Audits observed attack timing vs UNB windows, validates weekdays (July 3–7, 2017), analyzes IP/NAT distributions, and measures flow start vs completion ordering.
2. `02_audit_thursday_padding_defect.py` — Verifies the 288,602 trailing empty comma lines in `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`.
3. `03_audit_clock_hours_distribution.py` — Audits the unique clock hours per capture day to prove strict AM (8–11), Noon (12), and PM (1–5) partitioning.
4. `04_generate_step0_report.py` — Synthesizes all Step 0 findings into versioned reports (`step0_inspection_findings_v1.json` and `.md`).

