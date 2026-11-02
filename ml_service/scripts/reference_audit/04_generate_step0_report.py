"""
04_generate_step0_report.py
───────────────────────────
Synthesizes all Step 0 findings into versioned artifacts:
  artifacts/reference/step0_inspection_findings_v1.json
  artifacts/reference/step0_inspection_findings_v1.md
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timezone

_ROOT = Path(r"d:\Hybrid Threat Analysis and Cyber Defence Support System - Personal")
sys.path.insert(0, str(_ROOT))

from config.paths import Paths

# Load item1_item4_audit.json
with open(Paths.REFERENCE_DIR / "item1_item4_audit.json", "r", encoding="utf-8") as f:
    item1_item4 = json.load(f)

# Load audit_items_1_2_3.json
with open(Paths.REFERENCE_DIR / "audit_items_1_2_3.json", "r", encoding="utf-8") as f:
    audit_data = json.load(f)

step0_report = {
    "report_version": "1.0",
    "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    "dataset_source": "GeneratedLabelledFlows/TrafficLabelling_ (UNB Official Export)",
    "comparator_source": "MachineLearningCSV/MachineLearningCVE (Phase 1-7 Baseline)",
    "findings": {
        "1_column_schema": {
            "traffic_labelling_cols": 85,
            "machine_learning_cols": 79,
            "identifier_cols_restored": [
                "Flow ID",
                "Source IP",
                "Source Port",
                "Destination IP",
                "Protocol",
                "Timestamp"
            ],
            "common_feature_cols_count": 79,
            "missing_cols_in_traffic_labelling": []
        },
        "2_row_counts_and_thursday_defect": {
            "Monday": {"labelled_lines": 529918, "raw_data_rows": 529918, "difference": 0},
            "Tuesday": {"labelled_lines": 445909, "raw_data_rows": 445909, "difference": 0},
            "Wednesday": {"labelled_lines": 692703, "raw_data_rows": 692703, "difference": 0},
            "Thursday_Morning": {
                "labelled_lines": 458968,
                "raw_data_rows": 170366,
                "difference": 288602,
                "defect_nature": "Exactly 288,602 trailing empty comma lines (',,,,,...') after data row 170,366"
            },
            "Thursday_Afternoon": {"labelled_lines": 288602, "raw_data_rows": 288602, "difference": 0},
            "Friday_Morning": {"labelled_lines": 191033, "raw_data_rows": 191033, "difference": 0},
            "Friday_PortScan": {"labelled_lines": 286467, "raw_data_rows": 286467, "difference": 0},
            "Friday_DDoS": {"labelled_lines": 225745, "raw_data_rows": 225745, "difference": 0},
            "total_valid_flows": 2830743
        },
        "3_row_alignment": {
            "shared_columns_tested": ["Destination Port", "Flow Duration", "Total Fwd Packets", "Label"],
            "total_mismatches_observed": 0,
            "alignment_percentage": 100.0
        },
        "4_timestamp_mechanics": {
            "date_format": "%d/%m/%Y (Day/Month/Year)",
            "dates_observed": audit_data["item_2_dates"],
            "weekday_validation": "All dates match July 3-7, 2017 (Monday through Friday) perfectly matching filenames",
            "resolution": {
                "Monday": "Seconds resolution (%d/%m/%Y %H:%M:%S)",
                "Tuesday_through_Friday": "Minute resolution (%d/%m/%Y %H:%M)"
            },
            "hour_partitioning_proof": {
                "morning_hours": [8, 9, 10, 11],
                "noon_hour": [12],
                "afternoon_hours": [1, 2, 3, 4, 5],
                "disambiguation_rule": "h in [1..5] -> h + 12 (13:00..17:00 PM); h in [8..11] -> AM; h == 12 -> 12:00 Noon",
                "ambiguity": "None. Hours 6 and 7 do not exist in the captures."
            },
            "timezone": {
                "offset_vs_unb_schedule": "no offset relative to the published schedule"
            },
            "ordering_analysis": {
                "finding": "Raw CSV rows are not sorted by minute-level start time (24.55% backward start-time jumps observed in Tuesday).",
                "tested_status": "The mechanism producing this ordering is untested; end-time testing is invalid at minute resolution.",
                "sub_minute_loss": "Sub-minute flow arrival order within a single minute is lost in Tuesday-Friday; tiebreaking by row_id is arbitrary.",
                "remedy": "Sort flows deterministically by (epoch_s, row_id) within each correlation grouping key."
            }
        },
        "5_ground_truth_attack_timings": item1_item4["timing_table"],
        "6_ip_inventory_and_nat": {
            "gateway_ip": "172.16.0.1",
            "gateway_total_flows": audit_data["item_1b_172_summary"]["total_172_flows"],
            "gateway_share_of_all_cleaned_flows": round(audit_data["item_1b_172_summary"]["total_172_flows"] / 2520798 * 100, 2),
            "gateway_share_of_benign_flows": round(audit_data["item_1b_172_summary"]["benign_172_percentage"], 4),
            "nat_aliasing_finding": (
                "In the UNB testbed architecture, 172.16.0.1 is the firewall/NAT interface through which "
                "external attacker machines (Kali 205.174.165.x) reached the victim subnet (192.168.10.x). "
                "It aliases multiple attack vectors (FTP, SSH, DoS, Web attacks, PortScan, DDoS)."
            ),
            "per_day_inventory": audit_data["item_1b_inventory"]
        },
        "7_deduplication_impact_per_label": item1_item4["dedup_table"]
    }
}

out_json = Paths.REFERENCE_DIR / "step0_inspection_findings_v1.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(step0_report, f, indent=2)

# Generate Markdown
md_lines = [
    "# Step 0 Inspection & Ground-Truth Findings Report (v1.0)",
    "",
    f"**Generated:** {step0_report['timestamp_utc']}  ",
    "**Inputs:** `GeneratedLabelledFlows/TrafficLabelling_` vs `MachineLearningCSV/MachineLearningCVE`  ",
    "**Audited Rows:** 2,830,743 raw flows → 2,520,798 cleaned flows  ",
    "",
    "---",
    "",
    "## 1. Schema & Column Restoration",
    "",
    "The official UNB `TrafficLabelling_` export contains **85 columns**, restoring the 6 critical network identifiers omitted from `MachineLearningCVE`:",
    "- `Flow ID` (string)",
    "- `Source IP` (string)",
    "- `Source Port` (int32)",
    "- `Destination IP` (string)",
    "- `Protocol` (int32)",
    "- `Timestamp` (string)",
    "",
    "All 79 feature columns present in `MachineLearningCVE` match `TrafficLabelling_` with 0 missing columns.",
    "",
    "---",
    "",
    "## 2. Row Count Audit & Thursday Trailing-Lines Defect",
    "",
    "| Day | TrafficLabelling Lines | MachineLearningCVE Rows | Match | Notes |",
    "|---|---|---|---|---|",
]

for day, d_data in step0_report["findings"]["2_row_counts_and_thursday_defect"].items():
    if not isinstance(d_data, dict):
        continue
    notes = d_data.get("defect_nature", "Exact match")
    md_lines.append(f"| **{day}** | {d_data['labelled_lines']:,} | {d_data['raw_data_rows']:,} | {'✅' if d_data['difference'] == 0 else '⚠️'} | {notes} |")

md_lines.extend([
    "",
    "**Thursday Morning Defect Analysis:**",
    "`Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` contains 458,968 lines in `TrafficLabelling_`. An exhaustive byte audit confirmed lines 1 to 170,366 are valid data rows that match `MachineLearningCVE` with **zero mismatches**. Lines 170,367 to 458,968 (288,602 lines) consist entirely of empty commas (`,,,,,...`), generated by export padding. Streaming with `raw_reader` naturally terminates at line 170,366, eliminating the defect without manual data alteration.",
    "",
    "---",
    "",
    "## 3. Date & Weekday Verification",
    "",
    "Parsing `%d/%m/%Y` yields dates strictly spanning **July 3 to July 7, 2017**, perfectly matching the capture filenames.",
    "*(Note: The UNB dataset page states only 'Mon 09:00 start, Fri 17:00 end', approximate; table shows actual observed min/max timestamps)*:",
    "",
    "| Day Name | Date Observed | Weekday Verified | Observed Clock Span |",
    "|---|---|---|---|",
])

for day, dt_data in step0_report["findings"]["4_timestamp_mechanics"]["dates_observed"].items():
    md_lines.append(f"| **{day}** | {dt_data['dates'][0]} | {dt_data['weekdays'][0]} | {dt_data['min_ts']} to {dt_data['max_ts']} |")

md_lines.extend([
    "",
    "---",
    "",
    "## 4. Flow Timing, Chronology, & Mechanics",
    "",
    "- **Non-Chronological Row Order:** Raw CSV rows are not sorted by minute-level start time. Across the first 50,000 Tuesday flows, **24.55% of adjacent rows jump backward** in start timestamp.",
    "- **Timing Mechanism Status:** The underlying mechanism producing this file ordering is untested. End-time testing ($t_{\\text{start}} + \\text{duration}$) is invalid at minute resolution because durations are in microseconds while start times are truncated to minutes.",
    "- **Sub-Minute Loss & Tiebreaking:** Sub-minute arrival order within the same minute is permanently lost in the Tuesday–Friday captures. In the correlation engine, sorting by `(epoch_s, row_id)` uses `row_id` as an arbitrary, deterministic tiebreaker.",
    "- **12-Hour Clock Disambiguation:** Hours in Tuesday–Friday captures are strictly partitioned into `{8, 9, 10, 11}` (AM), `{12}` (Noon), and `{1, 2, 3, 4, 5}` (PM). Converting hours 1..5 to 13..17 produces deterministic 24-hour ISO timestamps.",
    "- **Timezone:** No offset relative to the published UNB schedule.",
    "",
    "---",
    "",
    "## 5. Ground-Truth Attack Timing vs UNB Documented Schedule (±5 min Window)",
    "",
    "| Attack Label (Day) | Documented Window | Total Flows | Observed Span | Inside Window ±5m | Outside Window ±5m | 1st–99th Percentile Span (or Min-Max if N < 100) |",
    "|---|---|---|---|---|---|---|",
])

for r in step0_report["findings"]["5_ground_truth_attack_timings"]:
    span_col = (
        f"{r['obs_first']} .. {r['obs_last']} (min-max)"
        if r['total_flows'] < 100
        else f"{r['p01']} .. {r['p99']}"
    )
    md_lines.append(
        f"| **{r['label']}** ({r['source_day']}) | {r['doc_window']} | {r['total_flows']:,} | "
        f"{r['obs_first']} .. {r['obs_last']} | {r['inside_pm5m']:,} | "
        f"{r['outside_pm5m']:,} ({r['pct_outside']:.2f}%) | {span_col} |"
    )

md_lines.extend([
    "",
    "---",
    "",
    "## 6. Deduplication Impact per Attack Label",
    "",
    "| Attack Label | Raw Count (`eda_summary.json`) | Cleaned Count (`flow_reference_v1`) | Flows Removed | % Removed |",
    "|---|---|---|---|---|",
])

for d in step0_report["findings"]["7_deduplication_impact_per_label"]:
    md_lines.append(
        f"| **{d['label']}** | {d['raw_count']:,} | {d['cleaned_count']:,} | {d['removed_count']:,} | {d['pct_removed']:.2f}% |"
    )

md_lines.extend([
    "",
    "---",
    "",
    "## 7. IP Inventory & NAT Gateway Analysis",
    "",
    "### Gateway `172.16.0.1` Breakdown",
    f"- Total flows using `172.16.0.1` as Source IP: **{audit_data['item_1b_172_summary']['total_172_flows']:,}** ({audit_data['item_1b_172_summary']['total_172_flows']/2520798*100:.2f}% of all cleaned flows).",
    f"- BENIGN flows using `172.16.0.1`: **{audit_data['item_1b_172_summary']['benign_172_flows']:,}** ({audit_data['item_1b_172_summary']['benign_172_percentage']:.4f}% of benign traffic).",
    "- **NAT Aliasing Reality:** `172.16.0.1` represents the edge firewall/NAT router. All external Kali attacks (FTP/SSH brute force, DoS, Web attacks, PortScan, DDoS) appear with source `172.16.0.1`.",
    "",
    "### Key Attack Endpoints",
    "- **Brute Force (Tuesday):** Attacker `172.16.0.1` -> Victim `192.168.10.50` on port 21 (FTP) and port 22 (SSH).",
    "- **DoS / DDoS:** Attacker `172.16.0.1` -> Victim `192.168.10.50` on port 80 (HTTP).",
    "- **Heartbleed:** Attacker `172.16.0.1` -> Victim `192.168.10.51` on port 444.",
    "- **Web Attacks:** Attacker `172.16.0.1` -> Victim `192.168.10.50` on port 80.",
    "- **Infiltration:** Internal infected client `192.168.10.8` -> External C2 server `205.174.165.73` on port 444.",
    "- **Botnet:** Internal clients (`192.168.10.15`, `192.168.10.8`, `192.168.10.9`) <-> External C2 `205.174.165.73` on port 8080.",
])

out_md = Paths.REFERENCE_DIR / "step0_inspection_findings_v1.md"
with open(out_md, "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))

print(f"Step 0 reports successfully regenerated at:")
print(f"  {out_json}")
print(f"  {out_md}")
