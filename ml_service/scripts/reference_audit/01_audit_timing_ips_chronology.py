"""
audit_items_1_2_3.py
────────────────────
Audits:
  1a. Ground-truth timing per attack label per day (first, last, span, timezone check)
  1b. IP inventory across all 2,520,798 cleaned flows (top Src, Dst, Dport by count for BENIGN vs each attack)
      Attacker IP analysis, 172.16.0.1 aliasing check, BENIGN share using 172.16.0.1
  2.  Weekday assertion: %d/%m/%Y parse vs filename day (July 3-7, 2017)
  3.  Flow Timestamp ordering analysis: is Timestamp flow START time? Compare Timestamp + Flow Duration vs row order.
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import pyarrow.parquet as pq

_ROOT = Path(r"d:\Hybrid Threat Analysis and Cyber Defence Support System - Personal")
sys.path.insert(0, str(_ROOT))
from config.paths import Paths

ref_path = Paths.REFERENCE_DIR / "flow_reference_v1.parquet"
table = pq.read_table(ref_path)
df_ref = table.to_pandas()

print(f"Total rows in flow_reference_v1: {len(df_ref):,}")

# ── Item 2: Weekday assertion ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ITEM 2: Weekday & Date Assertion")
print("=" * 60)

day_dates = {}
for source_day, group in df_ref.groupby("source_day", sort=False):
    # Parse iso timestamps
    dt_series = pd.to_datetime(group["timestamp_iso"])
    dates = dt_series.dt.date.unique()
    weekdays = dt_series.dt.day_name().unique()
    min_dt = dt_series.min()
    max_dt = dt_series.max()
    print(f"Day: {source_day}")
    print(f"  Unique Dates: {dates}")
    print(f"  Weekdays: {weekdays}")
    print(f"  Range: {min_dt} to {max_dt}")
    day_dates[source_day] = {
        "dates": [str(d) for d in dates],
        "weekdays": list(weekdays),
        "min_ts": str(min_dt),
        "max_ts": str(max_dt),
    }

# ── Item 1a: Ground-Truth Timing per Attack Label ────────────────────────────
print("\n" + "=" * 60)
print("ITEM 1a: Attack Timing per Label per Day")
print("=" * 60)

attack_timing = {}
for (source_day, label), group in df_ref.groupby(["source_day", "label"], sort=False):
    if label == "BENIGN":
        continue
    dt_series = pd.to_datetime(group["timestamp_iso"])
    min_t = dt_series.min()
    max_t = dt_series.max()
    span_seconds = (max_t - min_t).total_seconds()
    span_hours = span_seconds / 3600.0
    count = len(group)
    
    print(f"[{source_day}] {label}:")
    print(f"  Flow Count: {count:,}")
    print(f"  First Seen (ISO): {min_t}")
    print(f"  Last Seen  (ISO): {max_t}")
    print(f"  Span: {span_seconds:.0f} s ({span_hours:.2f} h)")
    
    attack_timing[f"{source_day}::{label}"] = {
        "source_day": source_day,
        "label": label,
        "count": count,
        "first_seen": str(min_t),
        "last_seen": str(max_t),
        "span_seconds": span_seconds,
        "span_hours": round(span_hours, 2),
    }

# ── Item 1b: IP Inventory & 172.16.0.1 Analysis ──────────────────────────────
print("\n" + "=" * 60)
print("ITEM 1b: IP Inventory across all 2,520,798 cleaned rows")
print("=" * 60)

# Check 172.16.0.1 usage
src_172 = df_ref[df_ref["source_ip"] == "172.16.0.1"]
print(f"Total flows with source_ip == '172.16.0.1': {len(src_172):,} / {len(df_ref):,} ({len(src_172)/len(df_ref)*100:.2f}%)")
print("Breakdown of 172.16.0.1 by Label:")
print(src_172["label"].value_counts())

benign_flows = df_ref[df_ref["label"] == "BENIGN"]
benign_172_count = (benign_flows["source_ip"] == "172.16.0.1").sum()
print(f"\nBENIGN flows with source_ip == '172.16.0.1': {benign_172_count:,} / {len(benign_flows):,} ({benign_172_count/len(benign_flows)*100:.4f}%)")

# Per-day, per-label top IP and Ports
ip_inventory = {}
for source_day, day_group in df_ref.groupby("source_day", sort=False):
    print(f"\n--- {source_day} ---")
    ip_inventory[source_day] = {}
    for label, lbl_group in day_group.groupby("label", sort=False):
        top_src = lbl_group["source_ip"].value_counts().head(3).to_dict()
        top_dst = lbl_group["destination_ip"].value_counts().head(3).to_dict()
        top_dport = lbl_group["destination_port"].value_counts().head(3).to_dict()
        
        ip_inventory[source_day][label] = {
            "count": len(lbl_group),
            "top_source_ips": {str(k): int(v) for k, v in top_src.items()},
            "top_dest_ips": {str(k): int(v) for k, v in top_dst.items()},
            "top_dest_ports": {int(k): int(v) for k, v in top_dport.items()},
        }
        if label != "BENIGN":
            print(f"  {label} ({len(lbl_group):,} flows):")
            print(f"    Top Src: {top_src}")
            print(f"    Top Dst: {top_dst}")
            print(f"    Top Dport: {top_dport}")

# ── Item 3: Flow Timing Order Analysis (Flow Duration vs Row Index) ──────────
print("\n" + "=" * 60)
print("ITEM 3: Timestamp & Ordering Analysis (Flow Start vs End, File Order)")
print("=" * 60)

# Let's inspect Tuesday raw file:
# Read Timestamp and Flow Duration (in microseconds) for first 1,000 rows of Tuesday
raw_tue = pd.read_csv(
    Paths.LABELLED_TUESDAY_CSV,
    nrows=1000,
    usecols=lambda c: c.strip() in ["Timestamp", "Flow Duration"],
    encoding="utf-8",
    encoding_errors="replace"
)
raw_tue.columns = raw_tue.columns.str.strip()

# Let's check how timestamps change
# Compare start timestamp vs end timestamp = start + (Flow Duration / 1e6)
parsed_start = []
for t in raw_tue["Timestamp"]:
    parts = t.strip().split(" ")
    d_p = [int(x) for x in parts[0].split("/")]
    t_p = [int(x) for x in parts[1].split(":")]
    h = t_p[0]
    if 1 <= h <= 5:
        h += 12
    dt = datetime(d_p[2], d_p[1], d_p[0], h, t_p[1], t_p[2] if len(t_p) > 2 else 0)
    parsed_start.append(dt.timestamp())

start_s = np.array(parsed_start)
dur_s = raw_tue["Flow Duration"].astype(float).to_numpy() / 1e6
end_s = start_s + dur_s

# Check monotonicity of start_s vs end_s vs row order
start_diffs = np.diff(start_s)
end_diffs = np.diff(end_s)

negative_start_jumps = (start_diffs < 0).sum()
negative_end_jumps = (end_diffs < 0).sum()

print(f"First 1,000 Tuesday flows:")
print(f"  Negative start time jumps (row[i+1] < row[i]): {negative_start_jumps} / 999")
print(f"  Negative end time jumps   (row[i+1] < row[i]): {negative_end_jumps} / 999")
print(f"  Min duration: {dur_s.min():.6f} s | Max duration: {dur_s.max():.2f} s")
print("  Sample of first 10 rows (Index, Timestamp_raw, Start_s, Duration_s, End_s):")
for i in range(10):
    print(f"    Row {i:2d}: '{raw_tue['Timestamp'].iloc[i]}' | Start={start_s[i]:.0f} | Dur={dur_s[i]:.4f}s | End={end_s[i]:.4f}")

# Check larger sample across Tuesday
print("\nChecking monotonicity across first 50,000 Tuesday rows:")
raw_tue_50k = pd.read_csv(
    Paths.LABELLED_TUESDAY_CSV,
    nrows=50000,
    usecols=lambda c: c.strip() in ["Timestamp", "Flow Duration"],
    encoding="utf-8",
    encoding_errors="replace"
)
raw_tue_50k.columns = raw_tue_50k.columns.str.strip()
parsed_start_50k = []
for t in raw_tue_50k["Timestamp"]:
    parts = t.strip().split(" ")
    d_p = [int(x) for x in parts[0].split("/")]
    t_p = [int(x) for x in parts[1].split(":")]
    h = t_p[0]
    if 1 <= h <= 5:
        h += 12
    dt = datetime(d_p[2], d_p[1], d_p[0], h, t_p[1], t_p[2] if len(t_p) > 2 else 0)
    parsed_start_50k.append(dt.timestamp())

s_50k = np.array(parsed_start_50k)
d_50k = raw_tue_50k["Flow Duration"].astype(float).to_numpy() / 1e6
e_50k = s_50k + d_50k
print(f"  Start time negative jumps: {(np.diff(s_50k) < 0).sum():,} / 49,999 ({(np.diff(s_50k) < 0).sum()/49999*100:.2f}%)")
print(f"  End time negative jumps:   {(np.diff(e_50k) < 0).sum():,} / 49,999 ({(np.diff(e_50k) < 0).sum()/49999*100:.2f}%)")

# Save structured audit results
audit_results = {
    "item_2_dates": day_dates,
    "item_1a_attack_timing": attack_timing,
    "item_1b_172_summary": {
        "total_172_flows": len(src_172),
        "benign_172_flows": int(benign_172_count),
        "benign_172_percentage": float(benign_172_count / len(benign_flows) * 100),
    },
    "item_1b_inventory": ip_inventory,
}

import json
out_audit = _ROOT / "artifacts" / "reference" / "audit_items_1_2_3.json"
with open(out_audit, "w", encoding="utf-8") as f:
    json.dump(audit_results, f, indent=2)
print(f"\nAudit results saved to: {out_audit}")
