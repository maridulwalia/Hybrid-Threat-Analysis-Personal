"""
verify_hours_distribution.py
─────────────────────────────
Check unique hours in all files to verify that 1-5 = PM and 8-11 = AM, 12 = noon
"""
import sys
from pathlib import Path
import pandas as pd

_ROOT = Path(r"d:\Hybrid Threat Analysis and Cyber Defence Support System - Personal")
sys.path.insert(0, str(_ROOT))
from config.paths import Paths

all_files = [
    ("Monday", Paths.LABELLED_MONDAY_CSV, 529918),
    ("Tuesday", Paths.LABELLED_TUESDAY_CSV, 445909),
    ("Wednesday", Paths.LABELLED_WEDNESDAY_CSV, 692703),
    ("Thursday-AM", Paths.LABELLED_THURSDAY_AM_CSV, 170366),
    ("Thursday-PM", Paths.LABELLED_THURSDAY_PM_CSV, 288602),
    ("Friday-AM", Paths.LABELLED_FRIDAY_AM_CSV, 191033),
    ("Friday-PortScan", Paths.LABELLED_FRIDAY_PORTSCAN_CSV, 286467),
    ("Friday-DDoS", Paths.LABELLED_FRIDAY_DDOS_CSV, 225745),
]

for name, p, max_r in all_files:
    df = pd.read_csv(p, nrows=max_r, usecols=lambda c: c.strip() == "Timestamp", encoding="cp1252", low_memory=False)
    ts = df.iloc[:, 0].dropna().str.strip()
    
    hours = set()
    for t in ts:
        time_part = t.split(" ")[-1]
        h = int(time_part.split(":")[0])
        hours.add(h)
        
    print(f"{name:15s} -> Unique hours: {sorted(list(hours))}")
