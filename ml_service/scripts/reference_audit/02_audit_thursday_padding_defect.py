"""
check_empty_lines.py
────────────────────
Check whether lines 170,367 to 458,968 in Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv are all empty/commas
"""
import sys
from pathlib import Path

_ROOT = Path(r"d:\Hybrid Threat Analysis and Cyber Defence Support System - Personal")
sys.path.insert(0, str(_ROOT))
from config.paths import Paths

label_thurs_am = Paths.LABELLED_THURSDAY_AM_CSV

non_empty_count = 0
empty_count = 0

with open(label_thurs_am, "r", encoding="cp1252", errors="replace") as f:
    for idx, line in enumerate(f):
        if idx <= 170366:
            continue
        cleaned = line.strip().replace(",", "")
        if cleaned:
            non_empty_count += 1
            if non_empty_count <= 5:
                print(f"Non-empty line at {idx}: {line[:60]}")
        else:
            empty_count += 1

print(f"Total lines checked after line 170,366: {empty_count + non_empty_count}")
print(f"Empty comma lines: {empty_count}")
print(f"Non-empty lines: {non_empty_count}")
