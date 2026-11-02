"""
ml_service/tests/test_correlator.py
───────────────────────────────────
Unit tests for the Phase 8 Event Correlator Engine (standard unittest).
"""

import unittest
import numpy as np
import pandas as pd

from ml_service.src.correlation.correlator import (
    correlate_flows,
    correlate_tumbling_bins,
)


class TestCorrelatorEngine(unittest.TestCase):

    def test_ground_truth_blindness(self):
        """Engine must strictly reject DataFrames containing ground-truth columns."""
        df_with_label = pd.DataFrame({
            "row_id": [1],
            "source_ip": ["192.168.10.50"],
            "destination_ip": ["172.16.0.1"],
            "destination_port": [80],
            "epoch_s": [1000],
            "predicted_class": ["DoS Hulk"],
            "confidence": [0.99],
            "label": ["DoS Hulk"],  # FORBIDDEN!
        })
        with self.assertRaises(ValueError):
            correlate_flows(df_with_label)

        df_with_day = pd.DataFrame({
            "row_id": [1],
            "source_ip": ["192.168.10.50"],
            "destination_ip": ["172.16.0.1"],
            "destination_port": [80],
            "epoch_s": [1000],
            "predicted_class": ["DoS Hulk"],
            "confidence": [0.99],
            "source_day": ["Wednesday"],  # FORBIDDEN!
        })
        with self.assertRaises(ValueError):
            correlate_flows(df_with_day)

    def test_inclusive_boundary(self):
        """dt <= G (inclusive) must merge into 1 incident; dt > G must split into 2."""
        key = ("172.16.0.1", "192.168.10.50", 22)
        G = 300

        # Case 1: dt = 300 (should merge)
        df_merge = pd.DataFrame({
            "row_id": [1, 2],
            "source_ip": [key[0], key[0]],
            "destination_ip": [key[1], key[1]],
            "destination_port": [key[2], key[2]],
            "epoch_s": [1000, 1300],
            "predicted_class": ["SSH-Patator", "SSH-Patator"],
            "confidence": [0.95, 0.98],
        })
        inc_merge = correlate_flows(df_merge, gap_seconds=G)
        self.assertEqual(len(inc_merge), 1)
        self.assertEqual(inc_merge.iloc[0]["flow_count"], 2)
        self.assertEqual(inc_merge.iloc[0]["duration_s"], 300.0)

        # Case 2: dt = 301 (should split)
        df_split = pd.DataFrame({
            "row_id": [1, 2],
            "source_ip": [key[0], key[0]],
            "destination_ip": [key[1], key[1]],
            "destination_port": [key[2], key[2]],
            "epoch_s": [1000, 1301],
            "predicted_class": ["SSH-Patator", "SSH-Patator"],
            "confidence": [0.95, 0.98],
        })
        inc_split = correlate_flows(df_split, gap_seconds=G)
        self.assertEqual(len(inc_split), 2)
        self.assertEqual(inc_split.iloc[0]["flow_count"], 1)
        self.assertEqual(inc_split.iloc[1]["flow_count"], 1)

    def test_singletons(self):
        """Singletons (N=1) must be preserved with duration=0."""
        df = pd.DataFrame({
            "row_id": [42],
            "source_ip": ["172.16.0.1"],
            "destination_ip": ["192.168.10.50"],
            "destination_port": [21],
            "epoch_s": [1500],
            "predicted_class": ["FTP-Patator"],
            "confidence": [0.92],
        })
        inc = correlate_flows(df, gap_seconds=300)
        self.assertEqual(len(inc), 1)
        self.assertEqual(inc.iloc[0]["incident_id"], 1)
        self.assertEqual(inc.iloc[0]["flow_count"], 1)
        self.assertEqual(inc.iloc[0]["duration_s"], 0.0)
        self.assertEqual(inc.iloc[0]["flow_ids"], [42])

    def test_unsorted_input(self):
        """Out-of-order flows must be properly sorted by (epoch_s, row_id) and sessionized."""
        key = ("172.16.0.1", "192.168.10.50", 80)
        df_shuffled = pd.DataFrame({
            "row_id": [3, 1, 2],
            "source_ip": [key[0]] * 3,
            "destination_ip": [key[1]] * 3,
            "destination_port": [key[2]] * 3,
            "epoch_s": [1200, 1000, 1100],
            "predicted_class": ["DoS Hulk"] * 3,
            "confidence": [0.9] * 3,
        })
        inc = correlate_flows(df_shuffled, gap_seconds=300)
        self.assertEqual(len(inc), 1)
        self.assertEqual(inc.iloc[0]["flow_ids"], [1, 2, 3])
        self.assertEqual(inc.iloc[0]["start_time"], 1000)
        self.assertEqual(inc.iloc[0]["end_time"], 1200)

    def test_key_separation(self):
        """Different keys must never merge even if timestamps overlap."""
        df = pd.DataFrame({
            "row_id": [1, 2],
            "source_ip": ["172.16.0.1", "172.16.0.1"],
            "destination_ip": ["192.168.10.50", "192.168.10.50"],
            "destination_port": [21, 22],  # different ports
            "epoch_s": [1000, 1005],
            "predicted_class": ["FTP-Patator", "SSH-Patator"],
            "confidence": [0.95, 0.95],
        })
        inc = correlate_flows(df, gap_seconds=300)
        self.assertEqual(len(inc), 2)

    def test_primary_threat_rule_only(self):
        """If ML predicted BENIGN but Rule R104 triggered, primary_threat is 'RULE_ONLY:R104'."""
        df = pd.DataFrame({
            "row_id": [1, 2],
            "source_ip": ["172.16.0.1", "172.16.0.1"],
            "destination_ip": ["192.168.10.50", "192.168.10.50"],
            "destination_port": [21, 21],
            "epoch_s": [1000, 1010],
            "predicted_class": ["BENIGN", "BENIGN"],
            "confidence": [0.85, 0.90],
            "r104_flag": [True, True],
        })
        inc = correlate_flows(df, gap_seconds=300)
        self.assertEqual(len(inc), 1)
        self.assertEqual(inc.iloc[0]["primary_threat"], "RULE_ONLY:R104")
        self.assertTrue(inc.iloc[0]["has_rule_hit"])

    def test_primary_threat_majority(self):
        """Majority non-BENIGN class is selected as primary_threat."""
        df = pd.DataFrame({
            "row_id": [1, 2, 3],
            "source_ip": ["172.16.0.1"] * 3,
            "destination_ip": ["192.168.10.50"] * 3,
            "destination_port": [80] * 3,
            "epoch_s": [1000, 1010, 1020],
            "predicted_class": ["DoS Hulk", "DoS Hulk", "DoS slowloris"],
            "confidence": [0.9, 0.95, 0.8],
        })
        inc = correlate_flows(df, gap_seconds=300)
        self.assertEqual(len(inc), 1)
        self.assertEqual(inc.iloc[0]["primary_threat"], "DoS Hulk")

    def test_tumbling_baseline(self):
        """Fixed tumbling baseline splits across bin boundaries."""
        key = ("172.16.0.1", "192.168.10.50", 80)
        df = pd.DataFrame({
            "row_id": [1, 2],
            "source_ip": [key[0], key[0]],
            "destination_ip": [key[1], key[1]],
            "destination_port": [key[2], key[2]],
            "epoch_s": [299, 300],
            "predicted_class": ["DoS Hulk", "DoS Hulk"],
            "confidence": [0.9, 0.9],
        })
        inc = correlate_tumbling_bins(df, bin_seconds=300)
        self.assertEqual(len(inc), 2)


if __name__ == "__main__":
    unittest.main()
