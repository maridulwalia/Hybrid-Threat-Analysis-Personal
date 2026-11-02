"""
config/paths.py
───────────────
Single source of truth for all filesystem paths used across the project.

Every phase script, FastAPI module, and notebook should import from here
rather than hardcoding paths. This means a single edit here propagates
everywhere — no grep-and-replace needed if the project moves.

Usage
-----
    from config.paths import Paths

    df = pd.read_csv(Paths.TUESDAY_CSV)
    model.save(Paths.model("rf_baseline_v1.pkl"))
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the project root (one level up from this file).
_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")


class Paths:
    # ── Project root ─────────────────────────────────────────────────────────
    ROOT: Path = _ROOT

    # ── Dataset (CIC-IDS2017) ────────────────────────────────────────────────
    DATASET_DIR: Path = Path(
        os.getenv("DATASET_DIR", str(_ROOT / "MachineLearningCSV" / "MachineLearningCVE"))
    )

    # Individual daily CSV files — use these constants instead of string literals.
    MONDAY_CSV    = DATASET_DIR / "Monday-WorkingHours.pcap_ISCX.csv"
    TUESDAY_CSV   = DATASET_DIR / "Tuesday-WorkingHours.pcap_ISCX.csv"
    WEDNESDAY_CSV = DATASET_DIR / "Wednesday-workingHours.pcap_ISCX.csv"
    THURSDAY_AM_CSV = DATASET_DIR / "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
    THURSDAY_PM_CSV = DATASET_DIR / "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"
    FRIDAY_AM_CSV = DATASET_DIR / "Friday-WorkingHours-Morning.pcap_ISCX.csv"
    FRIDAY_PORTSCAN_CSV = DATASET_DIR / "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
    FRIDAY_DDOS_CSV     = DATASET_DIR / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

    ALL_CSVS: list[Path] = [
        MONDAY_CSV,
        TUESDAY_CSV,
        WEDNESDAY_CSV,
        THURSDAY_AM_CSV,
        THURSDAY_PM_CSV,
        FRIDAY_AM_CSV,
        FRIDAY_PORTSCAN_CSV,
        FRIDAY_DDOS_CSV,
    ]

    # ── Labelled flows with IPs & timestamps (TrafficLabelling export) ────────
    TRAFFIC_LABELLING_DIR: Path = Path(
        os.getenv(
            "TRAFFIC_LABELLING_DIR",
            str(_ROOT / "GeneratedLabelledFlows" / "TrafficLabelling_"),
        )
    )

    LABELLED_MONDAY_CSV = TRAFFIC_LABELLING_DIR / "Monday-WorkingHours.pcap_ISCX.csv"
    LABELLED_TUESDAY_CSV = TRAFFIC_LABELLING_DIR / "Tuesday-WorkingHours.pcap_ISCX.csv"
    LABELLED_WEDNESDAY_CSV = TRAFFIC_LABELLING_DIR / "Wednesday-workingHours.pcap_ISCX.csv"
    LABELLED_THURSDAY_AM_CSV = TRAFFIC_LABELLING_DIR / "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
    LABELLED_THURSDAY_PM_CSV = TRAFFIC_LABELLING_DIR / "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"
    LABELLED_FRIDAY_AM_CSV = TRAFFIC_LABELLING_DIR / "Friday-WorkingHours-Morning.pcap_ISCX.csv"
    LABELLED_FRIDAY_PORTSCAN_CSV = TRAFFIC_LABELLING_DIR / "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
    LABELLED_FRIDAY_DDOS_CSV = TRAFFIC_LABELLING_DIR / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

    LABELLED_CSVS: list[Path] = [
        LABELLED_MONDAY_CSV,
        LABELLED_TUESDAY_CSV,
        LABELLED_WEDNESDAY_CSV,
        LABELLED_THURSDAY_AM_CSV,
        LABELLED_THURSDAY_PM_CSV,
        LABELLED_FRIDAY_AM_CSV,
        LABELLED_FRIDAY_PORTSCAN_CSV,
        LABELLED_FRIDAY_DDOS_CSV,
    ]

    # ── Saved models (Phase 4+) ───────────────────────────────────────────────
    MODELS_DIR: Path = Path(
        os.getenv("MODELS_DIR", str(_ROOT / "models"))
    )

    @staticmethod
    def model(filename: str) -> Path:
        """Return the full path for a versioned model file.

        Example:
            Paths.model("rf_baseline_v1.pkl")
            → <root>/models/rf_baseline_v1.pkl
        """
        return Paths.MODELS_DIR / filename

    # ── Evaluation artifacts (Phase 4+) ──────────────────────────────────────
    ARTIFACTS_DIR: Path = Path(
        os.getenv("ARTIFACTS_DIR", str(_ROOT / "artifacts"))
    )

    # ── Reference table directory (Phase 8+) ──────────────────────────────────
    REFERENCE_DIR: Path = ARTIFACTS_DIR / "reference"

    @staticmethod
    def artifact(filename: str) -> Path:
        """Return the full path for an evaluation artifact.

        Example:
            Paths.artifact("rf_metrics_v1.json")
            → <root>/artifacts/rf_metrics_v1.json
        """
        return Paths.ARTIFACTS_DIR / filename

    # ── Chroma vector DB (Phase 10+) ─────────────────────────────────────────
    CHROMA_PERSIST_DIR: Path = Path(
        os.getenv("CHROMA_PERSIST_DIR", str(_ROOT / "artifacts" / "chroma_db"))
    )

    # ── Split artifacts (Phase 3+) ────────────────────────────────────────────
    SPLITS_DIR: Path = ARTIFACTS_DIR / "splits"

    @staticmethod
    def split(filename: str) -> Path:
        """Return the full path for a train/test split artifact."""
        return Paths.SPLITS_DIR / filename

    # ── Cleaned dataset outputs (Phase 2+) ───────────────────────────────────
    CLEANED_DIR: Path = ARTIFACTS_DIR / "cleaned"

    # Ordered list of all 8 cleaned CSV files — order is fixed and must match
    # the iteration order used by Phase 2 (clean.py) and Phase 3 (feature_engineer.py).
    # The is_train_mask produced in Phase 3 encodes row positions in this exact order.
    # Never reorder this list without regenerating all Phase 3+ split artifacts.
    CLEANED_CSVS: list[Path] = [
        CLEANED_DIR / "Monday_cleaned.csv",
        CLEANED_DIR / "Tuesday_cleaned.csv",
        CLEANED_DIR / "Wednesday_workingHours_cleaned.csv",
        CLEANED_DIR / "Thursday_Morning_WebAttacks_cleaned.csv",
        CLEANED_DIR / "Thursday_Afternoon_Infilteration_cleaned.csv",
        CLEANED_DIR / "Friday_Morning_cleaned.csv",
        CLEANED_DIR / "Friday_Afternoon_PortScan_cleaned.csv",
        CLEANED_DIR / "Friday_Afternoon_DDos_cleaned.csv",
    ]

    @staticmethod
    def cleaned(filename: str) -> Path:
        """Return the full path for a cleaned daily CSV file.

        Example:
            Paths.cleaned("Monday_cleaned.csv")
            → <root>/artifacts/cleaned/Monday_cleaned.csv
        """
        return Paths.CLEANED_DIR / filename


def validate_paths() -> None:
    """Raise FileNotFoundError if critical directories do not exist.

    Call this at startup in any script that reads the dataset, to catch
    misconfigured paths early rather than deep inside a training loop.
    """
    if not Paths.DATASET_DIR.exists():
        raise FileNotFoundError(
            f"DATASET_DIR not found: {Paths.DATASET_DIR}\n"
            "Check your .env file and ensure the CIC-IDS2017 CSVs are present."
        )
    for csv in Paths.ALL_CSVS:
        if not csv.exists():
            raise FileNotFoundError(
                f"Expected dataset file not found: {csv}\n"
                "Confirm all 8 CIC-IDS2017 CSVs are present in DATASET_DIR."
            )
    # Ensure output directories exist (create silently if missing).
    Paths.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    Paths.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    Paths.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    Paths.CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    Paths.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Quick smoke test — run with: python config/paths.py
    print("Project root :", Paths.ROOT)
    print("Dataset dir  :", Paths.DATASET_DIR)
    print("Models dir   :", Paths.MODELS_DIR)
    print("Artifacts dir:", Paths.ARTIFACTS_DIR)
    print()
    try:
        validate_paths()
        print("[OK]  All paths valid.")
    except FileNotFoundError as e:
        print(f"[ERROR]  {e}")
