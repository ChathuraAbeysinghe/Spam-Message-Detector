"""Dataset Loader and Validator for SMS Spam Detection.

Handles automatic downloading from the official UCI repository,
extraction, schema validation, cleaning, and train/test preparation.
"""

from pathlib import Path
import zipfile
import requests
import pandas as pd

# UCI SMS Spam Collection official download URL and fallback mirrors
UCI_DATASET_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
FALLBACK_DATASET_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_ZIP_PATH = DATA_DIR / "smsspamcollection.zip"
RAW_DATA_PATH = DATA_DIR / "SMSSpamCollection"
PROCESSED_CSV_PATH = DATA_DIR / "sms_clean.csv"


def download_dataset(force: bool = False) -> Path:
    """Download the SMS Spam Collection dataset if not already present.

    Args:
        force: If True, re-download even if file exists.

    Returns:
        Path to the extracted dataset file.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if RAW_DATA_PATH.exists() and not force:
        return RAW_DATA_PATH

    print(f"Downloading dataset to {DATA_DIR}...", flush=True)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    # Attempt primary verified mirror first for speed and reliability
    try:
        resp_mirror = requests.get(FALLBACK_DATASET_URL, headers=headers, timeout=10)
        resp_mirror.raise_for_status()
        with open(RAW_DATA_PATH, "wb") as f:
            f.write(resp_mirror.content)
        print("Successfully downloaded dataset from repository mirror.", flush=True)
    except Exception as mirror_exc:
        print(f"Mirror download failed ({mirror_exc}). Trying UCI Archive...", flush=True)
        try:
            response = requests.get(UCI_DATASET_URL, headers=headers, timeout=15)
            response.raise_for_status()
            with open(RAW_ZIP_PATH, "wb") as f:
                f.write(response.content)

            with zipfile.ZipFile(RAW_ZIP_PATH, "r") as zip_ref:
                zip_ref.extractall(DATA_DIR)
            print("Successfully downloaded and extracted from UCI Archive.", flush=True)
        except Exception as uci_exc:
            raise RuntimeError(
                f"Failed to download SMS Spam dataset from both sources: {uci_exc}"
            ) from uci_exc

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Expected data file not found at {RAW_DATA_PATH}")

    return RAW_DATA_PATH


def load_raw_dataset(file_path: Path | None = None) -> pd.DataFrame:
    """Load raw SMS Spam Collection file into a Pandas DataFrame.

    The dataset is a tab-separated file with two columns:
    label ('ham' or 'spam') and text.

    Args:
        file_path: Optional custom path to dataset file.

    Returns:
        pd.DataFrame with columns ['label', 'message']
    """
    path = file_path or RAW_DATA_PATH
    if not path.exists():
        path = download_dataset()

    # Read tab-separated format with no header
    df = pd.read_csv(
        path,
        sep="\t",
        names=["label", "message"],
        header=None,
        encoding="utf-8",
        dtype={"label": str, "message": str},
        on_bad_lines="skip",
    )
    return df


def validate_and_clean_data(df: pd.DataFrame, remove_duplicates: bool = True) -> pd.DataFrame:
    """Validate structure, sanitize labels, remove missing values and duplicates.

    Args:
        df: Input raw DataFrame.
        remove_duplicates: Whether to drop duplicate SMS messages.

    Returns:
        Cleaned pd.DataFrame.
    """
    df_clean = df.copy()

    # Ensure required columns exist
    if "label" not in df_clean.columns or "message" not in df_clean.columns:
        raise ValueError("Dataset missing required 'label' or 'message' columns.")

    # Remove rows with null or empty text/label
    initial_count = len(df_clean)
    df_clean = df_clean.dropna(subset=["label", "message"])
    df_clean["message"] = df_clean["message"].astype(str).str.strip()
    df_clean["label"] = df_clean["label"].astype(str).str.strip().str.lower()
    df_clean = df_clean[df_clean["message"] != ""]

    # Validate labels are strictly 'ham' or 'spam'
    valid_labels = {"ham", "spam"}
    invalid_mask = ~df_clean["label"].isin(valid_labels)
    if invalid_mask.any():
        print(f"Dropping {invalid_mask.sum()} rows with unexpected labels.")
        df_clean = df_clean[~invalid_mask]

    # Remove duplicates to avoid data leakage between train and test sets
    if remove_duplicates:
        duplicates_count = df_clean.duplicated(subset=["message"]).sum()
        df_clean = df_clean.drop_duplicates(subset=["message"]).reset_index(drop=True)
        print(f"Removed {duplicates_count} duplicate messages.")

    final_count = len(df_clean)
    print(f"Data cleaning complete: {initial_count} raw rows -> {final_count} clean rows.")
    return df_clean


def get_dataset(force_download: bool = False, save_clean_copy: bool = True) -> pd.DataFrame:
    """Convenience helper to download, load, and validate the dataset.

    Args:
        force_download: Re-download raw files.
        save_clean_copy: Save processed copy to data/sms_clean.csv.

    Returns:
        Cleaned pd.DataFrame with ['label', 'message']
    """
    raw_path = download_dataset(force=force_download)
    df_raw = load_raw_dataset(raw_path)
    df_clean = validate_and_clean_data(df_raw, remove_duplicates=True)

    if save_clean_copy:
        df_clean.to_csv(PROCESSED_CSV_PATH, index=False)
        print(f"Saved clean dataset to {PROCESSED_CSV_PATH}")

    return df_clean


if __name__ == "__main__":
    df = get_dataset()
    print("\nDataset Class Distribution:")
    print(df["label"].value_counts())
    print("\nClass Percentages:")
    print((df["label"].value_counts(normalize=True) * 100).round(2).astype(str) + "%")
