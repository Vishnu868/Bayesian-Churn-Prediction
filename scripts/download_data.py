"""Download the IBM Telco Customer Churn dataset (7,043 customers, 21 columns).

    python scripts/download_data.py                # -> data/WA_Fn-UseC_-Telco-Customer-Churn.csv

The file is the same one published on Kaggle as ``blastchar/telco-customer-churn`` (originally an IBM
Watson Analytics sample). Kaggle needs a login, so we fetch the identical CSV from IBM's public GitHub
mirrors. Downloads resume after an interruption (HTTP Range), are retried with back-off, are written to a
``.part`` file and are only moved into place once the content passes a schema check.
"""
import argparse
import csv
import hashlib
import io
import os
import sys
import time
import urllib.error
import urllib.request

FILENAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MIRRORS = [
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
    "https://raw.githubusercontent.com/treselle-systems/customer_churn_analysis/master/"
    "WA_Fn-UseC_-Telco-Customer-Churn.csv",
]
EXPECTED_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "Churn",
]
EXPECTED_ROWS = 7043


def _fetch(url: str, part: str, retries: int = 5, timeout: int = 60) -> None:
    """Download ``url`` into ``part``, resuming from the bytes already on disk."""
    for attempt in range(1, retries + 1):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        req = urllib.request.Request(url, headers={"User-Agent": "telco-churn-downloader/1.0"})
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                mode = "ab" if have and r.status == 206 else "wb"  # server ignored Range -> restart
                with open(part, mode) as f:
                    while chunk := r.read(1 << 16):
                        f.write(chunk)
            return
        except urllib.error.HTTPError as e:
            if e.code == 416:          # requested range not satisfiable: file already complete
                return
            err = e
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            err = e
        wait = min(2 ** attempt, 30)
        print(f"  attempt {attempt}/{retries} failed ({err}); retrying in {wait}s", file=sys.stderr)
        time.sleep(wait)
    raise RuntimeError(f"could not download {url}")


def validate(path: str) -> int:
    """Check header and row count; return number of data rows."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        if header != EXPECTED_COLUMNS:
            raise ValueError(f"unexpected header: {header}")
        n = sum(1 for row in reader if row)
    if n != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} rows, found {n}")
    return n


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data", help="output directory")
    ap.add_argument("--force", action="store_true", help="re-download even if a valid file exists")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dest = os.path.join(args.out, FILENAME)

    if os.path.exists(dest) and not args.force:
        try:
            n = validate(dest)
            print(f"{dest} already present and valid ({n} rows, sha256 {sha256(dest)[:12]}…)")
            return
        except ValueError as e:
            print(f"existing file invalid ({e}); re-downloading")

    part = dest + ".part"
    for url in MIRRORS:
        print(f"downloading {url}")
        try:
            _fetch(url, part)
            n = validate(part)
        except (RuntimeError, ValueError, UnicodeDecodeError) as e:
            print(f"  mirror failed: {e}", file=sys.stderr)
            if os.path.exists(part):
                os.remove(part)
            continue
        os.replace(part, dest)
        print(f"saved {dest}: {n} rows, {os.path.getsize(dest):,} bytes, sha256 {sha256(dest)}")
        return
    sys.exit("all mirrors failed - download the CSV manually from "
             "https://www.kaggle.com/datasets/blastchar/telco-customer-churn and save it as " + dest)


if __name__ == "__main__":
    main()
