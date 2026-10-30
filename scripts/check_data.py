"""Check that local image copies match the files of the study (configs/image_checksums.sha256).

Example:
    python scripts/check_data.py --data-root /path/to/images [--collection kaggle-ct]
Reports, per collection, the images named in the fold files that are missing or differ in content.
"""
import argparse
import hashlib
import os
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--collection")
    a = p.parse_args()
    status = Counter()
    with open(os.path.join(ROOT, "configs", "image_checksums.sha256"), encoding="utf-8") as f:
        entries = [line.rstrip("\n").split("  ", 1) for line in f if not line.startswith("#")]
    for digest, rel in entries:
        collection = rel.split("/", 1)[0]
        if a.collection and collection != a.collection:
            continue
        path = os.path.join(a.data_root, rel)
        if not os.path.exists(path):
            status[collection, "missing"] += 1
        else:
            with open(path, "rb") as f:
                status[collection, "ok" if hashlib.sha256(f.read()).hexdigest() == digest else "different"] += 1
    for collection in sorted({c for c, _ in status}):
        counts = {k: status[collection, k] for k in ("ok", "missing", "different")}
        print(f"{collection:20s} ok {counts['ok']:5d}  missing {counts['missing']:5d}  different {counts['different']:5d}")
    if any(status[k] for k in status if k[1] != "ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
