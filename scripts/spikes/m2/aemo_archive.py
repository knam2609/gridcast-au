"""M2 step 0: sizes and schema history of AEMO forecast archives on NEMWeb (Oct 2024 → Sep 2026).

usage: uv run python scripts/spikes/m2/aemo_archive.py

Downloads nothing in full: sizes come from HTTP HEAD, and each file's MMS header row (I-row) comes
from inflating only its first 64 KB (HTTP Range). Prints monthly sizes, totals, and every month in
which a table's version or column list changes. Output feeds docs/m2_data_spike.md §4-6.
"""

from __future__ import annotations

import struct
import time
import zlib

import httpx
import pandas as pd

UA = "gridcast-au/0.1 (+https://github.com/knam2609/gridcast-au)"
ROOT = "https://nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM"
MONTHS = pd.period_range("2024-10", "2026-09", freq="M")
FILES = {
    # label: (folder, file-name stem between PUBLIC_ARCHIVE# and #<yyyymm>010000.zip)
    "PREDISPATCHREGIONSUM (all runs)": ("PREDISP_ALL_DATA", "PREDISPATCHREGIONSUM#ALL#FILE01"),
    "PREDISPATCHREGIONSUM (DATA/)": ("DATA", "PREDISPATCHREGIONSUM#FILE01"),
    "P5MIN_REGIONSOLUTION (all runs)": ("P5MIN_ALL_DATA", "P5MIN_REGIONSOLUTION#ALL#FILE01"),
}


def url(month: pd.Period, folder: str, stem: str) -> str:
    y, m = month.year, f"{month.month:02d}"
    name = f"PUBLIC_ARCHIVE#{stem}#{y}{m}010000.zip".replace("#", "%23")
    return f"{ROOT}/{y}/MMSDM_{y}_{m}/MMSDM_Historical_Data_SQLLoader/{folder}/{name}"


def i_row(client: httpx.Client, u: str) -> list[str]:
    b = client.get(u, headers={"Range": "bytes=0-65535"}).content
    if b[:4] != b"PK\x03\x04":
        raise ValueError(f"not a zip local header: {u}")
    name_len, extra_len = struct.unpack("<HH", b[26:30])
    text = zlib.decompressobj(-15).decompress(b[30 + name_len + extra_len :]).decode("ascii")
    return next(line for line in text.splitlines() if line.startswith("I,")).split(",")


def main() -> None:
    sizes = []
    schema_changes = []
    with httpx.Client(headers={"User-Agent": UA}, timeout=60) as client:
        last: dict[str, tuple[str, list[str]]] = {}
        for month in MONTHS:
            for label, (folder, stem) in FILES.items():
                u = url(month, folder, stem)
                head = client.head(u)
                sizes.append(
                    {
                        "month": str(month),
                        "file": label,
                        "status": head.status_code,
                        "mb": int(head.headers.get("content-length", 0)) / 1e6,
                    }
                )
                if "all runs" in label:
                    row = i_row(client, u)
                    version, cols = row[3], row[4:]
                    if label not in last or last[label] != (version, cols):
                        prev_cols = last[label][1] if label in last else cols
                        schema_changes.append(
                            {
                                "month": str(month),
                                "file": label,
                                "version": version,
                                "n_cols": len(cols),
                                "added": [c for c in cols if c not in prev_cols],
                                "removed": [c for c in prev_cols if c not in cols],
                            }
                        )
                    last[label] = (version, cols)
                time.sleep(0.4)
    df = pd.DataFrame(sizes)
    pd.set_option("display.width", 250)
    print("HTTP status counts:", df.status.value_counts().to_dict())
    print("\nMB per month:")
    print(
        df.pivot_table(index="month", columns="file", values="mb", aggfunc="sum")
        .round(1)
        .to_string()
    )
    print("\nTotals over", len(MONTHS), "months (GB):")
    print((df.groupby("file").mb.sum() / 1e3).round(2).to_string())
    print("\nSchema (version / column list) at the first month and at every change:")
    print(pd.DataFrame(schema_changes).to_string(index=False))


if __name__ == "__main__":
    main()
