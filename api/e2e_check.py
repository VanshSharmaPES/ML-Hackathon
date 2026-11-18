"""End-to-end check: call a running PathSnap API for every trip it serves and record the results.

Run from any laptop that can reach the API. For the two-laptop test, run it on Laptop 2:
    python -m api.e2e_check --api http://<laptop1-ip>:8000 --out outputs/e2e_student_test.csv

Writes one CSV row per trip (status, metrics, server and round-trip time) and exits non-zero
if any trip fails. The API computes the metrics; this script never reads trip files.
"""
import argparse
import csv
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

METRIC_FIELDS = ["top1_accuracy", "route_precision", "route_recall", "route_f1", "continuity"]
FIELDS = ["trip_id", "status", "n_points", "route_arcs", "method", "scored", *METRIC_FIELDS,
          "server_total_ms", "round_trip_ms", "error", "client_host", "api_url", "checked_at"]


def check_trip(base: str, trip: dict, timeout: float, host: str) -> dict:
    row = {"trip_id": trip["trip_id"], "n_points": trip["n_points"], "client_host": host, "api_url": base,
           "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    start = time.perf_counter()
    try:
        r = requests.post(f"{base}/predict", json={"trip_id": trip["trip_id"]}, timeout=timeout)
    except requests.RequestException as exc:
        row.update(status="unreachable", error=type(exc).__name__,
                   round_trip_ms=round((time.perf_counter() - start) * 1000))
        return row
    row["round_trip_ms"] = round((time.perf_counter() - start) * 1000)
    row["status"] = r.status_code
    if not r.ok:
        try:
            row["error"] = r.json().get("detail", r.text)
        except ValueError:
            row["error"] = r.text[:200]
        return row
    data = r.json()
    metrics = data["metrics"]
    row.update(method=data["method"], route_arcs=len(data["predicted_arcs"]),
               scored=metrics.get("scored", True), server_total_ms=round(data["latency_ms"]["total"]))
    for key in METRIC_FIELDS:
        value = metrics.get(key)
        row[key] = None if value is None else round(value, 4)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Run every trip through a running PathSnap API")
    parser.add_argument("--api", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--out", default="outputs/e2e_student_test.csv", help="CSV to write")
    parser.add_argument("--timeout", type=float, default=120, help="Seconds to wait per trip")
    args = parser.parse_args()
    base = args.api.rstrip("/")

    try:
        health = requests.get(f"{base}/health", timeout=10).json()
        trips = requests.get(f"{base}/trips", timeout=10).json()["trips"]
    except requests.RequestException as exc:
        print(f"Cannot reach the API at {base}: {type(exc).__name__}. Is uvicorn running with --host 0.0.0.0?")
        return 2
    host = socket.gethostname()
    print(f"API {base} | method={health['method']} | model_loaded={health['model_loaded']} | "
          f"trips={len(trips)} | client={host}")

    rows = []
    for trip in trips:
        row = check_trip(base, trip, args.timeout, host)
        rows.append(row)
        if row["status"] == 200:
            top1 = "n/a" if row.get("top1_accuracy") is None else f"{row['top1_accuracy']:.3f}"
            print(f"  {row['trip_id']}  200  points={row['n_points']:>5}  top1={top1}  "
                  f"server={row['server_total_ms']} ms  round_trip={row['round_trip_ms']} ms")
        else:
            print(f"  {row['trip_id']}  {row['status']}  {row.get('error', '')}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    ok = [r for r in rows if r["status"] == 200]
    scored = [r for r in ok if r.get("top1_accuracy") is not None]
    print(f"{len(ok)}/{len(rows)} trips succeeded. Wrote {out}")
    if scored:
        points = sum(r["n_points"] for r in scored)
        weighted = sum(r["top1_accuracy"] * r["n_points"] for r in scored) / points
        print(f"Top-1 accuracy: {weighted:.4f} over {points:,} points "
              f"(mean per trip {sum(r['top1_accuracy'] for r in scored) / len(scored):.4f})")
    if ok:
        print(f"Slowest round trip: {max(r['round_trip_ms'] for r in ok)} ms")
    return 0 if len(ok) == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
