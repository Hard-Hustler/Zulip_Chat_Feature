# minio_exporter.py
import time
import json
import io
import os
import urllib3
from minio import Minio
from prometheus_client import start_http_server, Gauge, Counter, Histogram
import pandas as pd
import numpy as np

# ── Prometheus metrics ──────────────────────────────────────────────────────
OBJECTS_TOTAL       = Gauge('minio_objects_total', 'Total objects in bucket', ['bucket'])
MISSING_RATE        = Gauge('data_missing_rate', 'Fraction of missing values per field', ['bucket', 'field'])
SCHEMA_VIOLATIONS   = Counter('data_schema_violations_total', 'Schema violations detected', ['bucket', 'field'])
DRIFT_PSI           = Gauge('data_drift_psi', 'Population Stability Index per feature', ['bucket', 'feature'])
RECORD_COUNT        = Gauge('data_record_count', 'Number of records in latest scanned file', ['bucket'])
SCAN_DURATION       = Histogram('data_scan_duration_seconds', 'Time to scan a file', ['bucket'])
LAST_SCAN_TIMESTAMP = Gauge('data_last_scan_timestamp', 'Unix timestamp of last scan', ['bucket'])

BUCKET = os.getenv('MINIO_BUCKET', 'zulip-rewriter')
MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT', '129.114.27.192.nip.io') 
ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY')
SECRET_KEY = os.getenv('MINIO_SECRET_KEY')

# ── Baseline distribution (compute once from a reference file or hardcode) ──
# Keys = feature names, values = histogram bin edges and reference counts
BASELINE = {}  # populated on first scan

http_client = urllib3.PoolManager(
    cert_reqs='CERT_NONE',
    assert_hostname=False
)
client = Minio(
    "minio.129.114.27.192.nip.io", # Use the host without https:// prefix
    access_key=ACCESS_KEY,
    secret_key=SECRET_KEY,
    secure=True, # Set to False to bypass the SSL certificate requirement
    http_client=http_client
)

def compute_psi(baseline_counts, current_counts, epsilon=1e-6):
    b = np.array(baseline_counts, dtype=float) + epsilon
    c = np.array(current_counts, dtype=float) + epsilon
    b /= b.sum(); c /= c.sum()
    return float(np.sum((c - b) * np.log(c / b)))

def scan_latest_object():
    global BASELINE
    objects = list(client.list_objects(BUCKET, recursive=True))
    OBJECTS_TOTAL.labels(bucket=BUCKET).set(len(objects))

    # pick the most recently modified JSON/JSONL/CSV object
    data_objects = [o for o in objects if o.object_name.endswith(('.json', '.jsonl', '.csv'))]
    if not data_objects:
        return
    latest = max(data_objects, key=lambda o: o.last_modified)

    with SCAN_DURATION.labels(bucket=BUCKET).time():
        response = client.get_object(BUCKET, latest.object_name)
        raw = response.read()

    # parse
    if latest.object_name.endswith('.csv'):
        df = pd.read_csv(io.BytesIO(raw))
    else:  # json / jsonl
        try:
            records = json.loads(raw)
            if isinstance(records, dict):
                records = [records]
        except json.JSONDecodeError:
            records = [json.loads(l) for l in raw.splitlines() if l.strip()]
        df = pd.DataFrame(records)

    RECORD_COUNT.labels(bucket=BUCKET).set(len(df))

    # ── Missing rate ─────────────────────────────────────────────────────────
    for col in df.columns:
        rate = df[col].isna().mean()
        MISSING_RATE.labels(bucket=BUCKET, field=col).set(rate)

    # ── Schema: numeric fields should not contain strings ────────────────────
    for col in df.select_dtypes(include='object').columns:
        coerced = pd.to_numeric(df[col], errors='coerce')
        n_violations = coerced.notna().sum()  # values that look numeric but are strings
        if n_violations:
            SCHEMA_VIOLATIONS.labels(bucket=BUCKET, field=col).inc(n_violations)

    # ── Drift (PSI) for numeric columns ──────────────────────────────────────
    numeric_cols = df.select_dtypes(include='number').columns
    if not BASELINE:
        # first run — set baseline
        for col in numeric_cols:
            counts, edges = np.histogram(df[col].dropna(), bins=10)
            BASELINE[col] = {'counts': counts.tolist(), 'edges': edges.tolist()}
        print(f"[{time.strftime('%H:%M:%S')}] Baseline set from {latest.object_name}")
    else:
        for col in numeric_cols:
            if col not in BASELINE:
                continue
            edges = np.array(BASELINE[col]['edges'])
            counts, _ = np.histogram(df[col].dropna(), bins=edges)
            psi = compute_psi(BASELINE[col]['counts'], counts)
            DRIFT_PSI.labels(bucket=BUCKET, feature=col).set(psi)

    LAST_SCAN_TIMESTAMP.labels(bucket=BUCKET).set(time.time())
    print(f"[{time.strftime('%H:%M:%S')}] Scanned {latest.object_name} — {len(df)} records")

if __name__ == '__main__':
    start_http_server(8000)
    print("Exporter running on :8000")
    while True:
        try:
            scan_latest_object()
        except Exception as e:
            print(f"Error: {e}")
        time.sleep(60)  # scan every 60s
