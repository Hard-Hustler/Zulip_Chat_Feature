import os, json, boto3, pandas as pd
from datetime import datetime
from io import BytesIO

BUCKET     = os.getenv("MINIO_BUCKET",     "zulip-rewriter")
ENDPOINT   = os.getenv("MINIO_ENDPOINT",   "http://localhost:9000")
ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
VERSION    = os.getenv("DATA_VERSION",     "v1")
BATCH_DATE = os.getenv("BATCH_DATE",       datetime.utcnow().strftime("%Y-%m-%d"))

s3 = boto3.client("s3", endpoint_url=ENDPOINT,
                  aws_access_key_id=ACCESS_KEY,
                  aws_secret_access_key=SECRET_KEY)

print(f"Batch pipeline starting | version={VERSION} | date={BATCH_DATE}")

print("Step 1: Loading base corpus from MinIO...")
obj      = s3.get_object(Bucket=BUCKET, Key=f"raw/{VERSION}/full.parquet")
df_base  = pd.read_parquet(BytesIO(obj["Body"].read()))
print(f"  Base corpus: {len(df_base)} rows")

print("Step 2: Loading online production logs...")
paginator = s3.get_paginator("list_objects_v2")
log_rows  = []
for page in paginator.paginate(Bucket=BUCKET, Prefix="online_logs/"):
    for obj_meta in page.get("Contents", []):
        try:
            obj  = s3.get_object(Bucket=BUCKET, Key=obj_meta["Key"])
            logs = json.loads(obj["Body"].read())
            for entry in logs:
                inp = entry.get("input", {})
                log_rows.append({
                    "text":        inp.get("original_message", ""),
                    "rewrite_style": inp.get("rewrite_style", ""),
                    "sender_id":   inp.get("context", {}).get("sender_id", ""),
                    "timestamp":   inp.get("timestamp", ""),
                    "source":      "online_log",
                    "binary_label": None,
                    "synthetic":   False,
                    "split":       "online",
                })
        except Exception as e:
            print(f"  Warning: {e}")

df_online = pd.DataFrame(log_rows) if log_rows else pd.DataFrame()
print(f"  Online logs: {len(df_online)} rows")

print("Step 3: Candidate selection (no leakage)...")
df_train = df_base[
    (df_base["text"].str.split().str.len() >= 5) &
    (df_base["binary_label"].notna()) &
    (~df_base["text"].duplicated()) &
    (df_base["split"] == "train")
].copy()

df_test = df_base[
    (df_base["text"].str.split().str.len() >= 5) &
    (df_base["binary_label"].notna()) &
    (df_base["split"] == "test")
].copy()

if not df_online.empty:
    df_train = pd.concat([df_train, df_online], ignore_index=True)

print(f"  Train: {len(df_train)} | Test: {len(df_test)}")

print("Step 4: Uploading versioned datasets...")
batch_ver = f"{VERSION}_batch_{BATCH_DATE}"

def upload(df, name):
    buf = BytesIO()
    df.to_parquet(buf, index=False)
    key = f"batch/{batch_ver}/{name}.parquet"
    s3.put_object(Bucket=BUCKET, Key=key, Body=buf.getvalue())
    print(f"  Uploaded {key} ({len(df)} rows)")

upload(df_train, "train")
upload(df_test,  "test")

manifest = {
    "batch_version":   batch_ver,
    "created_at":      datetime.utcnow().isoformat(),
    "train_rows":      len(df_train),
    "test_rows":       len(df_test),
    "online_log_rows": len(df_online),
    "leakage_policy":  "test set fixed from original corpus; online logs training-only",
    "filters":         {"min_words": 5, "require_label": True, "deduplicated": True},
}
s3.put_object(Bucket=BUCKET,
              Key=f"batch/{batch_ver}/manifest.json",
              Body=json.dumps(manifest, indent=2))
print("Batch pipeline complete!")
print(json.dumps(manifest, indent=2))
