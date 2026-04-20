import os, json, time, random, requests, boto3, pandas as pd
from datetime import datetime
from io import BytesIO
from botocore.client import Config

ENDPOINT   = os.getenv("REWRITE_URL",      "http://localhost:8000/rewrite")
BUCKET     = os.getenv("MINIO_BUCKET",     "zulip-rewriter")
ENDPOINT_S3 = os.getenv("MINIO_ENDPOINT",   "https://129.114.27.192.nip.io") # Added https
ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
RATE_SEC   = float(os.getenv("RATE_SECONDS", "2"))
VERSION    = os.getenv("DATA_VERSION",     "v1")

s3 = boto3.client(
    "s3", 
    endpoint_url=ENDPOINT_S3, 
    aws_access_key_id=ACCESS_KEY, 
    aws_secret_access_key=SECRET_KEY,
    verify=False, # Bypasses the SSL error for .nip.io
    config=Config(
        signature_version='s3v4',
        s3={'addressing_style': 'path'}
    )
)

print("Loading training data from MinIO...")
obj = s3.get_object(Bucket=BUCKET, Key=f"raw/{VERSION}/train.parquet")
df  = pd.read_parquet(BytesIO(obj["Body"].read()))
texts = df["text"].dropna().tolist()
print(f"Loaded {len(texts)} messages")

STREAMS = ["general", "python", "help", "backend", "frontend"]
TOPICS  = ["debugging", "code review", "question", "bug report"]
STYLES  = ["formal", "informal"]
USERS   = [f"user_{i}" for i in range(1, 51)]

log_buffer = []
print(f"Generating requests every {RATE_SEC}s — Ctrl+C to stop\n")

try:
    while True:
        payload = {
            "message_id":       f"msg_{random.randint(1000,9999)}",
            "original_message": random.choice(texts),
            "rewrite_style":    random.choice(STYLES),
            "context": {
                "stream":    random.choice(STREAMS),
                "topic":     random.choice(TOPICS),
                "sender_id": random.choice(USERS),
            },
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        try:
            resp   = requests.post(ENDPOINT, json=payload, timeout=3)
            status = resp.status_code
            body   = resp.json() if resp.ok else {}
        except Exception as e:
            status = 0
            body   = {"error": str(e)}

        log_buffer.append({"input": payload, "output": body, "http_status": status})
        print(f"[{payload['timestamp']}] style={payload['rewrite_style']} status={status}")

        if len(log_buffer) >= 50:
            ts  = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
            key = f"online_logs/{ts}.json"
            s3.put_object(Bucket=BUCKET, Key=key, Body=json.dumps(log_buffer))
            print(f"  Flushed {len(log_buffer)} entries to MinIO: {key}")
            log_buffer = []

        time.sleep(RATE_SEC)

except KeyboardInterrupt:
    if log_buffer:
        ts  = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        key = f"online_logs/{ts}_final.json"
        s3.put_object(Bucket=BUCKET, Key=key, Body=json.dumps(log_buffer))
        print(f"Flushed {len(log_buffer)} remaining entries")
    print("Generator stopped.")
