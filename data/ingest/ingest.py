import os, json, boto3, pandas as pd
from convokit.model import Corpus
from convokit.util import download
from sklearn.model_selection import train_test_split

BUCKET     = os.getenv("MINIO_BUCKET",     "zulip-rewriter")
ENDPOINT   = os.getenv("MINIO_ENDPOINT",   "https://129.114.27.192.nip.io")
ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
VERSION    = os.getenv("DATA_VERSION",     "v1")

print("Step 1: Downloading Stack Exchange Politeness Corpus...")
corpus = Corpus(filename=download("stack-exchange-politeness-corpus"))
print(f"Loaded {len(list(corpus.iter_utterances()))} utterances")

print("Step 2: Extracting utterances...")
rows = []
for utt in corpus.iter_utterances():
    meta = utt.meta
    rows.append({
        "id":               utt.id,
        "text":             utt.text,
        "normalized_score": meta.get("Normalized Score", None),
        "binary_label":     meta.get("Binary", None),
        "community":        meta.get("community", ""),
        "split":            meta.get("split", "train"),
        "synthetic":        False,
    })

df = pd.DataFrame(rows)
print(f"Extracted {len(df)} rows")
print(df["binary_label"].value_counts())

print("Step 3: Synthetic augmentation...")
polite_prefixes   = ["Could you please ", "Would you mind ", "I would appreciate if "]
informal_prefixes = ["hey ", "just ", "yo can you "]

synthetic = []
for _, row in df.iterrows():
    if pd.isna(row["binary_label"]): continue
    if row["binary_label"] == 1:
        for pfx in polite_prefixes:
            synthetic.append({**row, "text": pfx + row["text"], "synthetic": True})
    elif row["binary_label"] == -1:
        for pfx in informal_prefixes:
            synthetic.append({**row, "text": pfx + row["text"], "synthetic": True})

df_aug = pd.concat([df, pd.DataFrame(synthetic)], ignore_index=True)
print(f"After augmentation: {len(df_aug)} rows ({len(synthetic)} synthetic)")

print("Step 4: Train/val/test split...")
df_train, df_temp = train_test_split(df_aug, test_size=0.2, random_state=42,
                                      stratify=df_aug["binary_label"])
df_val, df_test   = train_test_split(df_temp, test_size=0.5, random_state=42,
                                      stratify=df_temp["binary_label"])
print(f"Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")

print("Step 5: Uploading to MinIO...")
s3 = boto3.client(
    "s3", 
    endpoint_url=ENDPOINT, 
    aws_access_key_id=ACCESS_KEY, 
    aws_secret_access_key=SECRET_KEY,
    verify=False  # This allows connection to the .nip.io endpoint without SSL errors
)
try:
    s3.create_bucket(Bucket=BUCKET)
    print(f"Bucket '{BUCKET}' created")
except Exception:
    print(f"Bucket '{BUCKET}' already exists")

def upload(df, name):
    path = f"/tmp/{name}.parquet"
    df.to_parquet(path, index=False)
    key = f"raw/{VERSION}/{name}.parquet"
    s3.upload_file(path, BUCKET, key)
    print(f"  Uploaded {key} ({len(df)} rows)")

upload(df_train, "train")
upload(df_val,   "val")
upload(df_test,  "test")
upload(df_aug,   "full")

manifest = {
    "version":        VERSION,
    "source":         "stack-exchange-politeness-corpus (ConvoKit CC BY 4.0)",
    "total_rows":     len(df_aug),
    "original_rows":  len(df),
    "synthetic_rows": len(synthetic),
    "splits":         {"train": len(df_train), "val": len(df_val), "test": len(df_test)},
    "schema": {
        "id":               "string - utterance ID",
        "text":             "string - message text",
        "normalized_score": "float - politeness score",
        "binary_label":     "int - 1=polite, 0=neutral, -1=impolite",
        "community":        "string - StackExchange community",
        "split":            "string - train/test",
        "synthetic":        "bool - True if augmented"
    },
    "ingested_at": pd.Timestamp.utcnow().isoformat(),
}
with open("/tmp/manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
s3.upload_file("/tmp/manifest.json", BUCKET, f"raw/{VERSION}/manifest.json")
print("Manifest uploaded.")
print("Ingestion complete!")
