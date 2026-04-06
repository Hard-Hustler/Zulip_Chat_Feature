import os, json, re, boto3
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List

app = FastAPI(title="Zulip Message Rewriter - Online Feature Service")

BUCKET     = os.getenv("MINIO_BUCKET",     "zulip-rewriter")
ENDPOINT   = os.getenv("MINIO_ENDPOINT",   "http://localhost:9000")
ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")

s3 = boto3.client("s3", endpoint_url=ENDPOINT,
                  aws_access_key_id=ACCESS_KEY,
                  aws_secret_access_key=SECRET_KEY)

POLITE_MARKERS   = [r"\bplease\b",r"\bthank\b",r"\bcould you\b",
                    r"\bwould you\b",r"\bi appreciate\b",r"\bkindly\b"]
INFORMAL_MARKERS = [r"\bhey\b",r"\byo\b",r"\bu\b",r"\bgonna\b",
                    r"\bwanna\b",r"\bbtw\b",r"\bomg\b",r"\blol\b"]

def extract_features(text: str) -> dict:
    tl = text.lower()
    words = tl.split()
    polite   = sum(1 for p in POLITE_MARKERS   if re.search(p, tl))
    informal = sum(1 for p in INFORMAL_MARKERS if re.search(p, tl))
    return {
        "word_count":            len(words),
        "char_count":            len(text),
        "polite_marker_count":   polite,
        "informal_marker_count": informal,
        "has_question_mark":     int("?" in text),
        "has_exclamation":       int("!" in text),
        "estimated_formality":   round((polite - informal) / max(len(words), 1), 4),
    }

class RewriteRequest(BaseModel):
    message_id:       str
    original_message: str
    rewrite_style:    str
    context:          dict = {}
    timestamp:        Optional[str] = None

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/features")
def compute_features(req: RewriteRequest):
    feats = extract_features(req.original_message)
    return {"message_id": req.message_id, "features": feats}

@app.post("/rewrite")
def rewrite(req: RewriteRequest):
    feats = extract_features(req.original_message)
    msg   = req.original_message

    if req.rewrite_style == "formal":
        msg = msg.replace("hey", "Hello").replace("u ", "you ")
        msg = msg.replace("gonna", "going to").replace("wanna", "want to")
        msg = msg.replace("btw", "by the way").replace("omg", "oh my")
        if not msg.endswith("?") and not msg.endswith("."):
            msg = msg + "."
        if not any(msg.lower().startswith(p) for p in ["could","would","please","hello"]):
            msg = "Could you please " + msg[0].lower() + msg[1:]
    else:
        msg = msg.replace("Hello", "hey").replace("Could you please", "hey can you")
        msg = msg.replace("going to", "gonna").replace("want to", "wanna")

    result = {
        "message_id":        req.message_id,
        "original_message":  req.original_message,
        "rewrite_style":     req.rewrite_style,
        "rewritten_message": msg,
        "politeness_score":  round(feats["estimated_formality"] + 0.5, 3),
        "features":          feats,
        "model_version":     "v1.0",
    }

    try:
        key = f"feature_logs/{req.timestamp or 'unknown'}_{req.message_id}.json"
        s3.put_object(Bucket=BUCKET, Key=key, Body=json.dumps(result))
    except Exception:
        pass

    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
