"""P-track adversarial probe vs live Render API (read-only, no model changes).
Run: python backend/ml/adversarial_probe.py
Writes JSON to backend/ml/artifacts/adversarial_probe_raw.json
"""
import json, urllib.request, urllib.error
from pathlib import Path

API = "https://phishlens-api-tlx8.onrender.com/api/predict"
OUT = Path(__file__).resolve().parents[0] / "artifacts" / "adversarial_probe_raw.json"

TESTS = [
    {"n": 1, "name": "Obvious phishing", "expected": "PHISHING",
     "subject": "Urgent Account Verification",
     "body": "Dear Customer,\n\nYour account has been temporarily suspended due to suspicious activity.\n\nPlease verify your account immediately using the link below:\n\nhttp://secure-account-verification.example.com\n\nFailure to verify within 24 hours will result in permanent suspension.\n\nRegards,\nSecurity Team"},
    {"n": 2, "name": "Social-engineering phishing", "expected": "PHISHING / SUSPICIOUS",
     "subject": "Quick favour before the meeting",
     "body": "Hi John,\n\nCould you please review the attached document before our meeting?\n\nYou can access it here:\nhttp://document-review.example.com\n\nThanks,\nSarah"},
    {"n": 3, "name": "AI-style legitimate", "expected": "LEGITIMATE",
     "subject": "Weekly Project Update",
     "body": "Hi Team,\n\nI wanted to share a quick update regarding this week's project progress.\n\nThe frontend implementation is progressing as planned, and we are currently reviewing the remaining integration tasks.\n\nWe'll discuss the outstanding items during tomorrow's meeting.\n\nBest,\nAlex"},
    {"n": 4, "name": "Phishing without obvious keywords", "expected": "PHISHING / SUSPICIOUS",
     "subject": "Document available for review",
     "body": "Hi,\n\nThe document you requested is now available.\n\nPlease access it through the following portal:\n\nhttp://document-portal.example.com\n\nLet me know if you have any issues.\n\nThanks,\nMichael"},
    {"n": 5, "name": "Legitimate security email", "expected": "LEGITIMATE",
     "subject": "Security awareness training",
     "body": "Hi everyone,\n\nAs part of our security awareness program, we'll be discussing phishing, malware, suspicious links, password safety, and account protection.\n\nThe training session will be held tomorrow at 10 AM.\n\nRegards,\nIT Team"},
]

def call(text, timeout=120):
    req = urllib.request.Request(API, data=json.dumps({"text": text}).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode())
        except Exception:
            return e.code, {"error": "http-error-no-json"}
    except Exception as e:
        return -1, {"error": type(e).__name__ + ": " + str(e)}

results = []
for t in TESTS:
    text = "Subject: %s\n\n%s" % (t["subject"], t["body"])
    status, data = call(text)
    rec = {"n": t["n"], "name": t["name"], "subject": t["subject"],
           "expected": t["expected"], "http": status,
           "label": data.get("label"), "confidence": data.get("confidence"),
           "probabilities": data.get("probabilities"),
           "top_features": (data.get("explanation") or {}).get("top_features"),
           "model_version": data.get("model_version"), "raw": data}
    results.append(rec)
    print("TEST %d %s -> http=%s label=%s conf=%s" % (t["n"], t["name"], status, rec["label"], rec["confidence"]))

OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
print("wrote", OUT)
