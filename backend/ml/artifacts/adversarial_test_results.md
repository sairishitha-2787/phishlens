# Adversarial Test Results — model_v2.pkl via live API (P track)

Date: 2026-09-12

Tester: P (frontend/security/adversarial testing)

Model: `svm-tfidf-v2-20260911` (served as `model_v2.pkl`)

API: `POST https://phishlens-api-tlx8.onrender.com/api/predict` with `{"text": "Subject: <subject>\n\n<body>"}`

Method: `backend/ml/adversarial_probe.py` (read-only, no model changes). Raw JSON: `adversarial_probe_raw.json`.

Note: the 3-way model has no generic "suspicious" label — either `ai_phishing` or `human_phishing` counts as a phishing verdict.

## Summary

| Test | Expected | Actual | Confidence | Pass/Fail |
|------|----------|--------|------------|-----------|
| 1 — Obvious phishing | PHISHING | `ai_phishing` | 48% | PASS |
| 2 — Social-engineering phishing | PHISHING / SUSPICIOUS | `human_phishing` | 47% | PASS |
| 3 — AI-style legitimate | LEGITIMATE | `legitimate` | 48% | PASS |
| 4 — Phishing without obvious keywords | PHISHING / SUSPICIOUS | `ai_phishing` | 44% | PASS |
| 5 — Legitimate security email | LEGITIMATE | `human_phishing` | 42% | FAIL |

Score: 4/5. All confidences are low (42–48%), i.e. softmax over SVM margins, uncalibrated — consistent with the known R note.

## Per-test detail

### Test 1 — Obvious phishing (PASS)

- Subject: `Urgent Account Verification`
- Expected: PHISHING.
- Actual: `ai_phishing` @ 0.482 (probs: legit 0.059 / ai 0.482 / human 0.459).
- Top features: `secure, verify, subject urgent, urgent, link, contains 1 link, 9 urgency cues`.
- Observation: Correct verdict, but margin between `ai_phishing` and `human_phishing` is tiny (0.023). Urgency + link signals fire as designed.

### Test 2 — Social-engineering phishing (PASS)

- Subject: `Quick favour before the meeting`
- Expected: PHISHING / SUSPICIOUS.
- Actual: `human_phishing` @ 0.472 (legit 0.207 / ai 0.321 / human 0.472).
- Top features: `document, attached, subject, the attached, here http, contains 1 link`.
- Observation: Recorded honestly — model catches the polite BEC-style lure, no urgency keywords needed. Plausible `human_phishing` (not ai) assignment.

### Test 3 — AI-style legitimate (PASS — key confound test)

- Subject: `Weekly Project Update`
- Expected: LEGITIMATE.
- Actual: `legitimate` @ 0.483 (legit 0.483 / ai 0.243 / human 0.274).
- Top features: `weekly, integration, update regarding, discuss, regarding this`.
- Observation: Most important test — v1 had 93.5% false alarms on AI-written legitimate mail; v2 correctly returns legitimate here. Confound fix holds on this sample.

### Test 4 — Phishing without obvious keywords (PASS)

- Subject: `Document available for review`
- Expected: PHISHING / SUSPICIOUS.
- Actual: `ai_phishing` @ 0.440 (legit 0.188 / ai 0.440 / human 0.372).
- Top features: `document, the document, available for, any issues, portal, contains 1 link`.
- Observation: No "verify/suspend/urgent" words, yet link + portal phrasing still triggers a phishing verdict. Good.

### Test 5 — Legitimate security email (FAIL — honest false positive)

- Subject: `Security awareness training`
- Expected: LEGITIMATE.
- Actual: `human_phishing` @ 0.421 (legit 0.329 / ai 0.250 / human 0.421).
- Top features: `account, subject, awareness, links, tomorrow`.
- Observation: False positive. Security-vocabulary (`phishing, malware, suspicious links, password safety, account protection`) pushes a benign training notice into `human_phishing`, though legitimate is a close second (0.329). Flagged for R: consider security-awareness training mails as an additional control set; do NOT retrain here (P scope). Low confidence (42%) means the UI should present this as uncertain, not certain.

## Notes for R (no action taken, P scope only)

- Overall calibration is flat (all 42–48%) — UI shows raw % without claiming calibration.
- Test 5 suggests vocabulary sensitivity worth a follow-up control set. No model/backend files were modified.
