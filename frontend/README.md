# PhishLens detection UI (P track)

Static page, no build step. Open `frontend/index.html` directly or serve the repo via GitHub Pages.

- `index.html` — form (subject + body), result, error/loading regions.
- `styles.css` — responsive styling, no framework.
- `app.js` — API integration. Central config: `PHISHLENS_API_BASE = "https://phishlens-api-tlx8.onrender.com"`.
  Sends `POST /api/predict` with `{"text": "Subject: <s>\n\n<body>"}`. Renders everything via `textContent`.
- Root `index.html` checklist is untouched.

Security: email/API content is untrusted text, never `innerHTML`/`eval`. 90s timeout for Render cold start, duplicate-submit guard, friendly errors for 400/422/500/network/timeout.
