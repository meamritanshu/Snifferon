## 2024-10-26 - [XSS and CSWSH]
**Vulnerability:** Found Cross-Site WebSocket Hijacking (CSWSH) via `cors_allowed_origins="*"` in SocketIO initialization, and Cross-Site Scripting (XSS) via unescaped DNS `qname` rendering.
**Learning:** `flask-socketio` uses unrestricted CORS by default if `cors_allowed_origins="*"` is specified. DNS packet domains may contain malicious payloads which, if rendered directly via JavaScript, can lead to XSS.
**Prevention:** Always restrict `cors_allowed_origins` to known domain origins, particularly `localhost` or `127.0.0.1` for local tools. Always HTML-escape external unvalidated inputs like DNS queries before sending to the UI, or properly sanitize it in the frontend.
