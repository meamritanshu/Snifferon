import re

with open('app.py', 'r') as f:
    content = f.read()

# Fix CORS
content = content.replace("cors_allowed_origins=\"*\"", "cors_allowed_origins=[\"http://localhost:5001\", \"http://127.0.0.1:5001\"]")

# Add html import if not there
if 'import html' not in content:
    content = content.replace("from flask import", "import html\nfrom flask import")

# Sanitize qname
# Find: if packet[DNS].qr == 0 and packet[DNS].qd: qname = packet[DNS].qd.qname.decode().rstrip('.')
# Replace: if packet[DNS].qr == 0 and packet[DNS].qd: qname = html.escape(packet[DNS].qd.qname.decode().rstrip('.'))

content = content.replace("qname = packet[DNS].qd.qname.decode().rstrip('.')", "qname = html.escape(packet[DNS].qd.qname.decode('utf-8', errors='replace').rstrip('.'))")

with open('app.py', 'w') as f:
    f.write(content)
