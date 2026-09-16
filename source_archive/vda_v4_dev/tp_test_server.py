#!/usr/bin/env python3
"""Static server for TheoreticallyPose browser testing from the PC.
GET serves /Users/tim/Desktop; PUT /upload/<name> saves exported files
back to the scratchpad so ffprobe verification happens on the Mac."""
import http.server, os, re, socketserver

DESKTOP = '/Users/tim/Desktop'
UPLOADS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOADS, exist_ok=True)

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=DESKTOP, **kw)

    def do_PUT(self):
        m = re.fullmatch(r'/upload/([A-Za-z0-9._-]{1,80})', self.path)
        if not m or '..' in m.group(1):
            self.send_error(403)
            return
        n = int(self.headers.get('Content-Length', 0))
        if n <= 0 or n > 800 * 1024 * 1024:
            self.send_error(411)
            return
        dest = os.path.join(UPLOADS, m.group(1))
        remaining, chunk = n, 1 << 20
        with open(dest, 'wb') as f:
            while remaining > 0:
                data = self.rfile.read(min(chunk, remaining))
                if not data:
                    break
                f.write(data)
                remaining -= len(data)
        self.send_response(200)
        self.send_header('Content-Length', '2')
        self.end_headers()
        self.wfile.write(b'ok')
        print(f'[upload] {dest} ({n} bytes)', flush=True)

with socketserver.ThreadingTCPServer(('0.0.0.0', 8123), Handler) as httpd:
    httpd.allow_reuse_address = True
    print('serving Desktop on 0.0.0.0:8123, uploads ->', UPLOADS, flush=True)
    httpd.serve_forever()
