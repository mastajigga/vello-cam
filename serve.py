#!/usr/bin/env python3
"""Petit serveur statique qui sert .wasm avec le bon Content-Type."""
import http.server, socketserver, functools, mimetypes, sys
mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("text/javascript", ".js")
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
Handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=sys.argv[2] if len(sys.argv) > 2 else ".")
class S(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
print(f"http://localhost:{PORT}", flush=True)
S(("0.0.0.0", PORT), Handler).serve_forever()
