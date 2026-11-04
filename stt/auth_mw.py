# -*- coding: utf-8 -*-
"""Auth middleware: everything requires the session cookie except the login page."""
from __future__ import annotations

import urllib.parse  # noqa: F401  (used by RedirectResponse flows in server)

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import RedirectResponse, PlainTextResponse

from . import auth

LOGIN_PAGE = """<!doctype html>
<html><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
<title>gemma-stt · entrar</title>
<style>
body{font-family:system-ui,sans-serif;background:#111;color:#eee;display:grid;place-items:center;height:100vh;margin:0}
form{background:#1c1c1c;padding:2rem;border-radius:12px;display:flex;flex-direction:column;gap:1rem;min-width:280px}
h1{font-size:1.1rem;margin:0;text-align:center}
input{padding:.7rem;border-radius:8px;border:1px solid #444;background:#222;color:#eee;font-size:1rem}
button{padding:.7rem;border-radius:8px;border:0;background:#3b82f6;color:#fff;font-weight:600;cursor:pointer}
.err{color:#f87171;font-size:.85rem;min-height:1.2em;text-align:center}
</style></head>
<body><form method=\"post\" action=\"/login\">
<h1>◈ gemma-stt</h1>
<p class=\"err\">@ERROR@</p>
<input type=\"password\" name=\"password\" placeholder=\"senha\" autofocus required autocomplete=\"current-password\">
<input type=\"hidden\" name=\"next\" value=\"@NEXT@\">
<button>Entrar</button>
</form></body></html>"""

PUBLIC = {"/login", "/logout", "/healthz"}  # login GET+POST, logout, tunnel health check


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path in PUBLIC:
            if path == "/login":
                if auth.verify(request.cookies.get(auth.COOKIE_NAME)) and request.method == "GET":
                    nxt = request.query_params.get("next", "/")
                    return RedirectResponse(_safe_next(nxt), status_code=302)
                return await call_next(request)
            if path == "/healthz":
                return PlainTextResponse("ok")
            return await call_next(request)
        if auth.verify(request.cookies.get(auth.COOKIE_NAME)):
            return await call_next(request)
        wants_html = "text/html" in (request.headers.get("accept") or "")
        if wants_html:
            return RedirectResponse(
                "/login?next=" + urllib.parse.quote(path, safe="/"), status_code=302
            )
        return PlainTextResponse("unauthorized", status_code=401)


def _safe_next(nxt: str) -> str:
    # only same-origin absolute paths, never "http..." — prevents open redirect
    if not nxt.startswith("/") or nxt.startswith("//"):
        return "/"
    return "/" + nxt.lstrip("/")
