# -*- coding: utf-8 -*-
"""Simple password auth for gemma-stt.

One shared password (APP_PASSWORD env or .env) => an HMAC-SHA256-signed HttpOnly
session cookie. No extra dependencies. Wrong attempts are throttled globally.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from pathlib import Path

COOKIE_NAME = "stt_session"
SESSION_TTL = 30 * 24 * 3600  # 30 days
_definition: dict = {}  # set by init_auth: {secret, pending}/{fails, first} throttle

MAX_ATTEMPTS = 8        # password attempts...
WINDOW = 600            # ...per this window (seconds), overall (single-user app)


def _b64(data: bytes) -> bytes:
    return base64.urlsafe_b64encode(data).rstrip(b"=")


def _unb64(data: str) -> bytes:
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))


def load_secret(path) -> bytes:
    """Lazily generate + persist a random signing secret."""
    f = Path(path) / "auth_secret"
    if f.exists():
        return bytes.fromhex(f.read_text().strip())
    key = secrets.token_bytes(32)
    f.parent.mkdir(exist_ok=True)
    f.write_text(key.hex())
    return key


def init_auth(password: str, data_dir) -> None:
    _definition["secret"] = load_secret(data_dir)
    _definition["password"] = password
    _definition["fails"] = 0
    _definition["window_start"] = 0.0


def _token(secret: bytes, now: float) -> str:
    payload = json.dumps({"exp": int(now + SESSION_TTL), "k": secrets.token_hex(4)}).encode()
    return (_b64(payload) + b"." + _b64(_sign(secret, payload))).decode()


def new_session() -> tuple[str, int]:
    """(cookie, max_age) for a fresh login."""
    return _token(_definition["secret"], time.time()), SESSION_TTL


def _sign(secret: bytes, payload: bytes) -> bytes:
    return hmac.new(secret, payload, hashlib.sha256).digest()


def verify(token: str | None) -> bool:
    if not token or "secret" not in _definition:
        return False
    try:
        payload_b64, sig_b64 = token.split(".", 1)
        if not hmac.compare_digest(_sign(_definition["secret"], _unb64(payload_b64)), _unb64(sig_b64)):
            return False
        exp = json.loads(_unb64(payload_b64))["exp"]
    except Exception:
        return False
    return exp > time.time()


def _allowed_to_attempt() -> bool:
    now = time.time()
    if now - _definition["window_start"] > WINDOW:  # reset window
        _definition["window_start"] = now
        _definition["fails"] = 0
    return _definition["fails"] < MAX_ATTEMPTS


def _register_fail() -> None:
    now = time.time()
    if now - _definition["window_start"] > WINDOW:
        _definition["window_start"] = now
    _definition["fails"] = _definition["fails"] + 1


def check_password(pw: str) -> bool:
    """True only on correct password within the throttle window. Safely constant-time."""
    if not _allowed_to_attempt():
        return False
    ok = hmac.compare_digest((pw or "").encode(), _definition["password"].encode())
    if not ok:
        _register_fail()
    return ok


def attempts_left() -> int:
    return max(0, MAX_ATTEMPTS - _definition["fails"])
