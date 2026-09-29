"""Local operator accounts and session tokens for the Inspector service.

Accounts live in inspector/users.json as PBKDF2 hashes, never the password.
A token is base64url(claims) + "." + base64url(HMAC-SHA256 of the claims),
signed with a key made fresh at every service start: a restart signs everyone
out, and the key never touches disk. Same strings as the board's auth:
"invalid credentials" for a failed login, "unauthorized" for a bad token.

Create an account:  python inspector/auth.py add-user <name>
"""
import base64
import getpass
import hashlib
import hmac
import json
import os
import secrets
import sys
import time
from pathlib import Path

USERS_FILE = Path(__file__).parent / "users.json"
ITERATIONS = 600_000           # PBKDF2-SHA256, same count as the board's credential store
TOKEN_TTL = 8 * 3600           # one shift
_KEY = secrets.token_bytes(32)


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, ITERATIONS)
    return f"pbkdf2_sha256${ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, iterations, salt_hex, digest_hex = stored.split("$")
        if scheme != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac("sha256", password.encode(),
                                     bytes.fromhex(salt_hex), int(iterations))
    except ValueError:
        return False
    # bytes, not str: compare_digest raises TypeError on non-ASCII str
    return hmac.compare_digest(digest.hex().encode(), digest_hex.encode())


def load_users(path: Path = USERS_FILE) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def add_user(name: str, password: str, path: Path = USERS_FILE) -> None:
    users = load_users(path)
    users[name] = hash_password(password)
    path.write_text(json.dumps(users, indent=2))
    try:
        os.chmod(path, 0o600)  # owner only on Linux; a no-op on Windows
    except OSError:
        pass


def check_login(name: str, password: str, path: Path = USERS_FILE) -> bool:
    stored = load_users(path).get(name)
    if stored is None:
        hash_password(password)  # same cost as a real check: timing must not reveal which names exist
        return False
    return verify_password(password, stored)


def issue_token(name: str, now=None, ttl: int = TOKEN_TTL, key: bytes = None) -> str:
    now = int(time.time() if now is None else now)
    payload = _b64(json.dumps({"sub": name, "iat": now, "exp": now + ttl}).encode())
    tag = _b64(hmac.new(key or _KEY, payload.encode(), hashlib.sha256).digest())
    return f"{payload}.{tag}"


def check_token(header, now=None, key: bytes = None):
    """The account name behind an 'Authorization: Bearer <token>' header, or None."""
    scheme, _, token = (header or "").partition(" ")
    if scheme.lower() != "bearer":
        return None
    payload, _, tag = token.strip().partition(".")
    want = _b64(hmac.new(key or _KEY, payload.encode(), hashlib.sha256).digest())
    # bytes, not str: compare_digest raises TypeError on a non-ASCII header
    if not tag or not hmac.compare_digest(tag.encode(), want.encode()):
        return None
    try:
        claims = json.loads(_unb64(payload))
    except ValueError:
        return None
    if claims.get("exp", 0) <= (time.time() if now is None else now):
        return None
    return claims.get("sub")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "add-user":
        sys.exit("usage: python inspector/auth.py add-user <name>")
    name = sys.argv[2]
    password = getpass.getpass(f"password for {name}: ")
    if not password or password != getpass.getpass("again: "):
        sys.exit("empty password, or the two entries differ")
    add_user(name, password)
    print(f"added {name} to {USERS_FILE}")
