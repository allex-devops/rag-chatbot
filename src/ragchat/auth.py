import hashlib
import hmac
import os
import secrets


def hash_password(password: str, salt: bytes | None = None) -> tuple[bytes, bytes]:
    salt = salt or os.urandom(16)
    return hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1), salt


def verify_password(password: str, expected: bytes, salt: bytes) -> bool:
    got, _ = hash_password(password, salt)
    return hmac.compare_digest(got, expected)


def new_token() -> str:
    return secrets.token_urlsafe(32)


def token_hash(token: str) -> str:
    # only the hash goes in the database, so a leaked db file doesn't hand out live sessions
    return hashlib.sha256(token.encode()).hexdigest()
