import sqlite3
import threading
from pathlib import Path

SCHEMA = """
create table if not exists users (
    id integer primary key, name text unique not null,
    pw_hash blob not null, salt blob not null, role text not null default 'user');
create table if not exists sessions (
    token_hash text primary key, user_id integer not null references users(id), created real not null);
create table if not exists documents (
    id integer primary key, owner_id integer not null references users(id),
    filename text not null, chunks integer not null, created real not null);
create table if not exists conversations (
    id integer primary key, user_id integer not null references users(id), created real not null);
create table if not exists messages (
    id integer primary key, conversation_id integer not null references conversations(id),
    role text not null, content text not null, created real not null);
create table if not exists answers (
    id integer primary key, conversation_id integer not null references conversations(id),
    question text not null, answer text not null, sources text not null, grounded integer not null,
    status text not null default 'pending', reviewer_id integer, note text, created real not null);
"""


class DB:
    """Thin sqlite wrapper. One connection behind a lock: FastAPI runs sync endpoints on a thread pool."""

    def __init__(self, path: Path):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.lock:
            self.conn.executescript(SCHEMA)

    def run(self, sql: str, params: tuple = ()) -> int:
        with self.lock:
            cur = self.conn.execute(sql, params)
            self.conn.commit()
            return cur.lastrowid

    def all(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self.lock:
            return self.conn.execute(sql, params).fetchall()

    def one(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        with self.lock:
            return self.conn.execute(sql, params).fetchone()
