import json
import logging
import os
import re
import sqlite3
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from fastapi import Cookie, Depends, FastAPI, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from shared.llm import LLM, LLMError

from .auth import hash_password, new_token, token_hash, verify_password
from .db import DB
from .ingest import chunks_from_file
from .rag import DEFAULT_MIN_SCORE, Embedder, answer, default_min_score, embedder, llm_from_env
from .store import EmbeddingMismatch, Store

log = logging.getLogger("ragchat")

STATIC = Path(__file__).parent / "static"
ALLOWED_SUFFIXES = {".pdf", ".txt", ".md"}
MAX_UPLOAD = 20 * 1024 * 1024
SESSION_SECONDS = 7 * 24 * 3600
HISTORY_TURNS = 6  # messages sent back to the model as memory, i.e. the last three exchanges
# checked against when the name isn't found, so a wrong name and a wrong password take equally long
_DUMMY_HASH, _DUMMY_SALT = hash_password("not a real password")


class LoginThrottle:
    """Too many failed logins from one address for one name earns a wait, so guessing passwords is slow."""

    def __init__(self, max_failures: int = 10, window: float = 300.0, clock=time.monotonic):
        self.max_failures, self.window, self.clock = max_failures, window, clock
        self.failures: dict[tuple[str, str], deque] = defaultdict(deque)

    def _recent(self, key) -> deque:
        q = self.failures[key]
        while q and self.clock() - q[0] > self.window:
            q.popleft()
        return q

    def retry_after(self, key) -> int:
        q = self._recent(key)
        if len(q) < self.max_failures:
            return 0
        return max(1, int(self.window - (self.clock() - q[0])))

    def failed(self, key) -> None:
        self._recent(key).append(self.clock())

    def succeeded(self, key) -> None:
        self.failures.pop(key, None)


def strip_citations(text: str) -> str:
    # "[1]" in an old answer points at that turn's passages, and the next turn reuses the same numbers
    return re.sub(r"\s*\[\d+\]", "", text)


class Credentials(BaseModel):
    name: str = Field(pattern=r"^[A-Za-z0-9_-]{3,32}$")
    password: str = Field(min_length=8, max_length=200)


class ChatIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    conversation_id: int | None = None


class ReviewIn(BaseModel):
    decision: Literal["approved", "rejected"]
    note: str = Field("", max_length=1000)


def create_app(
    data_dir: Path | str,
    llm: LLM | None = None,
    embed: Embedder | None = None,
    expert_names: tuple[str, ...] = (),
    min_score: float = DEFAULT_MIN_SCORE,
    embed_model: str | None = None,
    throttle: LoginThrottle | None = None,
) -> FastAPI:
    data_dir = Path(data_dir)
    llm = llm or llm_from_env()  # fails at startup, before touching disk, if the provider isn't configured
    embed_model = embed_model or getattr(llm, "embed_model", None)
    (data_dir / "uploads").mkdir(parents=True, exist_ok=True)
    db = DB(data_dir / "app.db")
    store = Store(data_dir / "index", embed_model=embed_model)  # refuses to start on an index from another model
    embed = embed or embedder(llm)
    app = FastAPI(title="ragchat")
    throttle = throttle or LoginThrottle()

    def start_session(response: Response, user_id: int) -> None:
        token = new_token()
        db.run("insert into sessions values (?, ?, ?)", (token_hash(token), user_id, time.time()))
        # no Secure flag so it works on http://localhost; put it behind https before exposing it
        response.set_cookie("session", token, httponly=True, samesite="lax", max_age=SESSION_SECONDS)

    def current_user(session: str | None = Cookie(default=None)):
        row = None
        if session:
            row = db.one(
                "select u.id, u.name, u.role from sessions s join users u on u.id = s.user_id "
                "where s.token_hash = ? and s.created > ?",
                (token_hash(session), time.time() - SESSION_SECONDS),
            )
        if row is None:
            raise HTTPException(status_code=401, detail="log in first")
        return row

    def expert_only(user=Depends(current_user)):
        if user["role"] != "expert":
            raise HTTPException(status_code=403, detail="experts only")
        return user

    @contextmanager
    def upstream():
        """Turn a model outage or a mismatched index into a clean error instead of a bare 500."""
        try:
            yield
        except LLMError as e:
            log.warning("model service failed: %s", e)  # the detail can hold upstream bodies, keep it in the log
            raise HTTPException(status_code=502, detail="the model service is unavailable")
        except EmbeddingMismatch as e:
            raise HTTPException(status_code=409, detail=str(e))

    def own_conversation(conv_id: int, user) -> None:
        # 404 rather than 403 so ids of other people's conversations aren't confirmed to exist
        if not db.one("select 1 from conversations where id = ? and user_id = ?", (conv_id, user["id"])):
            raise HTTPException(status_code=404, detail="no such conversation")

    @app.post("/signup", status_code=201)
    def signup(creds: Credentials, response: Response):
        pw_hash, salt = hash_password(creds.password)
        role = "expert" if creds.name in expert_names else "user"
        try:
            user_id = db.run(
                "insert into users (name, pw_hash, salt, role) values (?, ?, ?, ?)", (creds.name, pw_hash, salt, role)
            )
        except sqlite3.IntegrityError:  # the unique constraint on name
            raise HTTPException(status_code=409, detail="that name is taken")
        start_session(response, user_id)
        return {"name": creds.name, "role": role}

    @app.post("/login")
    def login(creds: Credentials, request: Request, response: Response):
        key = (request.client.host if request.client else "?", creds.name)
        wait = throttle.retry_after(key)
        if wait:
            raise HTTPException(
                status_code=429, detail="too many failed attempts, try again later", headers={"Retry-After": str(wait)}
            )
        row = db.one("select * from users where name = ?", (creds.name,))
        ok = verify_password(creds.password, row["pw_hash"] if row else _DUMMY_HASH, row["salt"] if row else _DUMMY_SALT)
        if not (row and ok):
            throttle.failed(key)
            raise HTTPException(status_code=401, detail="wrong name or password")
        throttle.succeeded(key)
        start_session(response, row["id"])
        return {"name": row["name"], "role": row["role"]}

    @app.post("/logout")
    def logout(response: Response, session: str | None = Cookie(default=None)):
        if session:
            db.run("delete from sessions where token_hash = ?", (token_hash(session),))
        response.delete_cookie("session")
        return {"ok": True}

    @app.get("/me")
    def me(user=Depends(current_user)):
        return {"name": user["name"], "role": user["role"]}

    @app.post("/documents", status_code=201)
    def upload(file: UploadFile = File(...), user=Depends(current_user)):
        # basename only, so "../../x" can't write outside this user's folder
        name = re.sub(r"[^A-Za-z0-9._ -]", "_", Path(file.filename or "").name)
        if Path(name).suffix.lower() not in ALLOWED_SUFFIXES:
            raise HTTPException(status_code=400, detail="only .pdf, .txt and .md files are supported")
        data = file.file.read(MAX_UPLOAD + 1)
        if len(data) > MAX_UPLOAD:
            raise HTTPException(status_code=413, detail="file is larger than 20 MB")

        folder = data_dir / "uploads" / str(user["id"])
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / name
        path.write_bytes(data)
        try:
            chunks = chunks_from_file(path)
        except Exception:  # corrupt or password-protected PDF
            path.unlink(missing_ok=True)
            raise HTTPException(status_code=422, detail="couldn't read that file")
        if not chunks:
            path.unlink(missing_ok=True)
            raise HTTPException(status_code=422, detail="no text found in that file")
        try:
            with upstream():
                # drop the old version first: a shorter re-upload would otherwise leave its old tail searchable
                store.delete_source(name, owner=str(user["id"]))
                store.add(chunks, embed([c.text for c in chunks], "document"), owner=str(user["id"]))
        except HTTPException:
            path.unlink(missing_ok=True)  # don't keep a file we failed to index
            raise
        # same file name again replaces the old entry (the chunks were upserted under the same ids)
        db.run("delete from documents where owner_id = ? and filename = ?", (user["id"], name))
        db.run(
            "insert into documents (owner_id, filename, chunks, created) values (?, ?, ?, ?)",
            (user["id"], name, len(chunks), time.time()),
        )
        return {"filename": name, "chunks": len(chunks)}

    @app.delete("/documents/{filename}")
    def delete_document(filename: str, user=Depends(current_user)):
        row = db.one("select 1 from documents where owner_id = ? and filename = ?", (user["id"], filename))
        if not row:
            raise HTTPException(status_code=404, detail="no such document")
        store.delete_source(filename, owner=str(user["id"]))
        (data_dir / "uploads" / str(user["id"]) / filename).unlink(missing_ok=True)
        db.run("delete from documents where owner_id = ? and filename = ?", (user["id"], filename))
        return {"deleted": filename}

    @app.get("/documents")
    def documents(user=Depends(current_user)):
        rows = db.all("select filename, chunks from documents where owner_id = ? order by id", (user["id"],))
        return [dict(r) for r in rows]

    @app.get("/search")
    def search(q: str = Query(min_length=1, max_length=500), k: int = Query(5, ge=1, le=20), user=Depends(current_user)):
        with upstream():
            hits = store.query(embed([q], "query")[0], k=k, owner=str(user["id"]))
        return [{"source": h.source, "page": h.page, "score": round(h.score, 3), "text": h.text} for h in hits]

    @app.post("/chat")
    def chat(body: ChatIn, user=Depends(current_user)):
        if body.conversation_id is None:
            conv_id = db.run("insert into conversations (user_id, created) values (?, ?)", (user["id"], time.time()))
        else:
            conv_id = body.conversation_id
            own_conversation(conv_id, user)

        recent = db.all(
            "select role, content from messages where conversation_id = ? order by id desc limit ?",
            (conv_id, HISTORY_TURNS),
        )
        history = [{"role": r["role"], "content": strip_citations(r["content"])} for r in reversed(recent)]

        with upstream():
            result = answer(
                body.question, store, llm, embed, history=history, owner=str(user["id"]), min_score=min_score
            )
        now = time.time()
        db.run("insert into messages (conversation_id, role, content, created) values (?, 'user', ?, ?)", (conv_id, body.question, now))
        db.run("insert into messages (conversation_id, role, content, created) values (?, 'assistant', ?, ?)", (conv_id, result.text, now))
        sources = [{"source": s.source, "page": s.page, "score": round(s.score, 3)} for s in result.sources]
        answer_id = db.run(
            "insert into answers (conversation_id, question, answer, sources, grounded, created) values (?, ?, ?, ?, ?, ?)",
            (conv_id, body.question, result.text, json.dumps(sources), int(result.grounded), now),
        )
        return {
            "conversation_id": conv_id,
            "answer_id": answer_id,
            "answer": result.text,
            "grounded": result.grounded,
            "sources": sources,
        }

    @app.get("/conversations/{conv_id}")
    def conversation(conv_id: int, user=Depends(current_user)):
        own_conversation(conv_id, user)
        rows = db.all("select * from answers where conversation_id = ? order by id", (conv_id,))
        return [
            {
                "answer_id": r["id"],
                "question": r["question"],
                "answer": r["answer"],
                "sources": json.loads(r["sources"]),
                "review": {"status": r["status"], "note": r["note"]},
            }
            for r in rows
        ]

    @app.get("/reviews")
    def review_queue(user=Depends(expert_only)):
        rows = db.all("select * from answers where status = 'pending' and grounded = 1 order by id")
        return [
            {"answer_id": r["id"], "question": r["question"], "answer": r["answer"], "sources": json.loads(r["sources"])}
            for r in rows
        ]

    @app.post("/reviews/{answer_id}")
    def review(answer_id: int, body: ReviewIn, user=Depends(expert_only)):
        if not db.one("select 1 from answers where id = ?", (answer_id,)):
            raise HTTPException(status_code=404, detail="no such answer")
        db.run(
            "update answers set status = ?, reviewer_id = ?, note = ? where id = ?",
            (body.decision, user["id"], body.note, answer_id),
        )
        return {"answer_id": answer_id, "status": body.decision}

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    return app


def app_from_env() -> FastAPI:
    experts = tuple(n.strip() for n in os.environ.get("EXPERTS", "").split(",") if n.strip())
    return create_app(
        os.environ.get("RAGCHAT_DATA", "data"),
        expert_names=experts,
        min_score=default_min_score(),
    )
