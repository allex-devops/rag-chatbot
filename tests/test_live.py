"""Calls a real chat model and embedding model, so it uses a few API tokens.

Needs LLM_BASE_URL, LLM_API_KEY, LLM_MODEL and EMBED_MODEL (see .env.example):

    uv run --env-file .env pytest -m live -s
"""
import fitz
import pytest
from fastapi.testclient import TestClient

from ragchat.ingest import chunks_from_file
from ragchat.rag import ConfigError, answer, embedder, llm_from_env
from ragchat.store import Store
from ragchat.webapp import create_app

pytestmark = pytest.mark.live

POLICY = "The office is closed on public holidays. Staff get 25 days of paid leave a year."


@pytest.fixture(scope="module")
def llm():
    try:
        return llm_from_env()
    except ConfigError as e:
        pytest.skip(str(e))


def make_pdf(path, text):
    doc = fitz.open()
    doc.new_page().insert_textbox(fitz.Rect(50, 50, 550, 780), text, fontsize=12)
    doc.save(path)
    doc.close()


def test_real_model_answers_from_a_document_and_scores_off_topic_lower(tmp_path, llm):
    make_pdf(tmp_path / "policy.pdf", POLICY)
    embed = embedder(llm)
    store = Store(tmp_path / "idx", embed_model=llm.embed_model)
    chunks = chunks_from_file(tmp_path / "policy.pdf")
    store.add(chunks, embed([c.text for c in chunks], "document"))

    on_topic = answer("How many days of paid leave do staff get?", store, llm, embed, min_score=0.0)
    off_topic = answer("What is the capital of Mongolia?", store, llm, embed, min_score=0.0)
    print(f"\non-topic: {on_topic.text!r} (top score {on_topic.sources[0].score:.2f})")
    print(f"off-topic top score: {off_topic.sources[0].score:.2f}")
    assert "25" in on_topic.text
    assert on_topic.sources[0].score > off_topic.sources[0].score


def test_web_app_end_to_end_with_real_models(tmp_path, llm):
    """Sign up, upload a document, ask about it, then a follow-up: everything the browser would do."""
    pdf = tmp_path / "policy.pdf"
    make_pdf(pdf, POLICY)
    app = create_app(tmp_path / "data", llm=llm, embed_model=llm.embed_model)
    me = TestClient(app)
    assert me.post("/signup", json={"name": "livetest", "password": "correct horse battery"}).status_code == 201
    up = me.post("/documents", files={"file": (pdf.name, pdf.read_bytes(), "application/pdf")})
    assert up.status_code == 201 and up.json()["chunks"] > 0

    r = me.post("/chat", json={"question": "How many days of paid leave do staff get?"}).json()
    print(f"\nQ: How many days of paid leave do staff get?\nA: {r['answer'][:300]!r}\nsources: {r['sources'][:2]}")
    assert r["grounded"] and "25" in r["answer"]
    assert r["sources"][0]["source"] == pdf.name

    follow_up = me.post("/chat", json={"question": "Can you say that more briefly?", "conversation_id": r["conversation_id"]}).json()
    assert follow_up["answer"].strip()
