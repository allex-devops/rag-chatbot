"""Calls a real chat model and embedding model, so it uses a few API tokens.

Needs LLM_BASE_URL, LLM_API_KEY, LLM_MODEL and EMBED_MODEL (see .env.example), plus the shared papers
and question set:

    uv run --env-file .env pytest -m live -s
"""
import json
import statistics

import pytest
from shared.corpus import pdf_path
from shared.paths import DATA_DIR

from ragchat.ingest import chunks_from_file
from ragchat.rag import ConfigError, answer, embedder, llm_from_env
from ragchat.store import Store

pytestmark = pytest.mark.live
QUESTIONS = DATA_DIR / "evalset" / "questions.jsonl"
OFF_TOPIC = [
    "What is the best recipe for banana bread?",
    "Who won the 2018 football world cup?",
    "How do I change a flat tyre on a bicycle?",
    "What is the capital of Mongolia?",
]


@pytest.fixture(scope="module")
def llm():
    try:
        return llm_from_env()
    except ConfigError as e:
        pytest.skip(str(e))


def test_real_models_find_the_right_paper_and_score_unrelated_questions_lower(tmp_path, llm):
    if not QUESTIONS.exists():
        pytest.skip("question set not generated yet")
    rows = [json.loads(line) for line in QUESTIONS.read_text().splitlines()][:12]
    if len(rows) < 12:
        pytest.skip("fewer than 12 questions so far")

    embed = embedder(llm)
    store = Store(tmp_path / "idx", embed_model=llm.embed_model)
    for paper in {r["paper_id"] for r in rows}:
        chunks = chunks_from_file(pdf_path(paper))
        store.add(chunks, embed([c.text for c in chunks], "document"))  # cached, so this is quick

    right, on_topic, answers = 0, [], []
    for r in rows:
        a = answer(r["question"], store, llm, embed, min_score=0.0)
        top = a.sources[0]
        right += top.source == pdf_path(r["paper_id"]).name
        on_topic.append(top.score)
        answers.append(a.text)

    off_topic = [answer(q, store, llm, embed, min_score=0.0).sources[0].score for q in OFF_TOPIC]
    print(f"\nright paper ranked first: {right}/{len(rows)}")
    print(f"top score on-topic:  min {min(on_topic):.2f}  median {statistics.median(on_topic):.2f}  max {max(on_topic):.2f}")
    print(f"top score off-topic: min {min(off_topic):.2f}  median {statistics.median(off_topic):.2f}  max {max(off_topic):.2f}")
    print("sample answer:", answers[0][:200].replace("\n", " "))
    (DATA_DIR / "evalset" / "live_scores.json").write_text(json.dumps({"on_topic": on_topic, "off_topic": off_topic}))

    assert right / len(rows) >= 0.6
    assert all(a.strip() for a in answers)


def test_web_app_end_to_end_with_real_models(tmp_path, llm):
    """Sign up, upload a real paper, ask about it: everything the browser would do, minus the browser."""
    from fastapi.testclient import TestClient

    from ragchat.webapp import create_app

    if not QUESTIONS.exists():
        pytest.skip("question set not generated yet")
    row = json.loads(QUESTIONS.read_text().splitlines()[0])
    pdf = pdf_path(row["paper_id"])

    app = create_app(tmp_path / "data", llm=llm, embed_model=llm.embed_model)
    me = TestClient(app)
    assert me.post("/signup", json={"name": "livetest", "password": "correct horse battery"}).status_code == 201
    up = me.post("/documents", files={"file": (pdf.name, pdf.read_bytes(), "application/pdf")})
    assert up.status_code == 201 and up.json()["chunks"] > 5

    r = me.post("/chat", json={"question": row["question"]}).json()
    print(f"\nQ: {row['question']}\nA: {r['answer'][:300]!r}\nsources: {r['sources'][:2]}")
    assert r["grounded"] and r["answer"].strip()
    assert r["sources"][0]["source"] == pdf.name

    follow_up = me.post("/chat", json={"question": "Can you say that more briefly?", "conversation_id": r["conversation_id"]}).json()
    assert follow_up["answer"].strip()
