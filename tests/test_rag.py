import pytest
from conftest import fake_embed

from ragchat.ingest import chunks_from_file
from ragchat.rag import (
    DEFAULT_MIN_SCORE,
    NO_ANSWER,
    REQUIRED_ENV,
    ConfigError,
    answer,
    default_min_score,
    llm_from_env,
)


def test_min_score_can_be_set_from_the_environment(monkeypatch):
    monkeypatch.delenv("RAGCHAT_MIN_SCORE", raising=False)
    assert default_min_score() == DEFAULT_MIN_SCORE
    monkeypatch.setenv("RAGCHAT_MIN_SCORE", "0.45")
    assert default_min_score() == 0.45


def test_the_provider_must_be_configured_and_nothing_is_assumed(monkeypatch):
    for name in REQUIRED_ENV:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ConfigError) as err:
        llm_from_env()
    assert all(name in str(err.value) for name in REQUIRED_ENV)

    monkeypatch.setenv("LLM_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_MODEL", "chat-model")
    with pytest.raises(ConfigError, match="EMBED_MODEL"):  # one missing value is still an error
        llm_from_env()

    monkeypatch.setenv("EMBED_MODEL", "embed-model")
    llm = llm_from_env()
    assert (llm.base_url, llm.api_key, llm.model, llm.embed_model) == (
        "https://api.example.com/v1",
        "sk-test",
        "chat-model",
        "embed-model",
    )


def index(tmp_path, store, **files):
    for name, text in files.items():
        p = tmp_path / name
        p.write_text(text)
        chunks = chunks_from_file(p)
        store.add(chunks, fake_embed([c.text for c in chunks]))


def test_answers_from_the_relevant_document_and_cites_it(tmp_path, store, llm):
    index(tmp_path, store, **{"pets.txt": "cats purr when they are happy", "cars.txt": "engines burn fuel"})
    a = answer("why do cats purr", store, llm, fake_embed, min_score=0.2)
    assert a.grounded and a.text == "The answer is 42 [1]."
    assert a.sources[0].source == "pets.txt"
    prompt = llm.calls[0][-1]["content"]
    assert "[1] (pets.txt, p.1) cats purr" in prompt
    assert prompt.endswith("Question: why do cats purr")


def test_unrelated_question_gets_dont_know_without_calling_the_model(tmp_path, store, llm):
    index(tmp_path, store, **{"pets.txt": "cats purr when they are happy"})
    a = answer("what is the capital of france", store, llm, fake_embed, min_score=0.3)
    assert a.text == NO_ANSWER and not a.grounded
    assert llm.calls == []


def test_empty_index_is_dont_know(store, llm):
    a = answer("anything at all", store, llm, fake_embed)
    assert a.text == NO_ANSWER and llm.calls == []


def test_only_passages_above_the_threshold_reach_the_prompt(tmp_path, store, llm):
    index(tmp_path, store, **{"pets.txt": "cats purr when happy", "cars.txt": "engines burn fuel"})
    answer("do cats purr", store, llm, fake_embed, min_score=0.3)
    prompt = llm.calls[0][-1]["content"]
    assert "cats purr" in prompt and "engines" not in prompt


def test_history_goes_between_system_prompt_and_question(tmp_path, store, llm):
    index(tmp_path, store, **{"pets.txt": "cats purr when happy"})
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    answer("do cats purr", store, llm, fake_embed, min_score=0.2, history=history)
    roles = [m["role"] for m in llm.calls[0]]
    assert roles == ["system", "user", "assistant", "user"]


def test_reply_is_stripped_and_model_is_asked_deterministically(tmp_path, store):
    class Loud:
        def chat(self, messages, **kw):
            assert kw["temperature"] == 0
            return {"content": "  spaced out \n"}

    index(tmp_path, store, **{"pets.txt": "cats purr when happy"})
    assert answer("do cats purr", store, Loud(), fake_embed, min_score=0.2).text == "spaced out"


def test_trace_reports_each_stage_with_timings(tmp_path, store, llm):
    index(tmp_path, store, **{"pets.txt": "cats purr when happy"})
    events = []
    answer("do cats purr", store, llm, fake_embed, min_score=0.2, trace=lambda e, f: events.append((e, f)))
    assert [e for e, _ in events] == ["retrieve", "generate"]
    retrieve, generate = events[0][1], events[1][1]
    assert retrieve["usable"] == 1 and retrieve["top_score"] > 0.2
    assert {"embed_ms", "search_ms"} <= retrieve.keys() and generate["prompt_chars"] > 0


def test_trace_says_why_it_declined(tmp_path, store, llm):
    events = []
    answer("anything", store, llm, fake_embed, trace=lambda e, f: events.append((e, f)))
    assert events[-1] == ("decline", {"reason": "index is empty", "min_score": DEFAULT_MIN_SCORE})

    index(tmp_path, store, **{"pets.txt": "cats purr when happy"})
    events.clear()
    answer("quantum chromodynamics lattice", store, llm, fake_embed, min_score=0.9, trace=lambda e, f: events.append((e, f)))
    assert events[-1][1]["reason"] == "nothing above min_score"


def test_empty_model_reply_is_not_passed_off_as_a_grounded_answer(tmp_path, store):
    from ragchat.rag import EMPTY_REPLY

    class Silent:
        def chat(self, messages, **kw):
            return {"content": None}

    index(tmp_path, store, **{"pets.txt": "cats purr when happy"})
    events = []
    a = answer("do cats purr", store, Silent(), fake_embed, min_score=0.2, trace=lambda e, f: events.append(e))
    assert (a.text, a.grounded) == (EMPTY_REPLY, False)
    assert a.sources and events[-1] == "empty_reply"  # the passages that were found are still reported
