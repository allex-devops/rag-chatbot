import fitz
import pytest
from conftest import fake_embed

from ragchat.ingest import chunks_from_file
from ragchat.store import Store


def make_pdf(path, pages):
    doc = fitz.open()
    for text in pages:
        doc.new_page().insert_textbox(fitz.Rect(50, 50, 550, 780), text, fontsize=10)
    doc.save(path)
    doc.close()


def test_pdf_chunks_remember_their_page(tmp_path):
    pdf = tmp_path / "paper.pdf"
    make_pdf(pdf, ["Alpha page about cats.", "Second page about dogs."])
    chunks = chunks_from_file(pdf)
    assert [(c.page, c.source) for c in chunks] == [(1, "paper.pdf"), (2, "paper.pdf")]
    assert "dogs" in chunks[1].text


def test_text_and_markdown_files_work(tmp_path):
    (tmp_path / "a.txt").write_text("plain text")
    (tmp_path / "b.md").write_text("# markdown")
    assert chunks_from_file(tmp_path / "a.txt")[0].text == "plain text"
    assert chunks_from_file(tmp_path / "b.md")[0].page == 1


def test_unsupported_type_is_a_clear_error(tmp_path):
    (tmp_path / "x.docx").write_bytes(b"nope")
    with pytest.raises(ValueError, match="unsupported"):
        chunks_from_file(tmp_path / "x.docx")


def test_ids_are_stable_so_reingesting_does_not_duplicate(tmp_path, store):
    f = tmp_path / "a.txt"
    f.write_text("cats purr. " * 200)
    chunks = chunks_from_file(f)
    store.add(chunks, fake_embed([c.text for c in chunks]))
    n = store.count()
    store.add(chunks, fake_embed([c.text for c in chunks]))
    assert store.count() == n


def test_query_returns_closest_first_with_scores(tmp_path, store):
    (tmp_path / "pets.txt").write_text("cats purr and sleep all day")
    (tmp_path / "cars.txt").write_text("engines burn fuel to drive wheels")
    for name in ("pets.txt", "cars.txt"):
        chunks = chunks_from_file(tmp_path / name)
        store.add(chunks, fake_embed([c.text for c in chunks]))

    hits = store.query(fake_embed(["do cats purr"], "query")[0], k=2)
    assert [h.source for h in hits] == ["pets.txt", "cars.txt"]
    assert hits[0].score > hits[1].score


def test_query_on_empty_index_is_empty_not_an_error(store):
    assert store.query(fake_embed(["anything"])[0]) == []


def test_owner_filter_keeps_users_apart(tmp_path, store):
    f = tmp_path / "notes.txt"
    f.write_text("the secret launch code is banana")
    chunks = chunks_from_file(f)
    vecs = fake_embed([c.text for c in chunks])
    store.add(chunks, vecs, owner="alice")
    q = fake_embed(["launch code"], "query")[0]
    assert len(store.query(q, owner="alice")) == 1
    assert store.query(q, owner="bob") == []
    # same file uploaded by two users must not overwrite each other
    store.add(chunks, vecs, owner="bob")
    assert store.count() == 2


def test_index_survives_reopening(tmp_path):
    path = tmp_path / "idx"
    f = tmp_path / "a.txt"
    f.write_text("persistent facts")
    chunks = chunks_from_file(f)
    Store(path).add(chunks, fake_embed([c.text for c in chunks]))
    assert Store(path).count() == 1


def test_delete_source_removes_only_that_file(tmp_path, store):
    for name, text in (("a.txt", "cats purr"), ("b.txt", "dogs bark")):
        (tmp_path / name).write_text(text)
        chunks = chunks_from_file(tmp_path / name)
        store.add(chunks, fake_embed([c.text for c in chunks]))
    store.delete_source("a.txt")
    assert store.count() == 1
    assert store.query(fake_embed(["dogs bark"], "query")[0])[0].source == "b.txt"


def test_delete_source_respects_the_owner(tmp_path, store):
    (tmp_path / "n.txt").write_text("shared name")
    chunks = chunks_from_file(tmp_path / "n.txt")
    vecs = fake_embed([c.text for c in chunks])
    store.add(chunks, vecs, owner="alice")
    store.add(chunks, vecs, owner="bob")
    store.delete_source("n.txt", owner="alice")
    assert store.count() == 1
    assert store.query(vecs[0], owner="bob")


def test_index_remembers_which_embedding_model_built_it(tmp_path):
    from ragchat.store import EmbeddingMismatch

    Store(tmp_path / "idx", embed_model="model-a")
    Store(tmp_path / "idx", embed_model="model-a")  # same model, fine
    with pytest.raises(EmbeddingMismatch, match="built with 'model-a'"):
        Store(tmp_path / "idx", embed_model="model-b")


def test_no_model_name_means_no_check(tmp_path):
    Store(tmp_path / "idx", embed_model="model-a")
    Store(tmp_path / "idx")  # the tests and quick scripts that don't care


def test_wrong_vector_size_is_a_clear_error_not_a_crash_inside_chroma(tmp_path):
    import numpy as np

    from ragchat.store import EmbeddingMismatch

    store = Store(tmp_path / "idx")
    (tmp_path / "a.txt").write_text("cats purr")
    chunks = chunks_from_file(tmp_path / "a.txt")
    store.add(chunks, fake_embed([c.text for c in chunks]))
    with pytest.raises(EmbeddingMismatch, match="256-dimensional vectors but got 8"):
        store.query(np.ones(8, dtype=np.float32))
    with pytest.raises(EmbeddingMismatch, match="got 8"):
        store.add(chunks, np.ones((1, 8), dtype=np.float32))


def test_a_large_add_is_written_in_slices_and_nothing_is_lost(tmp_path, monkeypatch):
    from ragchat.ingest import Chunk

    monkeypatch.setattr("ragchat.store.WRITE_BATCH", 7)
    store = Store(tmp_path / "idx")
    # the fake embedder only sees letters, so give every chunk its own made-up word
    word = lambda i: "zz" + "".join(chr(97 + int(d)) for d in str(i))
    chunks = [Chunk(f"f:{i}", f"chunk about {word(i)}", "f.txt", 1) for i in range(25)]
    store.add(chunks, fake_embed([c.text for c in chunks]))
    assert store.count() == 25
    # every chunk still lines up with its own vector, including ones on and after slice boundaries
    for i in (0, 6, 7, 13, 14, 24):
        top = store.query(fake_embed([word(i)], "query")[0], k=1)[0]
        assert top.text == f"chunk about {word(i)}", i
