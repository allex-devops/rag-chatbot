import argparse
from pathlib import Path

from .ingest import chunks_from_file
from .rag import ConfigError, answer, default_min_score, embedder, llm_from_env
from .store import Store


def main(argv: list[str] | None = None) -> None:
    cutoff = default_min_score()
    ap = argparse.ArgumentParser(prog="ragchat")
    ap.add_argument("--data", default="data", help="where the vector index lives")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ing = sub.add_parser("ingest", help="add PDFs, .txt or .md files to the index")
    ing.add_argument("paths", nargs="+", type=Path)
    ask = sub.add_parser("ask", help="ask one question")
    ask.add_argument("question")
    ask.add_argument("--min-score", type=float, default=cutoff, help="similarity below which it answers 'I don't know'")
    sub.add_parser("chat", help="ask questions interactively")
    args = ap.parse_args(argv)

    try:
        llm = llm_from_env()
    except ConfigError as e:
        ap.error(str(e))
    embed = embedder(llm)
    store = Store(Path(args.data), embed_model=llm.embed_model)

    if args.cmd == "ingest":
        for path in args.paths:
            chunks = chunks_from_file(path)
            store.add(chunks, embed([c.text for c in chunks], "document"))
            print(f"{path.name}: {len(chunks)} chunks")
        print(f"index now holds {store.count()} chunks")
        return

    def ask_once(q: str, min_score: float = cutoff) -> None:
        a = answer(q, store, llm, embed, min_score=min_score)
        print(a.text)
        for i, s in enumerate(a.sources, 1):
            print(f"  [{i}] {s.source} p.{s.page} (score {s.score:.2f})")

    if args.cmd == "ask":
        ask_once(args.question, args.min_score)
    else:
        while True:
            try:
                q = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q:
                ask_once(q)


if __name__ == "__main__":
    main()
