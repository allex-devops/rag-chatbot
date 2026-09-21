# RAG Chatbot

Chat with your own documents. Upload PDFs, text or Markdown files, ask questions, and get answers that cite their sources, or "I don't know" when the documents don't cover it.

It runs as a command-line tool or as a multi-user web app with accounts, semantic search, conversation memory and an expert review queue. Documents are chunked, embedded and stored in Chroma; each question retrieves the closest passages and the model answers only from those.

## Requirements

- Python 3.12 and [`uv`](https://docs.astral.sh/uv/)
- An API key for OpenAI or Google Gemini, used for both chat and embeddings. Claude isn't supported because Anthropic has no embeddings API
- Some PDFs, `.txt` or `.md` files to ask about

Your document text is sent to the provider you choose.

## Setup

```bash
uv sync
cp .env.example .env    # uncomment one provider block and paste your key
```

## Run

Command line:

```bash
uv run --env-file .env python -m ragchat ingest path/to/*.pdf
uv run --env-file .env python -m ragchat ask "What does the report say about pricing?"
uv run --env-file .env python -m ragchat chat      # interactive, Ctrl-D to quit
```

Web app:

```bash
uv run --env-file .env uvicorn ragchat.webapp:app_from_env --factory --port 8001
```

Open http://localhost:8001, sign up, upload a file and ask questions. Anyone who signs up with a name listed in `EXPERTS` can approve or reject answers in the review queue.

## Configuration

Set these in `.env`. The first four are required and have no defaults: the app won't start until you've chosen a provider and given it your own key.

| Variable | Purpose |
|---|---|
| `LLM_BASE_URL` | the provider's OpenAI-compatible endpoint |
| `LLM_API_KEY` | your provider key |
| `LLM_MODEL` | chat model name |
| `EMBED_MODEL` | embedding model name |
| `RAGCHAT_MIN_SCORE` | similarity below which it answers "I don't know" (default `0.7`) |
| `EXPERTS` | comma-separated names allowed to review answers |
| `RAGCHAT_DATA` | where the index, uploads and database live (default `./data`) |

The index remembers which embedding model built it and refuses to open with a different one, so after changing `EMBED_MODEL` use a fresh data folder (or delete the old index) and ingest your files again.

### The "I don't know" cutoff

When no passage is similar enough to the question, it answers "I don't know" without calling the chat model. The default cutoff of `0.7` was measured on one small open embedding model and sat in a thin gap between on-topic and unrelated questions. Similarity scores differ between embedding models, so if it declines questions it should answer, or answers unrelated ones, adjust `RAGCHAT_MIN_SCORE` (or pass `--min-score` to `ask`). `ask` prints each source's score to help.

## Web API

| Endpoint | What it does |
|---|---|
| `POST /signup`, `/login`, `/logout`, `GET /me` | accounts, cookie session |
| `POST /documents`, `GET /documents` | upload (.pdf .txt .md, 20 MB max) and list your files |
| `DELETE /documents/{filename}` | remove one of your files |
| `GET /search?q=` | semantic search over your files only |
| `POST /chat` | answer with sources; send `conversation_id` to keep the thread |
| `GET /conversations/{id}` | your answers and their review status |
| `GET /reviews`, `POST /reviews/{id}` | experts only |

## Tests

```bash
uv run pytest                                  # offline, no key needed
uv run --env-file .env pytest -m live -s       # real models; uses API tokens
```

## Known limits

The session cookie has no `Secure` flag so it works on localhost; put the app behind HTTPS before exposing it. There is no password reset, and no rate limiting apart from the login lockout.
