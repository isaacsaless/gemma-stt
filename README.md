# gemma-stt

Fast, private speech-to-text with **semantic search** over transcripts. Runs 100% on
your machine — the audio never leaves the computer. Upload an audio file, watch the
transcript appear in real time, then search it by meaning ("the part where she talks
about where she grew up") and click a result to jump the player to that moment.

## How it works

- **Transcription** — [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
  `large-v3-turbo` (int8 via CTranslate2). Any language, including Portuguese;
  detects the language automatically. Produces capitalization, punctuation and
  word-level timestamps (used by the player and the search index).
- **Semantic search** — the transcript is split into chunks at sentence boundaries
  (~10–56 words) and embedded with the text encoder of
  [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2)
  (270M params, 768-d embeddings, multilingual). Chunks are embedded with
  `title: none | text: …`, queries with `task: search result | query: …`, ranked by
  cosine similarity. Descriptive queries like *"the part where the girl talks about
  who she grew up with"* are normalized first (scaffolding words such as "the part
  where", "the guy says", "at the moment in which" are stripped before embedding).
- **Web app** — plain FastAPI + vanilla HTML/CSS/JS. Jobs run in the background and
  report progress over Server-Sent Events; results are searchable in memory.

## Requirements

- Python **3.13**
- Any Windows or Linux machine. A CUDA GPU speeds things up a lot (the RTX 3050 6 GB
  this project was developed on reaches ~2× realtime), but CPU-only works fine —
  expect roughly 1–3× realtime on the whisper step depending on vCPUs. Good fit for
  a 4–8 vCPU VPS; see [Running on a VPS](#running-on-a-vps).
- ffmpeg audio decoding is bundled via `imageio-ffmpeg`, no system install needed.

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\pip install -r requirements.txt
# Linux/macOS:
.venv/bin/pip install -r requirements.txt
```

With an NVIDIA GPU you can also install the CUDA build of torch for faster
embeddings (the project's `scripts/install.sh` does exactly this):

```bash
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Then download the search-embedding model weights (~1.9 GB into `models/`):

```bash
bash scripts/download_models.sh
```

The whisper `large-v3-turbo` weights (~1.6 GB) are fetched automatically by
faster-whisper to the Hugging Face cache on first use.

> On CPU-only machines, drop the line with the index-url (the plain `torch` wheel
> from `requirements.txt` works fine).

## Run

```bash
# Windows:
run.bat
# anywhere:
.venv/Scripts/python server.py        # or .venv/bin/python server.py
```

Open **http://localhost:8425**, drop an audio file, and search the transcript.

## Running on a VPS

The server has **one shared password** set by `APP_PASSWORD` (env var or a `.env`
file next to `server.py`). It refuses to start without it. Login issues an
HMAC-signed HttpOnly session cookie; attempts are throttled. There is no
authentication bypass route — upper endpoints (`/api/*`, uploads, audio, exports)
are protected like the UI.

On the server, bind only to loopback (`HOST=127.0.0.1`) and front it with an
authenticating proxy or a Cloudflare Tunnel. `scripts/gemma-stt.service` ships a
systemd unit that does exactly this:

```bash
./scripts/download_models.sh                                 # search weights
printf 'APP_PASSWORD=sua-senha-forte\n' > .env               # shared login password
sudo cp scripts/gemma-stt.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now gemma-stt
```

## HTTP API

| Endpoint | Purpose |
|---|---|
| `POST /api/upload` | multipart `file` → starts a transcription job |
| `POST /api/transcribe` | `{"path": "..."}` → same, from a local server path |
| `GET /api/job/{id}` | status + result JSON |
| `GET /api/job/{id}/events` | live progress (SSE) |
| `GET /api/job/{id}/audio` | original audio (player streaming) |
| `POST /api/search` | `{"job_id", "q"}` → ranked hits with timestamps |
| `POST /api/export/{id}` | `{"fmt": "txt"\|"vtt"}` → download the transcript |
| `GET /api/status` | health + active device (cuda/cpu) |

## Project structure

```
server.py            FastAPI app: upload → job → SSE progress → search
stt/whisper_stt.py   faster-whisper wrapper (audio → text + word timestamps)
stt/audio.py         ffmpeg decode: any format → float32 mono 16 kHz
stt/text_embed.py    EmbeddingGemma 2 text encoder (768-d embeddings)
stt/search.py        sentence chunking, query normalization, cosine search
stt/transcribe.py    engine: lazy model loading + device selection
static/              vanilla JS UI (upload, player, search, export)
scripts/             download_models.sh · install.sh
```

## Limitations

- Search index lives in process memory; jobs are lost on restart. Transcripts live
  entirely client-side + server memory as well.
- Large-v3-turbo needs ~1.6 GB disk (downloaded on first use) and ~2 GB VRAM in
  CUDA mode (int8 CPU mode uses ~2 GB RAM instead).
- Word timestamps come from whisper; very noisy audio can drift a little.

## Credits

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — MIT
- [embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2) — Apache-2.0
  (used only as the semantic search text encoder)
- faster-whisper is trained on Whisper CT data (OpenAI) — see their license.
