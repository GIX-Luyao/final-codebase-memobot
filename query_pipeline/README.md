# Query Pipeline

The query pipeline runs the **robot interaction flow**: it recognizes who you are from an image, then starts a voice assistant that knows your name and can look up memories on your behalf.

## What It Does

1. **User recognition** — Loads an image (e.g. a photo of you), matches the face against the onboard face database, and looks up your profile (name, etc.) from the SQL database.
2. **Realtime voice mode** — Starts the robot client in **realtime audio mode** using OpenAI’s Realtime API. Your recognized name is added to the system prompt (e.g. “You are speaking with Jason”), and the agent can use `retrieveMemory` to search video/text memories when you ask questions.

So in one shot, `main.py` does: **image → who am I? → start voice assistant as that person**.

## How to Run `main.py`

Run from the **project root** (the `memobot` directory) so imports and paths resolve correctly:

```bash
# From project root
cd /path/to/memobot
python query_pipeline/main.py
```

By default it uses `query_pipeline/image.png` as the input image.

To use another image:

```bash
python query_pipeline/main.py /path/to/your/photo.png
```

### Prerequisites

1. **Face database and onboarding**
   - At least one person must be onboarded and in the face database.
   - Run `onboarding/create_new_person.py` first to add people and their reference photos to `face_database/` and the SQL DB.

2. **Environment**
   - A `.env` in the **project root** or in `query_pipeline/` with:
     - `OPENAI_API_KEY=sk-...` (required for realtime voice).

3. **Python dependencies**
   - Install project deps from the repo root, e.g. `pip install -r requirements.txt`.
   - For realtime mode the robot client also expects: `sounddevice`, `numpy`, `websockets`.

If the face isn’t found or the DB has no match, `main.py` exits with an error and does not start the client.

## Module Overview

| File | Role |
|------|------|
| **`main.py`** | Entry point: load image → `recognize_user()` → `run_realtime_mode(user_name=...)`. |
| **`recognize_user.py`** | Face matching via `deepface/match_face.py`, then lookup in `onboarding` DB by `face_id`. Returns a person dict (e.g. `name`, `person_id`). |
| **`robo_client.py`** | Realtime voice agent (OpenAI Realtime API). Uses `user_name` in the system prompt and calls memory retrieval when the user asks about the past. |
| **`query.py`** | Query layer for video “memories” (TwelveLabs + Pinecone): embed text, search, then re-rank with relevance/importance/time decay. Used by the ingest/query pipeline; the robot client may call into it or a similar backend. |
| **`mock_memory_storage.py`** | In-memory mock memory store for testing without a real memory backend. |

## Optional: Run the robot client only (no face recognition)

To start the voice agent without running face recognition, and optionally pass a name:

```bash
python query_pipeline/robo_client.py --mode realtime --user-name "Jason"
```

Omit `--user-name` to run without injecting a user name into the system prompt.

## Optional: Run user recognition only

To only identify who is in an image (no robot client):

```bash
python query_pipeline/recognize_user.py
```

Uses `query_pipeline/image.png` by default. It prints the matched person record (e.g. `person_id`, `name`, `face_id`) and exits.
