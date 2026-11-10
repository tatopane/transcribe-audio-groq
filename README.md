# transcribe-audio-groq

Splits an audio file into chunks (cutting only during silences, never mid-word),
transcribes each chunk with Groq's Whisper API, and writes the combined
transcript to `transcripts/{filename}.md`. Chunk audio files are deleted as
soon as they've been successfully transcribed.

## Installation & Setup

### Prerequisites

1. **Python > 3.10**
   Make sure you have Python 3.11 or later installed. On macOS with Homebrew:
   ```bash
   brew install python
   ```

2. **FFmpeg**
   Install `ffmpeg` via Homebrew (`brew install ffmpeg`), which `pydub` requires to read and split audio formats (e.g. `.m4a`, `.mp3`):
   ```bash
   brew install ffmpeg
   ```

### Installation

1. **Clone the repository**:
   ```bash
   git clone git@github.com:tatopane/transcribe-audio-groq.git
   cd transcribe-audio-groq
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure your Groq API key** (optional, or pass via `--api-key`):
   ```bash
   cp .env.sample .env
   # Add your Groq API key to .env
   ```

## Usage

```bash
source .venv/bin/activate
python transcribe_audio.py path/to/audio.m4a --api-key gsk_...
```

The API key can also come from an environment variable or a `.env` file,
resolved in this order: `--api-key` flag → `GROQ_API_KEY` env var →
`GROQ_API_KEY` in a `.env` file (checked next to the script and in the
current directory). The script fails with an error if none is found.

```bash
export GROQ_API_KEY=gsk_...
python transcribe_audio.py path/to/audio.m4a
```

```bash
echo "GROQ_API_KEY=gsk_..." > .env
python transcribe_audio.py path/to/audio.m4a
```

The transcript is written to `transcripts/{filename}.md` (created relative to
wherever you run the script from).

### Options

| Flag | Default | Description |
|---|---|---|
| `--api-key` | `$GROQ_API_KEY` / `.env` | Groq API key |
| `--chunk-size-mb` | `10` | Target max size per chunk, in MB |
| `--bitrate-kbps` | `64` | Bitrate used when exporting chunks (used to estimate chunk duration from `--chunk-size-mb`) |
| `--min-silence-len` | `500` | Minimum silence length (ms) to be considered a valid split point |
| `--model` | `whisper-large-v3` | Groq transcription model |

## How splitting works

1. The audio is scanned for silences (`pydub.silence.detect_silence`).
2. A target max chunk duration is computed from `--chunk-size-mb` and
   `--bitrate-kbps`.
3. Split points are chosen at the midpoint of a detected silence closest to
   (but not exceeding, where possible) that target duration — so a chunk may
   run a bit over budget if no silence is available in time, but a split will
   never land mid-word.
4. If no silence can be found at all, the silence threshold is relaxed a few
   times before giving up and transcribing the whole file as a single chunk
   (with a warning printed).
