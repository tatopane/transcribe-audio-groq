#!/usr/bin/env python3
"""
Split an audio file into ~10MB chunks (cut only at silences), transcribe
each chunk with Groq's Whisper API, and write the combined transcript to
transcripts/{filename}.md.

Usage:
    python transcribe_audio.py path/to/audio.m4a --api-key gsk_...
    GROQ_API_KEY=gsk_... python transcribe_audio.py path/to/audio.m4a
    echo "GROQ_API_KEY=gsk_..." > .env && python transcribe_audio.py path/to/audio.m4a

Requires: pydub, groq, python-dotenv, and ffmpeg installed on the system.
    pip install pydub groq python-dotenv

The Groq API key is resolved in this order: --api-key flag, GROQ_API_KEY
environment variable, GROQ_API_KEY in a .env file (next to this script or
in the current directory).
"""
import argparse
import contextlib
import itertools
import os
import sys
import threading
import time

from dotenv import load_dotenv
from pydub import AudioSegment
from pydub.silence import detect_silence


@contextlib.contextmanager
def spinner(message):
    """Print a simple animated progress indicator while a block runs."""
    stop = threading.Event()

    def spin():
        start = time.time()
        for frame in itertools.cycle("|/-\\"):
            if stop.is_set():
                break
            elapsed = time.time() - start
            print(f"\r{message} {frame} ({elapsed:.0f}s)", end="", flush=True)
            time.sleep(0.2)

    thread = threading.Thread(target=spin, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()
        print("\r" + " " * (len(message) + 12) + "\r", end="", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_path", help="Path to the input audio file")
    parser.add_argument(
        "--api-key",
        default=None,
        help="Groq API key (defaults to GROQ_API_KEY env var or .env file)",
    )
    parser.add_argument(
        "--chunk-size-mb",
        type=float,
        default=10.0,
        help="Target max size per chunk in MB (default: 10)",
    )
    parser.add_argument(
        "--bitrate-kbps",
        type=int,
        default=64,
        help="Bitrate used when exporting chunks, in kbps (default: 64)",
    )
    parser.add_argument(
        "--min-silence-len",
        type=int,
        default=500,
        help="Minimum silence length in ms to be considered a split point (default: 500)",
    )
    parser.add_argument(
        "--model",
        default="whisper-large-v3",
        help="Groq transcription model (default: whisper-large-v3)",
    )
    return parser.parse_args()


def find_split_points(audio, max_chunk_ms, min_silence_len):
    """
    Return a sorted list of millisecond positions at which it is safe to
    split `audio` (i.e. positions that fall inside a detected silence),
    chosen so that no resulting chunk exceeds max_chunk_ms unless no
    silence is available before the limit is reached.
    """
    silence_thresh = audio.dBFS - 16
    silences = []
    attempts = 0
    # If nothing is found, progressively relax the threshold/min length
    # rather than falling back to a mid-word cut.
    while not silences and attempts < 6:
        with spinner("Detecting silences and computing split points..."):
            silences = detect_silence(
                audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh
            )
        if not silences:
            silence_thresh += 4  # be less strict about what counts as silence
            min_silence_len = max(100, min_silence_len // 2)
            attempts += 1

    if not silences:
        print(
            "WARNING: no silence detected in the audio; unable to guarantee "
            "clean split points. Transcribing as a single chunk instead.",
            file=sys.stderr,
        )
        return []

    midpoints = [(start + end) // 2 for start, end in silences]

    splits = []
    last_split = 0
    last_candidate = None
    for mid in midpoints:
        if mid - last_split > max_chunk_ms:
            if last_candidate is not None:
                splits.append(last_candidate)
                last_split = last_candidate
                last_candidate = None
                # Re-evaluate this midpoint against the new last_split.
                if mid - last_split > max_chunk_ms:
                    # Still too far — no silence in between, forced to
                    # split here even though the chunk will run over
                    # budget; this still guarantees a silence cut.
                    splits.append(mid)
                    last_split = mid
                else:
                    last_candidate = mid
            else:
                # No silence found yet within budget at all; forced split
                # at this (still valid) silence point.
                splits.append(mid)
                last_split = mid
        else:
            last_candidate = mid

    return splits


def split_audio(audio, max_chunk_ms, min_silence_len):
    splits = find_split_points(audio, max_chunk_ms, min_silence_len)
    boundaries = [0] + splits + [len(audio)]
    chunks = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end > start:
            chunks.append(audio[start:end])
    return chunks


def transcribe_chunk(client, path, model):
    with open(path, "rb") as f:
        transcription = client.audio.transcriptions.create(
            file=(os.path.basename(path), f.read()),
            model=model,
            temperature=0,
            response_format="verbose_json",
        )
    return transcription.text


def resolve_api_key(cli_api_key):
    if cli_api_key:
        return cli_api_key

    if os.environ.get("GROQ_API_KEY"):
        return os.environ["GROQ_API_KEY"]

    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
    load_dotenv()  # also check .env in the current working directory
    return os.environ.get("GROQ_API_KEY")


def main():
    args = parse_args()

    api_key = resolve_api_key(args.api_key)
    if not api_key:
        print(
            "Error: no Groq API key provided. Pass --api-key, set GROQ_API_KEY, "
            "or add GROQ_API_KEY to a .env file.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not os.path.isfile(args.audio_path):
        print(f"Error: file not found: {args.audio_path}", file=sys.stderr)
        sys.exit(1)

    from groq import Groq

    client = Groq(api_key=api_key)

    print(f"Loading audio: {args.audio_path}")
    audio = AudioSegment.from_file(args.audio_path)

    # Always export chunks as mp3: Groq accepts it, it's small at low
    # bitrates, and it avoids ffmpeg muxer-name mismatches some input
    # formats have (e.g. "m4a" isn't a valid ffmpeg muxer name — "ipod" is).
    export_format = "mp3"

    bytes_per_ms = (args.bitrate_kbps * 1000 / 8) / 1000
    max_chunk_ms = int((args.chunk_size_mb * 1024 * 1024) / bytes_per_ms)

    chunks = split_audio(audio, max_chunk_ms, args.min_silence_len)
    print(f"Split into {len(chunks)} chunk(s).")

    base_name = os.path.splitext(os.path.basename(args.audio_path))[0]
    os.makedirs("transcripts", exist_ok=True)
    output_path = os.path.join("transcripts", f"{base_name}.md")

    chunk_paths = []
    transcripts = []
    try:
        for i, chunk in enumerate(chunks, start=1):
            chunk_path = f"{base_name}.chunk{i:03d}.{export_format}"
            print(f"Exporting chunk {i}/{len(chunks)}: {chunk_path}")
            chunk.export(
                chunk_path, format=export_format, bitrate=f"{args.bitrate_kbps}k"
            )
            chunk_paths.append(chunk_path)

            print(f"Transcribing chunk {i}/{len(chunks)}...")
            text = transcribe_chunk(client, chunk_path, args.model)
            transcripts.append(text.strip())

            os.remove(chunk_path)
            chunk_paths.remove(chunk_path)
    finally:
        # Clean up any chunk files left behind if transcription failed partway.
        for path in chunk_paths:
            if os.path.exists(path):
                os.remove(path)

    with open(output_path, "w") as f:
        f.write(f"# Transcript: {base_name}\n\n")
        f.write("\n\n".join(transcripts))
        f.write("\n")

    print(f"Transcript written to {output_path}")


if __name__ == "__main__":
    main()
