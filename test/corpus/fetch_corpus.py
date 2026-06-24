"""Fetch + decode the real-audio validation corpus from Hugging Face (pure Python, no ffmpeg).

    speech -> openslr/librispeech_asr, clean/test (test-clean: ~40 speakers, CC BY 4.0)
    noise  -> DynamicSuperb/EnvironmentalSoundClassification_ESC50-ExteriorAndUrbanNoises
              (ESC-50 environmental sounds, CC BY-NC 3.0)

OpenSLR (the canonical LibriSpeech/MUSAN host) is frequently unreachable; these HF mirrors are on a
fast CDN. We redistribute NO audio — clips are fetched locally and gitignored. The committed
golden vector (built by reference_probs.py) records only derived probabilities + a per-clip sha256.

Output: data/*.pcm (16 kHz mono s16le) + data/index.tsv (name <tab> kind <tab> source).
Idempotent: cached parquet is reused. Requires: numpy, pyarrow, soundfile.

    python3 -m venv test/corpus/.venv
    test/corpus/.venv/bin/pip install onnxruntime numpy pyarrow soundfile
    test/corpus/.venv/bin/python test/corpus/fetch_corpus.py
"""
import io
import os
import shutil
import ssl
import urllib.request

import numpy as np

try:  # framework Pythons often lack a system CA bundle for urllib
    import certifi

    _SSL = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL = ssl.create_default_context()
import pyarrow.parquet as pq
import soundfile as sf

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, ".cache")
DATA = os.path.join(ROOT, "data")

TARGET_SR = 16000
N_SPEECH = int(os.environ.get("N_SPEECH", 20))
N_NOISE = int(os.environ.get("N_NOISE", 6))
CLIP_SECS = float(os.environ.get("CLIP_SECS", 10))

SPEECH_PARQUET = (
    "https://huggingface.co/datasets/openslr/librispeech_asr/"
    "resolve/main/clean/test/0000.parquet"
)
NOISE_PARQUET = (
    "https://huggingface.co/datasets/DynamicSuperb/"
    "EnvironmentalSoundClassification_ESC50-ExteriorAndUrbanNoises/"
    "resolve/main/data/test-00000-of-00001-d84d6443db8ea0ea.parquet"
)


def fetch(url, name):
    path = os.path.join(CACHE, name)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        print(f"cache hit: {name}")
        return path
    print(f"downloading {name} ...")
    with urllib.request.urlopen(url, context=_SSL) as r, open(path, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    print(f"  {os.path.getsize(path)} bytes")
    return path


def to_pcm16(audio_bytes):
    """Decode any soundfile-readable bytes -> 16 kHz mono s16le numpy array."""
    x, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
    x = x.mean(axis=1)  # downmix to mono
    if sr != TARGET_SR:  # linear resample (gate is NIF==reference on identical PCM, so this is fine)
        n = int(round(len(x) * TARGET_SR / sr))
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
    x = x[: int(CLIP_SECS * TARGET_SR)]
    return np.clip(np.round(x * 32768.0), -32768, 32767).astype("<i2")


def write_clip(idx_fh, name, kind, source, pcm16):
    pcm16.tofile(os.path.join(DATA, f"{name}.pcm"))
    idx_fh.write(f"{name}\t{kind}\t{source}\n")
    print(f"  {name:12s} {kind:6s} {len(pcm16) / TARGET_SR:5.1f}s  <- {source}")


def main():
    os.makedirs(CACHE, exist_ok=True)
    os.makedirs(DATA, exist_ok=True)
    speech_pq = fetch(SPEECH_PARQUET, "librispeech_test_clean.parquet")
    noise_pq = fetch(NOISE_PARQUET, "esc50_urban.parquet")

    with open(os.path.join(DATA, "index.tsv"), "w") as idx:
        # speech: spread across distinct speakers for diversity
        rows = pq.read_table(speech_pq, columns=["audio", "speaker_id", "id"]).to_pylist()
        by_speaker = {}
        for r in rows:
            by_speaker.setdefault(r.get("speaker_id"), []).append(r)
        picked, queues = [], list(by_speaker.values())
        i = 0
        while len(picked) < min(N_SPEECH, len(rows)):
            q = queues[i % len(queues)]
            if q:
                picked.append(q.pop(0))
            i += 1
            if i > 10000:
                break
        for n, r in enumerate(picked, 1):
            src = f"librispeech:{r.get('id') or r['audio']['path']} (spk {r.get('speaker_id')})"
            write_clip(idx, f"speech_{n:02d}", "speech", src, to_pcm16(r["audio"]["bytes"]))

        # noise: ESC-50 exterior/urban sounds
        nrows = pq.read_table(noise_pq).to_pylist()[:N_NOISE]
        for n, r in enumerate(nrows, 1):
            src = f"esc50:{r.get('file')} ({r.get('label')})"
            write_clip(idx, f"noise_{n:02d}", "noise", src, to_pcm16(r["audio"]["bytes"]))

    print(f"done -> {DATA}")
    print("next: build the golden vector with reference_probs.py")


if __name__ == "__main__":
    main()
