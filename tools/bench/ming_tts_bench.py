#!/usr/bin/env python3
"""Latency benchmark for Ming TTS on an audiocpp_server (offline and SSE streaming).

Prints wall time, time to first audio (TTFA), audio duration and real-time factor per
text, and saves every result as WAV so runs can be compared for quality.
"""

import argparse
import base64
import json
import os
import statistics
import time
import urllib.request
import wave

TEXTS = {
    "short": "Hallo, wie geht es dir heute?",
    "medium": "Das Wetter in Berlin ist heute sonnig bei zwanzig Grad.",
    "long": (
        "Das Wetter in Berlin ist heute sonnig bei zwanzig Grad. Am Nachmittag ziehen vereinzelt "
        "Wolken auf, aber es bleibt trocken. Morgen wird es etwas kühler, mit einem leichten Wind "
        "aus Westen."
    ),
}


def post(url, body, accept=None):
    headers = {"Content-Type": "application/json"}
    if accept:
        headers["Accept"] = accept
    return urllib.request.urlopen(
        urllib.request.Request(url, json.dumps(body).encode(), headers), timeout=600)


def write_wav(path, pcm, rate):
    with wave.open(path, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(pcm)


def offline(url, model, text, options, rate=None):
    start = time.perf_counter()
    data = post(url, {"model": model, "input": text, "response_format": "wav", "options": options}).read()
    wall = time.perf_counter() - start
    path = "/tmp/_bench.wav"
    with open(path, "wb") as f:
        f.write(data)
    with wave.open(path) as w:
        rate = w.getframerate()
        pcm = w.readframes(w.getnframes())
    return wall, wall, pcm, rate


def stream(url, model, text, options, rate):
    start = time.perf_counter()
    first = None
    pcm = bytearray()
    body = {"model": model, "input": text, "stream_format": "sse", "response_format": "pcm",
            "options": options}
    with post(url, body, "text/event-stream") as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            event = json.loads(payload)
            if event.get("type") == "speech.audio.delta":
                if first is None:
                    first = time.perf_counter() - start
                pcm += base64.b64decode(event["audio"])
    return time.perf_counter() - start, first, bytes(pcm), rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:5900/v1/audio/speech")
    parser.add_argument("--model", default="ming-tts")
    parser.add_argument("--mode", choices=["offline", "sse"], default="sse")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--texts", default="short,medium,long")
    parser.add_argument("--out", default="")
    parser.add_argument("--rate", type=int, default=44100)
    parser.add_argument("--option", action="append", default=[], help="name=value request option")
    args = parser.parse_args()
    options = dict(item.split("=", 1) for item in args.option)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
    run = offline if args.mode == "offline" else stream
    # Warm-up: first request allocates graphs.
    run(args.url, args.model, "Kurzer Test.", options, args.rate)
    totals = []
    for name in args.texts.split(","):
        walls, firsts = [], []
        for i in range(args.repeat):
            wall, first, pcm, rate = run(args.url, args.model, TEXTS[name], options, args.rate)
            walls.append(wall)
            firsts.append(first)
            if args.out and i == 0:
                write_wav(os.path.join(args.out, f"{name}.wav"), pcm, rate)
        audio = len(pcm) / 2 / rate
        wall = statistics.median(walls)
        first = statistics.median(firsts)
        totals.append(wall / audio)
        print(f"{name:7s} audio={audio:5.2f}s wall={wall:5.2f}s ttfa={first:5.2f}s RTF={wall / audio:4.2f}",
              flush=True)
    print(f"mean RTF={statistics.mean(totals):4.2f}")


if __name__ == "__main__":
    main()
