# Generate the voice lines with Fish Audio API (model: s2.1-pro-free).
# Uses curl (the sandbox proxy mangles chunked responses for urllib).
# API key comes from the FISH_KEY environment variable (never stored in files).
import json, os, subprocess

API = "https://api.fish.audio/v1/tts"
MODEL_INFO = "https://api.fish.audio/model/"
KEY = os.environ["FISH_KEY"]
REF = "44b6e5eeab214296bcfd73e767225229"

BASE = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(BASE, "work", "voice")
os.makedirs(OUT, exist_ok=True)

LINES = {
    "l0": "ハッピーハロウィン!ミューズだよ。今夜は歌っちゃうぞ!",
    "l1": "トリック・オア・トリート、おかしをちょうだい",
    "l2": "くれなきゃいたずら、しちゃうぞ、ほら",
    "l3": "かぼちゃランタン、ゆらゆらゆれて",
    "l4": "おばけもまじょも、あつまる夜",
    "l5": "ハロウィン、ハロウィン、つきの夜のパーティー",
    "l6": "トリック・オア・トリート、みんなで歌おう",
}

def curl(args):
    return subprocess.run(["curl", "-sS", "-K", "-", *args], capture_output=True,
                          text=True, timeout=180,
                          input=f'header = "Authorization: Bearer {KEY}"\n')

# voice model metadata for the record (best effort)
r = curl([MODEL_INFO + REF])
try:
    info = json.loads(r.stdout)
    keep = {k: info.get(k) for k in ("title", "languages", "tags", "visibility") if k in info}
    print("model info:", json.dumps(keep, ensure_ascii=False))
except Exception:
    print("model info unavailable:", r.stdout[:200], r.stderr[:200])

for lid, text in LINES.items():
    path = os.path.join(OUT, f"{lid}.wav")
    body = json.dumps({"text": text, "reference_id": REF, "format": "wav"})
    r = curl([API,
              "-H", "Content-Type: application/json",
              "-H", "model: s2.1-pro-free",
              "-d", body, "--output", path,
              "-w", "%{http_code}"])
    size = os.path.getsize(path) if os.path.exists(path) else 0
    print(lid, "http", r.stdout.strip(), size, "bytes")
    if r.stdout.strip() != "200" or size < 1000:
        raise SystemExit(f"generation failed for {lid}: {r.stderr[:300]}")
print("done")
