# Generate all lyric lines with the "Akari" voice (generic anime-style female,
# no character tie). Skips lines that already exist in work/voice2/.
import json, os, subprocess, sys

KEY = os.environ.get("FISH_KEY")
if not KEY:
    sys.exit("FISH_KEY not set")
REF = "97eda32449cf443bbc4b8782853ca554"  # あかり Akari
BASE = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(BASE, "work", "voice2")
os.makedirs(OUT, exist_ok=True)

LINES = {
    "l0": "ハッピーハロウィン!ミューズだよ。今夜は歌っちゃうぞ!",
    "l1": "トリック・オア・トリート、おかしをちょうだい",
    "l2": "くれなきゃいたずら、しちゃうぞ、ほら",
    "l3": "かぼちゃランタン、ゆらゆらゆれて",
    "l4": "おばけもまじょも、あつまる夜",
    "l5": "ハロウィン、ハロウィン、つきの夜のパーティー",
    "l6": "トリック・オア・トリート、みんなで歌おう",
    "l7": "まっくらやみよを、ともすランタン",
    "l8": "コウモリぱたぱた、そらをまわる",
    "l9": "くろねこひょっこり、かおをだして",
    "l10": "みんなあつまれ、おかしのいえへ",
    "l11": "ハロウィン、ハロウィン、おどろうあさまで",
    "l12": "トリック・オア・トリート、えがおでうたおう",
    "l13": "ハッピーハロウィン!またらいねん!",
}

for lid, text in LINES.items():
    path = os.path.join(OUT, f"{lid}.wav")
    if os.path.exists(path) and os.path.getsize(path) > 10000:
        print(lid, "exists, skip")
        continue
    payload = json.dumps({"text": text, "reference_id": REF, "format": "wav"})
    r = subprocess.run(
        ["curl", "-sS", "https://api.fish.audio/v1/tts",
         "-K", "-",
         "-H", "Content-Type: application/json",
         "-H", "model: s2.1-pro-free",
         "-d", payload, "--output", path, "-w", "%{http_code}"],
        input=f'header = "Authorization: Bearer {KEY}"\n', capture_output=True, text=True)
    print(lid, r.stdout.strip(), os.path.getsize(path) if os.path.exists(path) else 0)
print("done")
