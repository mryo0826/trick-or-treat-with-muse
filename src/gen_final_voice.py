# Generate the final sung lines: Fish TTS with the sung guides as reference
# audio (MessagePack body — JSON is rejected for inline references).
# Spoken lines l0/l13 are copied from the plain Akari takes (work/voice2/).
import json, os, shutil, subprocess, sys
import msgpack

KEY = os.environ.get("FISH_KEY")
if not KEY:
    sys.exit("FISH_KEY not set")
BASE = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(BASE, "work", "voice3")
os.makedirs(OUT, exist_ok=True)

with open(os.path.join(BASE, "work", "guides", "manifest.json")) as fh:
    manifest = json.load(fh)

for lid, info in manifest.items():
    opath = os.path.join(OUT, f"{lid}.wav")
    if os.path.exists(opath) and os.path.getsize(opath) > 10000:
        print(lid, "exists, skip")
        continue
    with open(os.path.join(BASE, info["guide"]), "rb") as fh:
        raw_audio = fh.read()
    payload = {"text": info["text"], "format": "wav",
               "references": [{"audio": raw_audio, "text": info["text"]}]}
    ppath = os.path.join(BASE, "work", "hack", f"final_{lid}.msgpack")
    with open(ppath, "wb") as fh:
        fh.write(msgpack.packb(payload, use_bin_type=True))
    r = subprocess.run(
        ["curl", "-sS", "https://api.fish.audio/v1/tts",
         "-K", "-",
         "-H", "Content-Type: application/msgpack",
         "-H", "model: s2.1-pro-free",
         "--data-binary", f"@{ppath}", "--output", opath, "-w", "%{http_code}"],
        input=f'header = "Authorization: Bearer {KEY}"\n', capture_output=True, text=True)
    ok = os.path.exists(opath) and os.path.getsize(opath) > 10000
    print(lid, r.stdout.strip(), os.path.getsize(opath) if os.path.exists(opath) else 0)
    if not ok:
        sys.exit(1)

for lid in ("l0", "l13"):
    shutil.copy(os.path.join(BASE, "work", "voice2", f"{lid}.wav"),
                os.path.join(OUT, f"{lid}.wav"))
print("voice3 done")
