# Fix l11 only: the phrase "おどろうあさまで" ran together. Regenerate the raw
# take with a comma ("おどろう、朝まで"), rebuild its pitch-corrected guide,
# re-run the reference hack, and cut an in-context excerpt for the user.
import json, os, subprocess, sys
import msgpack
import numpy as np
from scipy.io import wavfile

sys.path.insert(0, os.path.dirname(__file__))
from dsp import SR, load_mono, save_wav, f0_track

KEY = os.environ.get("FISH_KEY")
if not KEY:
    sys.exit("FISH_KEY not set")
BASE = os.path.join(os.path.dirname(__file__), "..")
AKARI = "97eda32449cf443bbc4b8782853ca554"
TEXT = "ハロウィン、ハロウィン、おどろう、朝まで"
SCALE_PC = {0, 2, 4, 5, 7, 8, 9, 11}

def quantize(f):
    mf = 69 + 12 * np.log2(f / 440.0)
    best, bd = None, 1e9
    for m in range(int(np.floor(mf)) - 2, int(np.floor(mf)) + 4):
        if m % 12 in SCALE_PC and abs(m - mf) < bd:
            best, bd = m, abs(m - mf)
    return 440.0 * 2 ** ((best - 69) / 12.0)

def pitch_correct(x, strength=0.85):
    f0, hop = f0_track(x)
    tgt = np.array([quantize(f) for f in f0])
    ratio_f = np.clip((tgt / f0) ** strength, 0.65, 1.5)
    ratio_f = np.convolve(ratio_f, np.ones(6) / 6, mode="same")
    ratio = np.interp(np.arange(len(x)), np.arange(len(f0)) * hop + hop // 2, ratio_f)
    G, H = 2048, 512
    xp = np.pad(x, G)
    out = np.zeros(len(xp))
    wsum = np.zeros(len(xp))
    win = np.hanning(G)
    for p in range(0, len(xp) - G, H):
        r = float(np.mean(ratio[max(0, p - G):p])) if p >= G else float(ratio[0])
        r = min(max(r, 0.65), 1.5)
        half = int(G * r / 2)
        c = p + G // 2
        src = xp[max(0, c - half):c + half]
        if len(src) < 8:
            continue
        out[p:p + G] += np.interp(np.linspace(0, len(src) - 1, G),
                                  np.arange(len(src)), src) * win
        wsum[p:p + G] += win
    out = out / np.maximum(wsum, 0.2)
    y = out[G:G + len(x)]
    return (y / max(1e-9, np.max(np.abs(y))) * 0.9).astype(np.float32)

# 1) fresh raw take with the comma
raw_path = os.path.join(BASE, "work", "voice2", "l11b.wav")
payload = json.dumps({"text": TEXT, "reference_id": AKARI, "format": "wav"})
r = subprocess.run(
    ["curl", "-sS", "https://api.fish.audio/v1/tts",
     "-K", "-",
     "-H", "Content-Type: application/json",
     "-H", "model: s2.1-pro-free",
     "-d", payload, "--output", raw_path, "-w", "%{http_code}"],
    input=f'header = "Authorization: Bearer {KEY}"\n', capture_output=True, text=True)
print("raw l11b", r.stdout.strip(), os.path.getsize(raw_path))

# 2) guide + hack generation -> replaces work/voice4/l11.wav (old kept aside)
raw = load_mono(raw_path)
guide = pitch_correct(raw)
gpath = os.path.join(BASE, "work", "guides_simple", "l11b.wav")
save_wav(gpath, guide)
with open(gpath, "rb") as fh:
    raw_audio = fh.read()
payload = {"text": TEXT, "format": "wav",
           "references": [{"audio": raw_audio, "text": TEXT}]}
ppath = os.path.join(BASE, "work", "hack", "fix_l11.msgpack")
with open(ppath, "wb") as fh:
    fh.write(msgpack.packb(payload, use_bin_type=True))
old = os.path.join(BASE, "work", "voice4", "l11.wav")
if os.path.exists(old):
    os.replace(old, os.path.join(BASE, "work", "voice4", "l11_old.wav"))
r = subprocess.run(
    ["curl", "-sS", "https://api.fish.audio/v1/tts",
     "-K", "-",
     "-H", "Content-Type: application/msgpack",
     "-H", "model: s2.1-pro-free",
     "--data-binary", f"@{ppath}", "--output", old, "-w", "%{http_code}"],
    input=f'header = "Authorization: Bearer {KEY}"\n', capture_output=True, text=True)
sr, d = wavfile.read(old)
print("new l11", r.stdout.strip(), os.path.getsize(old), round(len(d) / sr, 2), "s")

# 3) in-context excerpt: BGM 44.5-50.5s, line at its slot (45.4)
v = load_mono(old)
bgm_full = load_mono(os.path.join(BASE, "out", "bgm_60.wav"))
i0 = int(44.5 * SR)
N = int(6.0 * SR)
bgm = bgm_full[i0:i0 + N].copy()
voice = np.zeros(N, dtype=np.float32)
i = int(0.9 * SR)
j = min(N, i + len(v))
voice[i:j] += v[:j - i]
win = int(SR * 0.1)
rms = np.sqrt(np.convolve(voice ** 2, np.ones(win) / win, mode="same"))
duck = 1.0 - 0.5 * np.clip(rms * 4, 0, 1)
mix = bgm * 0.85 * duck + voice * 0.95
mix = mix / max(1e-9, np.max(np.abs(mix))) * 0.92
out = os.path.join(BASE, "out", "excerpt_l11.wav")
wavfile.write(out, SR, (mix * 32767).astype(np.int16))
print("wrote", out)
