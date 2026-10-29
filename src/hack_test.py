# Reference-hack test (idea from another AI, via the user): use the
# pitch-corrected (v2) takes as reference audio for a fresh TTS generation,
# so the model imitates the melodic prosody of the guide.
# Guide text = the same lyric line. Output: demo_hack.wav over the chorus BGM.
import base64, json, os, subprocess, sys
import msgpack
import numpy as np
from scipy.io import wavfile

KEY = os.environ.get("FISH_KEY")
if not KEY:
    sys.exit("FISH_KEY not set")
SR = 44100
BASE = os.path.join(os.path.dirname(__file__), "..")
SCALE_PC = {0, 2, 4, 5, 7, 8, 9, 11}

def load_mono(path):
    sr, d = wavfile.read(path)
    if d.ndim > 1:
        d = d[:, 0]
    return d.astype(np.float32) / 32768.0

def f0_track(x, frame=1764, hop=441):
    n = max(1, (len(x) - frame) // hop + 1)
    f0 = np.zeros(n)
    lo, hi = int(SR / 500), int(SR / 70)
    for i in range(n):
        seg = x[i * hop:i * hop + frame]
        if len(seg) < frame or np.sqrt(np.mean(seg ** 2)) < 0.01:
            continue
        seg = seg * np.hanning(frame)
        ac = np.correlate(seg, seg, "full")[frame - 1:]
        if ac[0] <= 0:
            continue
        acn = ac / ac[0]
        k = int(np.argmax(acn[lo:hi]))
        if acn[lo + k] > 0.32:
            f0[i] = SR / (lo + k)
    med = f0.copy()
    for i in range(n):
        w = f0[max(0, i - 2):i + 3]
        w = w[w > 0]
        med[i] = np.median(w) if len(w) else 0
    last = 220.0
    for i in range(n):
        if med[i] > 0:
            last = med[i]
        else:
            med[i] = last
    return med, hop

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
        out[p:p + G] += np.interp(np.linspace(0, len(src) - 1, G), np.arange(len(src)), src) * win
        wsum[p:p + G] += win
    out = out / np.maximum(wsum, 0.2)
    y = out[G:G + len(x)]
    return y / max(1e-9, np.max(np.abs(y))) * 0.9

LINES = {"l5": "ハロウィン、ハロウィン、つきの夜のパーティー",
         "l6": "トリック・オア・トリート、みんなで歌おう"}
os.makedirs(os.path.join(BASE, "work", "hack"), exist_ok=True)

new_voice = {}
for lid, text in LINES.items():
    raw = load_mono(os.path.join(BASE, "work", "voice2", f"{lid}.wav"))
    guide = pitch_correct(raw)
    gpath = os.path.join(BASE, "work", "hack", f"guide_{lid}.wav")
    wavfile.write(gpath, SR, (guide * 32767).astype(np.int16))
    with open(gpath, "rb") as fh:
        raw_audio = fh.read()
    # NOTE: inline `references` are only accepted with a MessagePack body.
    payload = {"text": text, "format": "wav",
               "references": [{"audio": raw_audio, "text": text}]}
    ppath = os.path.join(BASE, "work", "hack", f"payload_{lid}.msgpack")
    with open(ppath, "wb") as fh:
        fh.write(msgpack.packb(payload, use_bin_type=True))
    opath = os.path.join(BASE, "work", "hack", f"{lid}.wav")
    r = subprocess.run(
        ["curl", "-sS", "https://api.fish.audio/v1/tts",
         "-K", "-",
         "-H", "Content-Type: application/msgpack",
         "-H", "model: s2.1-pro-free",
         "--data-binary", f"@{ppath}", "--output", opath, "-w", "%{http_code}"],
        input=f'header = "Authorization: Bearer {KEY}"\n', capture_output=True, text=True)
    ok = os.path.exists(opath) and os.path.getsize(opath) > 10000
    print(lid, "http", r.stdout.strip(), "size", os.path.getsize(opath) if os.path.exists(opath) else 0)
    if not ok:
        with open(opath, "rb") as fh:
            print("resp head:", fh.read(300))
        sys.exit(1)
    new_voice[lid] = load_mono(opath)

DUR = 8.5
N = int(SR * DUR)
bgm_full = load_mono(os.path.join(BASE, "out", "bgm_60.wav"))
i0 = int(37.5 * SR)
bgm = bgm_full[i0:i0 + N].copy()
voice = np.zeros(N, dtype=np.float32)
for lid, start in [("l5", 0.4), ("l6", 4.15)]:
    v = new_voice[lid]
    i = int(start * SR)
    j = min(N, i + len(v))
    voice[i:j] += v[:j - i]
win = int(SR * 0.1)
rms = np.sqrt(np.convolve(voice ** 2, np.ones(win) / win, mode="same"))
duck = 1.0 - 0.5 * np.clip(rms * 4, 0, 1)
mix = bgm * 0.85 * duck + voice * 0.95
mix = mix / max(1e-9, np.max(np.abs(mix))) * 0.92
out = os.path.join(BASE, "out", "demo_hack.wav")
wavfile.write(out, SR, (mix * 32767).astype(np.int16))
print("wrote", out)
