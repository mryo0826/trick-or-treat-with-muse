# Rollback + scale-up of the APPROVED hack demo method: guides are the raw
# Akari takes with gentle scale-quantized pitch correction (same recipe as
# src/hack_test.py / demo_v2), used as Fish reference audio via MessagePack.
# The melody-forced guides (guide_sing.py) degraded quality and were dropped.
import json, os, subprocess, sys
import msgpack
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from dsp import SR, load_mono, save_wav, f0_track

KEY = os.environ.get("FISH_KEY")
if not KEY:
    sys.exit("FISH_KEY not set")
BASE = os.path.join(os.path.dirname(__file__), "..")
SCALE_PC = {0, 2, 4, 5, 7, 8, 9, 11}
TEXT = {"l1": "トリック・オア・トリート、おかしをちょうだい",
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
        "l12": "トリック・オア・トリート、えがおでうたおう"}

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

GDIR = os.path.join(BASE, "work", "guides_simple")
ODIR = os.path.join(BASE, "work", "voice4")
os.makedirs(GDIR, exist_ok=True)
os.makedirs(ODIR, exist_ok=True)
os.makedirs(os.path.join(BASE, "work", "hack"), exist_ok=True)

for lid, text in TEXT.items():
    opath = os.path.join(ODIR, f"{lid}.wav")
    if os.path.exists(opath) and os.path.getsize(opath) > 10000:
        print(lid, "exists, skip")
        continue
    raw = load_mono(os.path.join(BASE, "work", "voice2", f"{lid}.wav"))
    guide = pitch_correct(raw)
    gpath = os.path.join(GDIR, f"{lid}.wav")
    save_wav(gpath, guide)
    with open(gpath, "rb") as fh:
        raw_audio = fh.read()
    payload = {"text": text, "format": "wav",
               "references": [{"audio": raw_audio, "text": text}]}
    ppath = os.path.join(BASE, "work", "hack", f"all_{lid}.msgpack")
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

import shutil
for lid in ("l0", "l13"):
    shutil.copy(os.path.join(BASE, "work", "voice2", f"{lid}.wav"),
                os.path.join(ODIR, f"{lid}.wav"))
print("voice4 done")
