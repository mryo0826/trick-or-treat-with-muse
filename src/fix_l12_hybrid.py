# The user's chosen l12 text turned out to be a hybrid of vA/vB:
#   Trick or Treat。えがおでうたおー
# (English head + sustained-vowel tail). Generate it with the v2
# recipe (cleansed guide, temperature 0.6 / top_p 0.75), install as
# the solo stem, chorus it, rebuild the faded master, recut finale.
# Also: two calmer l13 alternates (punctuation softened) measured
# for their opening-phrase F0 - NOT installed, A/B only.
import json, os, subprocess, sys
import msgpack
import numpy as np
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, os.path.dirname(__file__))
from dsp import SR, load_mono, save_wav, f0_track, grain_shift

KEY = os.environ.get("FISH_KEY")
if not KEY:
    sys.exit("FISH_KEY not set")
BASE = os.path.join(os.path.dirname(__file__), "..")
AKARI = "97eda32449cf443bbc4b8782853ca554"
SCALE_PC = {0, 2, 4, 5, 7, 8, 9, 11}
HYBRID = "Trick or Treat。えがおでうたおー"

def quantize(f):
    if f <= 0:
        return 220.0
    mf = 69 + 12 * np.log2(f / 440.0)
    best, bd = None, 1e9
    for m in range(int(np.floor(mf)) - 2, int(np.floor(mf)) + 4):
        if m % 12 in SCALE_PC and abs(m - mf) < bd:
            best, bd = m, abs(m - mf)
    return 440.0 * 2 ** ((best - 69) / 12.0)

def pitch_correct(x, strength=0.85):
    f0, hop = f0_track(x)
    tgt = np.array([quantize(f) for f in f0])
    ratio_f = np.clip((tgt / np.maximum(f0, 1.0)) ** strength, 0.6, 1.55)
    ratio_f = np.convolve(ratio_f, np.ones(6) / 6, mode="same")
    ratio = np.interp(np.arange(len(x)), np.arange(len(f0)) * hop + hop // 2, ratio_f)
    G, H = 2048, 512
    xp = np.pad(x, G)
    out = np.zeros(len(xp))
    wsum = np.zeros(len(xp))
    win = np.hanning(G)
    for p in range(0, len(xp) - G, H):
        r = float(np.mean(ratio[max(0, p - G):p])) if p >= G else float(ratio[0])
        r = min(max(r, 0.6), 1.55)
        half = int(G * r / 2)
        c = p + G // 2
        src = xp[max(0, c - half):c + half]
        if len(src) < 8:
            continue
        out[p:p + G] += np.interp(np.linspace(0, len(src) - 1, G),
                                  np.arange(len(src)), src) * win
        wsum[p:p + G] += win
    out = out / np.maximum(wsum, 0.2)
    return out[G:G + len(x)].astype(np.float32)

def cleanse(x):
    sos = butter(4, 11000, btype="low", fs=SR, output="sos")
    y = sosfiltfilt(sos, x).astype(np.float32)
    peak = np.max(np.abs(y)) + 1e-9
    idx = np.where(np.abs(y) > 0.03 * peak)[0]
    if len(idx):
        a = max(0, idx[0] - int(0.03 * SR))
        b = min(len(y), idx[-1] + int(0.05 * SR))
        y = y[a:b].copy()
        f = int(0.005 * SR)
        y[:f] *= np.linspace(0, 1, f)
        y[-f:] *= np.linspace(1, 0, f)
    return (y / (np.max(np.abs(y)) + 1e-9) * 0.70).astype(np.float32)

def flat_windows(x, win_s=0.25):
    win = int(win_s * SR)
    out = []
    for i in range(0, max(1, len(x) - win), win):
        seg = x[i:i + win]
        f = np.abs(np.fft.rfft(seg * np.hanning(win))) + 1e-9
        band = f[5:100]
        out.append(float(np.exp(np.mean(np.log(band))) / np.mean(band)))
    return out

def metrics(x):
    fw = flat_windows(x)
    return float(np.mean(fw)), (max(fw[int(len(fw) * 0.6):]) if len(fw) > 3 else 1.0)

def gen_raw(text, path):
    payload = json.dumps({"text": text, "reference_id": AKARI, "format": "wav"})
    r = subprocess.run(["curl", "-sS", "https://api.fish.audio/v1/tts",
                        "-K", "-",
                        "-H", "Content-Type: application/json",
                        "-H", "model: s2.1-pro-free", "-d", payload,
                        "--output", path, "-w", "%{http_code}"],
                       input=f'header = "Authorization: Bearer {KEY}"\n', capture_output=True, text=True)
    return r.stdout.strip()

def hack2(guide_path, text, out_path):
    with open(guide_path, "rb") as fh:
        audio = fh.read()
    payload = {"text": text, "format": "wav", "temperature": 0.6, "top_p": 0.75,
               "references": [{"audio": audio, "text": text}]}
    ppath = os.path.join(BASE, "work", "hack", "l12hyb.msgpack")
    with open(ppath, "wb") as fh:
        fh.write(msgpack.packb(payload, use_bin_type=True))
    r = subprocess.run(["curl", "-sS", "https://api.fish.audio/v1/tts",
                        "-K", "-",
                        "-H", "Content-Type: application/msgpack",
                        "-H", "model: s2.1-pro-free", "--data-binary", f"@{ppath}",
                        "--output", out_path, "-w", "%{http_code}"],
                       input=f'header = "Authorization: Bearer {KEY}"\n', capture_output=True, text=True)
    return r.stdout.strip()

def chorus(v):
    up = grain_shift(v, 2 ** (7 / 1200))
    dn = grain_shift(v, 2 ** (-7 / 1200))
    n = len(v)
    fit = lambda x: x[:n] if len(x) >= n else np.pad(x, (0, n - len(x)))
    delayed = np.concatenate([np.zeros(int(0.09 * SR), dtype=np.float32), v])[:n]
    m = v * 0.9 + fit(up) * 0.42 + fit(dn) * 0.42 + delayed * 0.30
    return (m / max(1e-9, np.max(np.abs(m))) * 0.92).astype(np.float32)

rp = os.path.join(BASE, "work", "voice2", "l12_hyb.wav")
print("hybrid raw http", gen_raw(HYBRID, rp))
guide = cleanse(pitch_correct(load_mono(rp)))
gp = os.path.join(BASE, "work", "guides_simple", "l12_hyb.wav")
save_wav(gp, guide)
best = None
for roll in range(3):
    op = os.path.join(BASE, "work", "voice4", f"l12_hyb{roll}.wav")
    code = hack2(gp, HYBRID, op)
    x = load_mono(op)
    g, tw = metrics(x)
    score = g + 0.5 * tw
    print(f"  roll{roll}: http={code} dur={len(x)/SR:.2f} garble={g:.3f} tail={tw:.2f} score={score:.3f}")
    if 2.2 <= len(x) / SR <= 3.3 and (best is None or score < best[0]):
        best = (score, op)
if best:
    winner = load_mono(best[1])
    solo = os.path.join(BASE, "work", "voice4", "l12_solo.wav")
    os.replace(solo, os.path.join(BASE, "work", "voice4", "l12_solo_en.wav"))
    save_wav(solo, winner)
    stem = os.path.join(BASE, "work", "voice4", "l12.wav")
    os.replace(stem, os.path.join(BASE, "work", "voice4", "l12_minna_en.wav"))
    save_wav(stem, chorus(winner))
    print("hybrid installed:", os.path.basename(best[1]))
    r = subprocess.run([sys.executable, os.path.join(BASE, "src", "mix_final.py"),
                        "voice4", "mix_release"], capture_output=True, text=True)
    print("mix_final:", r.returncode)
    mr = os.path.join(BASE, "out", "mix_release.wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", mr, "-af",
                    "afade=t=out:st=58.6:d=1.4",
                    os.path.join(BASE, "out", "mix_release_faded.wav")], check=True)
    os.replace(mr, os.path.join(BASE, "out", "mix_release_nofade.wav"))
    os.replace(os.path.join(BASE, "out", "mix_release_faded.wav"), mr)
    mix = load_mono(mr)
    save_wav(os.path.join(BASE, "out", "finale_full.wav"),
             mix[int(44.4 * SR):].astype(np.float32))
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i",
                    os.path.join(BASE, "out", "finale_full.wav"),
                    os.path.join(BASE, "out", "finale_full.mp3")], check=True)
    print("finale_full recut with hybrid l12")

def f0_opening(x, upto=1.2):
    f0, hop = f0_track(x)
    n = int(upto * SR / hop)
    seg = f0[:n]
    seg = seg[seg > 0]
    if len(seg) < 5:
        return 0.0, 0.0
    return float(np.percentile(seg, 90)), float(np.median(seg[:max(3, len(seg) // 6)]))

cur = load_mono(os.path.join(BASE, "work", "voice4", "l13.wav"))
p90, start = f0_opening(cur)
print(f"l13 current: opening start~{start:.0f}Hz p90={p90:.0f}Hz")
alts = []
for tag, text in [("c1", "Happy Halloween、また来年"), ("c2", "Happy Halloween。また来年")]:
    ap = os.path.join(BASE, "work", "voice2", f"l13_{tag}.wav")
    print(tag, "http", gen_raw(text, ap))
    x = load_mono(ap)
    p90, start = f0_opening(x)
    print(f"  {tag} ({text}): dur={len(x)/SR:.2f} start~{start:.0f}Hz p90={p90:.0f}Hz")
    alts.append((p90, tag, x))
alts.sort()
calm = alts[0][2]
gap = np.zeros(int(0.5 * SR), dtype=np.float32)
ab = np.concatenate([cur, gap, calm])
ab = ab / max(1e-9, np.max(np.abs(ab))) * 0.92
save_wav(os.path.join(BASE, "out", "ab_l13.wav"), ab.astype(np.float32))
subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", os.path.join(BASE, "out", "ab_l13.wav"),
                os.path.join(BASE, "out", "ab_l13.mp3")], check=True)
print(f"wrote ab_l13.mp3 (1=current, 2=calmest alt {alts[0][1]})")
