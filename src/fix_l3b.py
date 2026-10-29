# Fix l3, round 2: the first fix flattened the guide's tail with hard
# per-mora steps; Fish's re-synthesis garbled exactly there (spectrogram:
# harmonic stacks dissolve into noise at the start of "yurete", absent in
# the raw take and the old stem). Round 2: try (a) the standard guide
# recipe from the new raw take and (b) a softened flattening (smoothed
# target steps), roll the hack twice per guide, and keep the candidate
# with the cleanest tail by measurement (spectral flatness = garble,
# pitch-step profile = the original intonation complaint).
import json, os, subprocess, sys
import msgpack
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from dsp import SR, load_mono, save_wav, f0_track, onsets

KEY = os.environ.get("FISH_KEY")
if not KEY:
    sys.exit("FISH_KEY not set")
BASE = os.path.join(os.path.dirname(__file__), "..")
CANON = "かぼちゃランタン、ゆらゆらゆれて"
SCALE_PC = {0, 2, 4, 5, 7, 8, 9, 11}

def quantize(f):
    if f <= 0:
        return 220.0
    mf = 69 + 12 * np.log2(f / 440.0)
    best, bd = None, 1e9
    for m in range(int(np.floor(mf)) - 2, int(np.floor(mf)) + 4):
        if m % 12 in SCALE_PC and abs(m - mf) < bd:
            best, bd = m, abs(m - mf)
    return 440.0 * 2 ** ((best - 69) / 12.0)

def pitch_correct_to(x, tgt, strength_f, smooth=6):
    f0, hop = f0_track(x)
    ratio_f = np.clip((tgt / np.maximum(f0, 1.0)) ** strength_f, 0.6, 1.55)
    ratio_f = np.convolve(ratio_f, np.ones(smooth) / smooth, mode="same")
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
    y = out[G:G + len(x)]
    return (y / max(1e-9, np.max(np.abs(y))) * 0.9).astype(np.float32)

def tail_bounds(raw):
    b = onsets(raw)
    segs = [(b[i], b[i + 1]) for i in range(len(b) - 1) if b[i + 1] - b[i] > int(0.05 * SR)]
    if len(segs) >= 8:
        return segs[-3:]
    n = len(raw)
    t0 = n - int(0.8 * SR)
    return [(t0 + i * (n - t0) // 3, t0 + (i + 1) * (n - t0) // 3) for i in range(3)]

def guide_standard(raw):
    f0, hop = f0_track(raw)
    tgt = np.array([quantize(f) for f in f0])
    return pitch_correct_to(raw, tgt, np.full(len(f0), 0.85))

def guide_softflat(raw):
    f0, hop = f0_track(raw)
    tgt = np.array([quantize(f) for f in f0], dtype=float)
    strength = np.full(len(f0), 0.85)
    for (s, e) in tail_bounds(raw):
        i0 = max(0, int(s / hop))
        i1 = min(len(f0), int(e / hop))
        if i1 - i0 >= 2:
            tgt[i0:i1] = float(np.median(tgt[i0:i1]))
            strength[i0:i1] = 0.95
    # smooth the steps: ~70ms ramps in target and strength across the tail
    k = max(3, int(0.07 * SR / hop))
    tgt = np.convolve(tgt, np.ones(k) / k, mode="same")
    strength = np.convolve(strength, np.ones(k) / k, mode="same")
    return pitch_correct_to(raw, tgt, strength)

def tail_window(x):
    return x[int(len(x) * 0.62):]

def garble(x):
    """Mean spectral flatness over the tail: clean singing is tonal (low),
    garbled synthesis is noise-like (high)."""
    seg = tail_window(x)
    win, hop = 1024, 256
    vals = []
    for i in range(0, max(1, len(seg) - win), hop):
        f = np.abs(np.fft.rfft(seg[i:i + win] * np.hanning(win))) + 1e-9
        band = f[5:100]
        vals.append(np.exp(np.mean(np.log(band))) / np.mean(band))
    return float(np.mean(vals)) if vals else 1.0

def pitch_profile(x):
    f0, hop = f0_track(x)
    midi = np.where(f0 > 0, 69 + 12 * np.log2(np.maximum(f0, 1) / 440.0), np.nan)
    tail = midi[int(len(midi) * 0.62):]
    tail = tail[~np.isnan(tail)]
    if len(tail) < 8:
        return 99.0, ""
    med = np.array([np.median(tail[max(0, i - 2):i + 3]) for i in range(len(tail))])
    steps = np.abs(np.diff(med))
    big = int(np.sum(steps > 2.0))          # intonation wobbles
    settle = float(np.std(med[-6:]))        # how flat the ending sits
    return big + settle, " ".join(f"{v:.0f}" for v in med[::4])

def hack(guide_path, out_path):
    with open(guide_path, "rb") as fh:
        audio = fh.read()
    payload = {"text": CANON, "format": "wav",
               "references": [{"audio": audio, "text": CANON}]}
    ppath = os.path.join(BASE, "work", "hack", "fix_l3b.msgpack")
    with open(ppath, "wb") as fh:
        fh.write(msgpack.packb(payload, use_bin_type=True))
    r = subprocess.run(
        ["curl", "-sS", "https://api.fish.audio/v1/tts",
         "-K", "-",
         "-H", "Content-Type: application/msgpack",
         "-H", "model: s2.1-pro-free",
         "--data-binary", f"@{ppath}", "--output", out_path, "-w", "%{http_code}"],
        input=f'header = "Authorization: Bearer {KEY}"\n', capture_output=True, text=True)
    return r.stdout.strip()

raw = load_mono(os.path.join(BASE, "work", "voice2", "l3a.wav"))
print("raw take A tail pitch:", pitch_profile(raw)[1])
print("old stem garble:", round(garble(load_mono(os.path.join(BASE, "work", "voice4", "l3_old.wav"))), 3),
      " round1 stem garble:", round(garble(load_mono(os.path.join(BASE, "work", "voice4", "l3.wav"))), 3))

results = []
for gname, gfn in [("std", guide_standard), ("soft", guide_softflat)]:
    guide = gfn(raw)
    gp = os.path.join(BASE, "work", "guides_simple", f"l3_{gname}.wav")
    save_wav(gp, guide)
    for roll in range(2):
        op = os.path.join(BASE, "work", "voice4", f"l3_{gname}{roll}.wav")
        code = hack(gp, op)
        x = load_mono(op)
        g = garble(x)
        pscore, prof = pitch_profile(x)
        print(f"{gname} roll{roll}: http={code} dur={len(x)/SR:.2f} garble={g:.3f} pitch={pscore:.2f} [{prof}]")
        results.append((g + pscore, op))

results.sort()
best = results[0][1]
print("winner:", best)
old = os.path.join(BASE, "work", "voice4", "l3.wav")
os.replace(old, os.path.join(BASE, "work", "voice4", "l3_round1.wav"))
os.replace(best, old)

r = subprocess.run([sys.executable, os.path.join(BASE, "src", "mix_final.py"),
                    "voice4", "mix_release"], capture_output=True, text=True)
print("mix_final:", r.returncode)
mix = load_mono(os.path.join(BASE, "out", "mix_release.wav"))
seg = mix[int(14.2 * SR):int(20.6 * SR)]
wav_out = os.path.join(BASE, "out", "excerpt_l3.wav")
save_wav(wav_out, seg.astype(np.float32))
subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", wav_out,
                os.path.join(BASE, "out", "excerpt_l3.mp3")], check=True)
print("wrote", wav_out)
