# v2 experiment: make the TTS voice "sing" by pitch-correcting it toward the
# song's scale (A minor). F0 is tracked by autocorrelation, quantized to the
# nearest scale tone (85% strength), then applied with a grain-based pitch
# shifter that preserves timing. No stacked delays this time (dry + ducking).
import os
import numpy as np
from scipy.io import wavfile

SR = 44100
BASE = os.path.join(os.path.dirname(__file__), "..")
SCALE_PC = {0, 2, 4, 5, 7, 8, 9, 11}  # C D E F G G#(Ab) A B

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
        seg_ac = acn[lo:hi]
        k = int(np.argmax(seg_ac))
        if seg_ac[k] > 0.32:
            f0[i] = SR / (lo + k)
    # median filter (window 5) + forward fill
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
    midi_f = 69 + 12 * np.log2(f / 440.0)
    best, bestd = None, 1e9
    base = int(np.floor(midi_f)) - 2
    for m in range(base, base + 6):
        if m % 12 in SCALE_PC:
            dd = abs(m - midi_f)
            if dd < bestd:
                best, bestd = m, dd
    return 440.0 * 2 ** ((best - 69) / 12.0)

def pitch_correct(x, strength=0.85):
    f0, hop = f0_track(x)
    tgt = np.array([quantize(f) for f in f0])
    ratio_frames = (tgt / f0) ** strength
    ratio_frames = np.clip(ratio_frames, 0.65, 1.5)
    # smooth (~60ms moving average over frames)
    k = 6
    ratio_frames = np.convolve(ratio_frames, np.ones(k) / k, mode="same")
    ratio = np.interp(np.arange(len(x)), np.arange(len(f0)) * hop + hop // 2, ratio_frames)
    # grain-based pitch shift, timing preserved
    G, H = 2048, 512
    xp = np.pad(x, G, mode="constant")
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
        res = np.interp(np.linspace(0, len(src) - 1, G), np.arange(len(src)), src)
        out[p:p + G] += res * win
        wsum[p:p + G] += win
    out = out / np.maximum(wsum, 0.2)
    y = out[G:G + len(x)]
    return y / max(1e-9, np.max(np.abs(y))) * 0.9

DUR = 8.5
N = int(SR * DUR)
bgm_full = load_mono(os.path.join(BASE, "out", "bgm_60.wav"))
i0 = int(37.5 * SR)
bgm = bgm_full[i0:i0 + N].copy()

voice = np.zeros(N, dtype=np.float32)
for lid, start in [("l5", 0.4), ("l6", 4.15)]:
    raw = load_mono(os.path.join(BASE, "work", "voice2", f"{lid}.wav"))
    fixed = pitch_correct(raw)
    i = int(start * SR)
    j = min(N, i + len(fixed))
    voice[i:j] += fixed[:j - i].astype(np.float32)

win = int(SR * 0.1)
rms = np.sqrt(np.convolve(voice ** 2, np.ones(win) / win, mode="same"))
duck = 1.0 - 0.5 * np.clip(rms * 4, 0, 1)
mix = bgm * 0.85 * duck + voice * 0.95
mix = mix / max(1e-9, np.max(np.abs(mix))) * 0.92

out = os.path.join(BASE, "out", "demo_v2.wav")
wavfile.write(out, SR, (mix * 32767).astype(np.int16))
print("wrote", out)
