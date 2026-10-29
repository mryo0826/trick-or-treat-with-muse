# Shared DSP helpers for the Halloween song pipeline.
import numpy as np
from scipy.io import wavfile

SR = 44100
NAMES = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5,
         "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}

def midi(name):
    if len(name) >= 2 and name[1] == "#":
        letter, octv = name[:2], int(name[2:])
    else:
        letter, octv = name[:1], int(name[1:])
    return 12 * (octv + 1) + NAMES[letter]

def freq(m):
    return 440.0 * 2 ** ((m - 69) / 12.0)

def load_mono(path):
    sr, d = wavfile.read(path)
    if d.ndim > 1:
        d = d[:, 0]
    return d.astype(np.float32) / 32768.0

def save_wav(path, x):
    x = np.clip(x, -1, 1)
    wavfile.write(path, SR, (x * 32767).astype(np.int16))

def f0_track(x, frame=1764, hop=441):
    n = max(1, (len(x) - frame) // hop + 1)
    f0 = np.zeros(n)
    lo, hi = int(SR / 500), int(SR / 70)
    for i in range(n):
        seg = x[i * hop:i * hop + frame]
        if len(seg) < frame or np.sqrt(np.mean(seg ** 2)) < 0.008:
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
    out = med.copy()
    for i in range(n):
        if out[i] > 0:
            last = out[i]
        else:
            out[i] = last
    return out, hop

def grain_shift(x, ratio):
    """Pitch-shift x by a constant ratio, preserving duration (grain OLA)."""
    ratio = float(min(max(ratio, 0.5), 2.0))
    if abs(ratio - 1.0) < 0.01:
        return x.copy()
    G, H = 2048, 512
    xp = np.pad(x, G)
    out = np.zeros(len(xp))
    wsum = np.zeros(len(xp))
    win = np.hanning(G)
    for p in range(0, len(xp) - G, H):
        half = int(G * ratio / 2)
        c = p + G // 2
        src = xp[max(0, c - half):c + half]
        if len(src) < 8:
            continue
        out[p:p + G] += np.interp(np.linspace(0, len(src) - 1, G),
                                  np.arange(len(src)), src) * win
        wsum[p:p + G] += win
    out = out / np.maximum(wsum, 0.2)
    return out[G:G + len(x)].astype(np.float32)

def stretch_to(x, target_len):
    """Time-stretch x to target_len samples by resampling (pitch follows)."""
    if len(x) < 4 or target_len < 4:
        return np.zeros(max(target_len, 0), dtype=np.float32)
    return np.interp(np.linspace(0, len(x) - 1, target_len),
                     np.arange(len(x)), x).astype(np.float32)

def onsets(x, min_gap_s=0.09, thresh_ratio=0.22):
    """Energy-rise onset positions (samples). Returns boundaries incl. 0/end."""
    hop = 220  # 5ms
    n = len(x) // hop
    env = np.array([np.sqrt(np.mean(x[i * hop:(i + 1) * hop] ** 2)) + 1e-9
                    for i in range(n)])
    peak = env.max()
    bounds = [0]
    last = -10 ** 9
    for i in range(2, n - 1):
        if env[i] > thresh_ratio * peak and env[i] > env[i - 1] * 1.25 \
                and env[i - 1] <= env[i - 2] * 1.25:
            pos = i * hop
            if pos - last >= int(min_gap_s * SR):
                bounds.append(pos)
                last = pos
    bounds.append(len(x))
    # merge micro segments (<70ms) into the previous one
    merged = [bounds[0]]
    for b in bounds[1:]:
        if b - merged[-1] < int(0.07 * SR) and len(merged) > 1:
            continue
        merged.append(b)
    if merged[-1] != len(x):
        merged.append(len(x))
    return merged
