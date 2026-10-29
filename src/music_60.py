# 60-second techno version of the Halloween song (all-original composition).
# 128 BPM, 32 bars = 60.0s. Structure: Intro 4 / A 8 / B 8 / Chorus 8 / Outro 4.
# Anti-monotony: section contrast (kick drops in Intro and B-breakdown),
# drum fills at section ends, riser + impact into the Chorus, B-section chord
# change (Am-F-Dm-E), Chorus melody variation and lift (C/G chords),
# synth stab answers after sung lines in A and Chorus.
import numpy as np
from scipy.io import wavfile
import os

SR = 44100
BPM = 128
BEAT = 60.0 / BPM
BAR = BEAT * 4
BARS = 32
DUR = BAR * BARS  # 60.0
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "out")
os.makedirs(OUT_DIR, exist_ok=True)

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

N = int(SR * (DUR + 1.0))

def add_to(buf, t0, wave, gain=1.0):
    i = int(t0 * SR)
    j = min(len(buf), i + len(wave))
    if j > i:
        buf[i:j] += wave[: j - i] * gain

def one_pole(x, a):
    y = np.empty_like(x)
    acc = 0.0
    for i, v in enumerate(x):
        acc += a * (v - acc)
        y[i] = acc
    return y

def saw(m, dur):
    f = freq(m)
    t = np.arange(int(SR * dur)) / SR
    return 2 * ((f * t) % 1.0) - 1

def kick(dur=0.24):
    t = np.arange(int(SR * dur)) / SR
    f = 130 * np.exp(-t * 16) + 44
    return np.sin(np.cumsum(2 * np.pi * f / SR)) * np.exp(-t * 13)

def clap(dur=0.18):
    t = np.arange(int(SR * dur)) / SR
    n = np.random.default_rng(11).standard_normal(len(t))
    return one_pole(n, 0.35) * np.exp(-t * 22) * 0.9

def hat(dur, decay, gain, seed=5):
    t = np.arange(int(SR * dur)) / SR
    n = np.random.default_rng(seed).standard_normal(len(t))
    return (n - one_pole(n, 0.08)) * np.exp(-t * decay) * gain

def tom(m, dur=0.16):
    t = np.arange(int(SR * dur)) / SR
    return np.sin(2 * np.pi * freq(m) * t) * np.exp(-t * 18)

def noise_riser(dur):
    t = np.arange(int(SR * dur)) / SR
    n = np.random.default_rng(3).standard_normal(len(t))
    lp = np.empty_like(n)
    acc = 0.0
    for i, v in enumerate(n):  # time-varying lowpass -> progressively brighter
        acc += (0.02 + 0.25 * (i / len(n))) * (v - acc)
        lp[i] = acc
    return (n - lp) * (t / dur) ** 2 * 0.5

def impact():
    t = np.arange(int(SR * 0.9)) / SR
    boom = np.sin(np.cumsum(2 * np.pi * (90 * np.exp(-t * 6) + 38) / SR)) * np.exp(-t * 5)
    n = np.random.default_rng(9).standard_normal(len(t))
    crash = (n - one_pole(n, 0.05)) * np.exp(-t * 7) * 0.4
    return boom * 0.9 + crash

# bar -> (bass root, pad letter)
ROOTS = (["A1", "A1", "F1", "E1"] +                 # Intro
         ["A1", "A1", "F1", "E1"] * 2 +              # A
         ["A1", "F1", "D2", "E1"] * 2 +              # B (Dm enters)
         ["A1", "F1", "C2", "G1", "A1", "F1", "E1", "A1"] +  # Chorus
         ["A1", "F1", "E1", "A1"])                   # Outro
PAD = {"A": ["A3", "C4", "E4"], "F": ["F3", "A3", "C4"],
       "E": ["E3", "G#3", "B3"], "D": ["D3", "F3", "A3"],
       "C": ["C4", "E4", "G4"], "G": ["G3", "B3", "D4"]}
M = [
    ["A4", "C5", "E5", "A5", "G5", "E5", "C5", "E5"],
    ["A5", "G5", "E5", "D5", "C5", "D5", "E5", None],
    ["F5", "A5", "G5", "F5", "E5", "F5", "A5", "G5"],
    ["E5", "G#5", "B5", "G#5", "E5", "D5", "B4", None],
    ["A4", "C5", "E5", "A5", "C6", "A5", "G5", "E5"],
    ["E5", "D5", "C5", "D5", "E5", "C5", "A4", None],
    ["F4", "A4", "C5", "F5", "E5", "C5", "A4", "C5"],
    ["B4", "D5", "E5", "G#5", "B5", "G#5", "E5", None],
    ["A5", "G5", "E5", "C5", "D5", "E5", "A4", "C5"],
    ["D5", "F5", "A5", "F5", "D5", "F5", "A5", "G5"],
    ["E5", "C5", "A4", "C5", "E5", "A5", "G5", "E5"],
    ["G#5", "B5", "G#5", "E5", "D5", "B4", "G#4", "B4"],
    ["A5", "F5", "C5", "F5", "A5", "G5", "F5", "E5"],
    ["D5", "B4", "G#4", "B4", "E5", None, None, None],
    ["A4", None, None, None, "A5", None, None, None],
    ["A4", None, "E5", None, "A5", None, None, None],
]
BM = [  # B-section counter melody (lower, spookier)
    ["A4", "C5", "E5", "C5", "B4", "G4", "E4", "G4"],
    ["F4", "A4", "C5", "A4", "G4", "E4", "F4", "A4"],
    ["D5", "A4", "F4", "A4", "D5", "F5", "E5", "D5"],
    ["E5", "B4", "G#4", "B4", "E5", "D5", "B4", "G#4"],
    ["A4", "C5", "E5", "A5", "G5", "E5", "C5", "E5"],
    ["F5", "E5", "C5", "A4", "C5", "F5", "E5", "C5"],
    ["D5", "F5", "A5", "F5", "E5", "D5", "F5", "A5"],
    ["G#5", "B5", "G#5", "E5", "B4", "G#4", "B4", None],
]
MELODY = ([None] * 4 + M[0:8] + BM +
          [M[4], M[12],
           ["E5", "G5", "C6", "G5", "E5", "D5", "C5", "E5"],   # over C
           ["D5", "G5", "B5", "G5", "D5", "B4", "G4", "B4"],   # over G
           M[0], M[6], M[13], M[14]] +
          [M[15], M[14], None, None])
assert len(ROOTS) == BARS and len(MELODY) == BARS

drums = np.zeros(N)
bass_bus = np.zeros(N)
pad_bus = np.zeros(N)
lead_bus = np.zeros(N)
arp_bus = np.zeros(N)
fx_bus = np.zeros(N)

SIX = BEAT / 4
FILL_BARS = {3, 11, 19, 27}
for bar in range(BARS):
    t0 = bar * BAR
    root_m = midi(ROOTS[bar])
    letter = ROOTS[bar][0]
    sec = "intro" if bar < 4 else "A" if bar < 12 else "B" if bar < 20 else "chorus" if bar < 28 else "outro"
    # --- drums by section ---
    kick_bars = (sec in ("A", "chorus")) or (sec == "B" and bar >= 16) or (sec == "outro" and bar < 30)
    if kick_bars:
        for q in range(4):
            add_to(drums, t0 + q * BEAT, kick(), 0.95)
    clap_bars = (sec in ("A", "chorus")) or (sec == "B" and bar >= 18)
    if clap_bars:
        for q in (1, 3):
            add_to(drums, t0 + q * BEAT, clap(), 0.55)
            if sec == "chorus":
                add_to(drums, t0 + q * BEAT, clap(), 0.3)
    if sec != "intro" or bar >= 2:
        for e in range(4):
            add_to(drums, t0 + e * BEAT + BEAT / 2, hat(0.09, 46, 0.30))
        for s16 in range(16):
            add_to(drums, t0 + s16 * SIX, hat(0.03, 120, 0.10))
    if bar in FILL_BARS:  # tom fill in the second half of the bar
        fill_notes = ["E4", "C4", "A3", "F3", "E3", "C3", "A2", "A2"]
        for i, nm in enumerate(fill_notes):
            add_to(drums, t0 + BAR / 2 + i * (BAR / 16), tom(midi(nm)), 0.5)
    # --- bass ---
    if sec == "A" or sec == "chorus" or (sec == "B" and bar >= 16):
        for s16 in range(16):
            oct_up = 12 if s16 in (6, 14) else 0
            w = saw(root_m + 12 + oct_up, SIX * 0.92)
            env = np.exp(-np.arange(len(w)) / SR * 9)
            add_to(bass_bus, t0 + s16 * SIX, one_pole(w, 0.22) * env, 0.5)
    elif sec == "B":  # breakdown: sparse 8th-note roots
        for e in range(8):
            w = saw(root_m + 12, BEAT * 0.4)
            env = np.exp(-np.arange(len(w)) / SR * 6)
            add_to(bass_bus, t0 + e * BEAT / 2, one_pole(w, 0.22) * env, 0.42)
    elif sec == "outro" and bar < 30:
        for e in range(8):
            w = saw(root_m + 12, BEAT * 0.4)
            env = np.exp(-np.arange(len(w)) / SR * 6)
            add_to(bass_bus, t0 + e * BEAT / 2, one_pole(w, 0.22) * env, 0.4)
    # --- pad (skip last bar) ---
    if bar < BARS - 1:
        for nm in PAD[letter]:
            m = midi(nm)
            for det in (-4, 3):
                f = freq(m) * (1 + det / 1000.0)
                t = np.arange(int(SR * BAR)) / SR
                w = (2 * ((f * t) % 1.0) - 1) * np.minimum(1, t * 3) * np.minimum(1, (BAR - t) * 3)
                add_to(pad_bus, t0, one_pole(w, 0.05), 0.055)
    # --- arp in intro/outro/B (16th chord tones) ---
    if sec in ("intro", "outro", "B"):
        tones = [midi(n) for n in PAD[letter]] + [midi(PAD[letter][0]) + 12]
        pat = [0, 1, 2, 3, 2, 1]
        g = 0.16 if sec == "B" else 0.22
        for s16 in range(16):
            w = saw(tones[pat[s16 % 6]], SIX * 0.8)
            env = np.exp(-np.arange(len(w)) / SR * 14)
            add_to(arp_bus, t0 + s16 * SIX, one_pole(w, 0.35) * env, g)
    # --- lead melody ---
    mel = MELODY[bar]
    if mel:
        lgain = 0.34 if sec == "chorus" else 0.28
        for e, nm in enumerate(mel):
            if nm:
                w = saw(midi(nm), BEAT * 0.46)
                env = np.exp(-np.arange(len(w)) / SR * 3.2)
                add_to(lead_bus, t0 + e * BEAT / 2, one_pole(w, 0.3) * env, lgain)

# risers and chorus impact
add_to(fx_bus, 3 * BAR + BAR / 2, noise_riser(BAR / 2), 0.5)   # into A
add_to(fx_bus, 18 * BAR, noise_riser(2 * BAR), 0.9)            # into Chorus
add_to(fx_bus, 20 * BAR, impact(), 1.0)
add_to(fx_bus, 26 * BAR, noise_riser(2 * BAR), 0.5)            # into final chorus bars

# synth stab answers after each sung line (A and Chorus line slots)
LINE_SLOTS = [7.9, 11.65, 15.4, 19.15, 37.9, 41.65, 45.4, 49.15]
for ls in LINE_SLOTS:
    bar = int(ls // BAR)
    for nm in PAD[ROOTS[bar][0]]:
        w = saw(midi(nm) + 12, 0.14)
        env = np.exp(-np.arange(len(w)) / SR * 16)
        add_to(lead_bus, ls + 2.35, one_pole(w, 0.3) * env, 0.16)

# dotted-eighth delay on lead
dly = int(SR * BEAT * 0.75)
lead_bus[dly:] += lead_bus[:-dly] * 0.30
lead_bus[2 * dly:] += lead_bus[:-2 * dly] * 0.16

# sidechain-style ducking
phase = (np.arange(N) / SR) % BEAT
duck = 1.0 - 0.42 * np.exp(-phase * 10)
bass_bus *= duck
pad_bus *= duck

track = drums + bass_bus + pad_bus + lead_bus + arp_bus + fx_bus
fade = int(SR * 1.2)
track[-fade:] *= np.linspace(1, 0, fade)
track = track / max(1e-9, np.max(np.abs(track))) * 0.85

wav_path = os.path.join(OUT_DIR, "bgm_60.wav")
wavfile.write(wav_path, SR, (track * 32767).astype(np.int16))
print("wrote", wav_path, f"{len(track)/SR:.1f}s")
