# Build "sung guides": reshape each raw Akari take so it actually follows the
# song's melody — mora segments (onset detection) are grouped onto melody
# note events, time-stretched to each note's length and pitch-shifted to the
# note. The guides are then used as Fish Audio reference audio (the "hack"),
# so the model imitates a genuinely melodic performance.
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from dsp import SR, load_mono, save_wav, f0_track, grain_shift, stretch_to, onsets, freq, midi

BASE = os.path.join(os.path.dirname(__file__), "..")
EIGHTH = (60.0 / 128) / 2  # 0.234375s at 128 BPM

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
BM = [
    ["A4", "C5", "E5", "C5", "B4", "G4", "E4", "G4"],
    ["F4", "A4", "C5", "A4", "G4", "E4", "F4", "A4"],
    ["D5", "A4", "F4", "A4", "D5", "F5", "E5", "D5"],
    ["E5", "B4", "G#4", "B4", "E5", "D5", "B4", "G#4"],
    ["A4", "C5", "E5", "A5", "G5", "E5", "C5", "E5"],
    ["F5", "E5", "C5", "A4", "C5", "F5", "E5", "C5"],
    ["D5", "F5", "A5", "F5", "E5", "D5", "F5", "A5"],
    ["G#5", "B5", "G#5", "E5", "B4", "G#4", "B4", None],
]
C_LINE = ["E5", "G5", "C6", "G5", "E5", "D5", "C5", "E5"]
G_LINE = ["D5", "G5", "B5", "G5", "D5", "B4", "G4", "B4"]

LINE_BARS = {"l1": (M[0], M[1]), "l2": (M[2], M[3]), "l3": (M[4], M[5]),
             "l4": (M[6], M[7]), "l7": (BM[0], BM[1]), "l8": (BM[2], BM[3]),
             "l9": (BM[4], BM[5]), "l10": (BM[6], BM[7]),
             "l5": (M[4], M[12]), "l6": (C_LINE, G_LINE),
             "l11": (M[0], M[6]), "l12": (M[13], M[14])}
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

def note_events(bars):
    ev = []  # (note_name, n_eighths)
    for bar in bars:
        for nm in bar:
            if nm is None and ev:
                ev[-1] = (ev[-1][0], ev[-1][1] + 1)
            elif nm is not None:
                ev.append((nm, 1))
    return ev

def build_guide(lid):
    raw = load_mono(os.path.join(BASE, "work", "voice2", f"{lid}.wav"))
    # trim to active region
    hop = 441
    env = np.array([np.sqrt(np.mean(raw[i:i + hop] ** 2)) + 1e-9
                    for i in range(0, len(raw) - hop, hop)])
    act = np.where(env > 0.04 * env.max())[0]
    raw = raw[max(0, act[0] * hop - 2000):min(len(raw), (act[-1] + 1) * hop + 2000)]
    bounds = onsets(raw)
    segs = [raw[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]
    segs = [s for s in segs if len(s) > 800]
    ev = note_events(LINE_BARS[lid])
    K, S = len(ev), len(segs)
    pieces = []
    for j, (nm, n8) in enumerate(ev):
        a = round(j * S / K)
        b = max(a + 1, round((j + 1) * S / K)) if j < K - 1 else S
        a = min(a, S - 1)
        b = min(b, S)
        audio = np.concatenate(segs[a:b]) if b > a else segs[a]
        target_len = int(n8 * EIGHTH * SR)
        f0, fh = f0_track(audio)
        voiced = f0[f0 > 0]
        f0med = float(np.median(voiced)) if len(voiced) else 220.0
        stretched = stretch_to(audio, target_len)
        f_after = f0med * (len(audio) / max(1, target_len))
        ratio = freq(midi(nm)) / max(40.0, f_after)
        pieces.append(grain_shift(stretched, ratio))
    # crossfade-concatenate (10ms)
    xf = int(0.01 * SR)
    guide = pieces[0]
    for p in pieces[1:]:
        if len(guide) > xf and len(p) > xf:
            fade_out = np.linspace(1, 0, xf)
            fade_in = np.linspace(0, 1, xf)
            guide = np.concatenate([guide[:-xf],
                                    guide[-xf:] * fade_out + p[:xf] * fade_in,
                                    p[xf:]])
        else:
            guide = np.concatenate([guide, p])
    guide = guide / max(1e-9, np.max(np.abs(guide))) * 0.9
    return guide.astype(np.float32)

os.makedirs(os.path.join(BASE, "work", "guides"), exist_ok=True)
manifest = {}
for lid in TEXT:
    g = build_guide(lid)
    path = os.path.join(BASE, "work", "guides", f"{lid}.wav")
    save_wav(path, g)
    manifest[lid] = {"guide": f"work/guides/{lid}.wav", "text": TEXT[lid],
                     "duration": round(len(g) / SR, 2)}
    print(lid, "guide", manifest[lid]["duration"], "s")
with open(os.path.join(BASE, "work", "guides", "manifest.json"), "w") as fh:
    json.dump(manifest, fh, ensure_ascii=False, indent=1)
print("guides done")
