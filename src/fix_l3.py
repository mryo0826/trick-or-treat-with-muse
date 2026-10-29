# Fix l3 only: the first mora of "yurete" carries a pitch spike inherited
# from the raw take (measured: a ~10-semitone leap in the tail contour of
# both the raw take and the current stem). The user asked for "yurete" to
# be sung flat. Plan: roll two fresh raw takes (canonical text and a
# comma variant that resets the prosody), build guides whose last three
# morae are pitch-flattened to steady notes, pick the cleaner take by
# tail-contour metrics, re-run the reference hack, rebuild the mix.
import json, os, subprocess, sys
import msgpack
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from dsp import SR, load_mono, save_wav, f0_track, onsets

KEY = os.environ.get("FISH_KEY")
if not KEY:
    sys.exit("FISH_KEY not set")
BASE = os.path.join(os.path.dirname(__file__), "..")
AKARI = "97eda32449cf443bbc4b8782853ca554"
CANON = "かぼちゃランタン、ゆらゆらゆれて"
COMMA = "かぼちゃランタン、ゆらゆら、ゆれて"
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

def pitch_correct_to(x, tgt, strength_f):
    f0, hop = f0_track(x)
    ratio_f = np.clip((tgt / np.maximum(f0, 1.0)) ** strength_f, 0.6, 1.55)
    ratio_f = np.convolve(ratio_f, np.ones(4) / 4, mode="same")
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
    """Sample spans of the last three morae (yurete), best effort."""
    b = onsets(raw)
    segs = [(b[i], b[i + 1]) for i in range(len(b) - 1) if b[i + 1] - b[i] > int(0.05 * SR)]
    if len(segs) >= 8:
        return segs[-3:]
    n = len(raw)
    t0 = n - int(0.8 * SR)
    return [(t0 + i * (n - t0) // 3, t0 + (i + 1) * (n - t0) // 3) for i in range(3)]

def build_guide(raw):
    f0, hop = f0_track(raw)
    tgt = np.array([quantize(f) for f in f0])
    strength = np.full(len(f0), 0.85)
    for (s, e) in tail_bounds(raw):
        i0 = max(0, int(s / hop))
        i1 = min(len(f0), int(e / hop))
        if i1 - i0 >= 2:
            med = float(np.median(tgt[i0:i1]))
            tgt[i0:i1] = med          # one steady note per mora
            strength[i0:i1] = 1.0     # fully flatten the tail contour
    return pitch_correct_to(raw, tgt, strength)

def tail_spikiness(x):
    f0, hop = f0_track(x)
    midi = np.where(f0 > 0, 69 + 12 * np.log2(np.maximum(f0, 1) / 440.0), np.nan)
    n = len(midi)
    tail = midi[int(n * 0.68):]
    tail = tail[~np.isnan(tail)]
    if len(tail) < 8:
        return 99.0
    med = np.array([np.median(tail[max(0, i - 2):i + 3]) for i in range(len(tail))])
    return float(np.max(np.abs(np.diff(med))) + np.std(med))

def gen_raw(text, path):
    payload = json.dumps({"text": text, "reference_id": AKARI, "format": "wav"})
    r = subprocess.run(
        ["curl", "-sS", "https://api.fish.audio/v1/tts",
         "-K", "-",
         "-H", "Content-Type: application/json",
         "-H", "model: s2.1-pro-free",
         "-d", payload, "--output", path, "-w", "%{http_code}"],
        input=f'header = "Authorization: Bearer {KEY}"\n', capture_output=True, text=True)
    print("raw", text[:12], r.stdout.strip(), os.path.getsize(path))

def hack(guide_path, guide_text, out_path):
    with open(guide_path, "rb") as fh:
        audio = fh.read()
    payload = {"text": CANON, "format": "wav",
               "references": [{"audio": audio, "text": guide_text}]}
    ppath = os.path.join(BASE, "work", "hack", "fix_l3.msgpack")
    with open(ppath, "wb") as fh:
        fh.write(msgpack.packb(payload, use_bin_type=True))
    r = subprocess.run(
        ["curl", "-sS", "https://api.fish.audio/v1/tts",
         "-K", "-",
         "-H", "Content-Type: application/msgpack",
         "-H", "model: s2.1-pro-free",
         "--data-binary", f"@{ppath}", "--output", out_path, "-w", "%{http_code}"],
        input=f'header = "Authorization: Bearer {KEY}"\n', capture_output=True, text=True)
    print("hack ->", out_path, r.stdout.strip(), os.path.getsize(out_path))

os.makedirs(os.path.join(BASE, "work", "voice2"), exist_ok=True)
cands = []
for tag, text in [("a", CANON), ("b", COMMA)]:
    rp = os.path.join(BASE, "work", "voice2", f"l3{tag}.wav")
    gen_raw(text, rp)
    raw = load_mono(rp)
    guide = build_guide(raw)
    gp = os.path.join(BASE, "work", "guides_simple", f"l3{tag}.wav")
    save_wav(gp, guide)
    score = tail_spikiness(guide)
    print(f"take {tag}: dur={len(raw)/SR:.2f}s guide tail spikiness={score:.2f}")
    cands.append((score, tag, text, gp))

cands.sort()
score, tag, text, gp = cands[0]
print("chosen take:", tag, "score", round(score, 2))
old = os.path.join(BASE, "work", "voice4", "l3.wav")
if os.path.exists(old):
    os.replace(old, os.path.join(BASE, "work", "voice4", "l3_old.wav"))
hack(gp, text, old)
new = load_mono(old)
print(f"new stem: dur={len(new)/SR:.2f}s tail spikiness={tail_spikiness(new):.2f}")

r = subprocess.run([sys.executable, os.path.join(BASE, "src", "mix_final.py"),
                    "voice4", "mix_release"], capture_output=True, text=True)
print("mix_final:", r.returncode, r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr[-300:])

mix = load_mono(os.path.join(BASE, "out", "mix_release.wav"))
i0, i1 = int(14.2 * SR), int(20.6 * SR)
seg = mix[i0:i1]
wav_out = os.path.join(BASE, "out", "excerpt_l3.wav")
save_wav(wav_out, seg.astype(np.float32))
subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", wav_out,
                os.path.join(BASE, "out", "excerpt_l3.mp3")], check=True)
print("wrote", wav_out)
