# Finale text revision (user adjudication, 2026-10-29): the generation
# texts themselves are rephrased - a full stop after the repeated phrase
# and English orthography for the English phrases:
#   l11: ハロウィン、ハロウィン。踊ろう、朝まで
#   l12: Trick or Treat。笑顔で、歌おう
#   l13: Happy Halloween。また来年！   (spoken line, plain take)
# l11/l12 go through the usual raw -> guide -> hack flow with gates
# (duration, garble, worst-window flatness in the tail for l12's
# "utaou", where the noise was heard). l12's winner is then chorused
# (finale treatment). The mix is rebuilt and a 58.6-60.0s audio fade-out
# is applied to the master.
import json, os, subprocess, sys
import msgpack
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from dsp import SR, load_mono, save_wav, f0_track, grain_shift

KEY = os.environ.get("FISH_KEY")
if not KEY:
    sys.exit("FISH_KEY not set")
BASE = os.path.join(os.path.dirname(__file__), "..")
AKARI = "97eda32449cf443bbc4b8782853ca554"
SCALE_PC = {0, 2, 4, 5, 7, 8, 9, 11}
NEW_TEXT = {"l11": "ハロウィン、ハロウィン。踊ろう、朝まで",
            "l12": "Trick or Treat。笑顔で、歌おう",
            "l13": "Happy Halloween。また来年！"}

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
    y = out[G:G + len(x)]
    return (y / max(1e-9, np.max(np.abs(y))) * 0.9).astype(np.float32)

def flat_windows(x, win_s=0.25):
    win = int(win_s * SR)
    out = []
    for i in range(0, max(1, len(x) - win), win):
        seg = x[i:i + win]
        f = np.abs(np.fft.rfft(seg * np.hanning(win))) + 1e-9
        band = f[5:100]
        out.append(float(np.exp(np.mean(np.log(band))) / np.mean(band)))
    return out

def garble(x):
    return float(np.mean(flat_windows(x)))

def gen_raw(text, path):
    payload = json.dumps({"text": text, "reference_id": AKARI, "format": "wav"})
    r = subprocess.run(
        ["curl", "-sS", "https://api.fish.audio/v1/tts",
         "-K", "-",
         "-H", "Content-Type: application/json",
         "-H", "model: s2.1-pro-free",
         "-d", payload, "--output", path, "-w", "%{http_code}"],
        input=f'header = "Authorization: Bearer {KEY}"\n', capture_output=True, text=True)
    return r.stdout.strip()

def hack(guide_path, text, out_path):
    with open(guide_path, "rb") as fh:
        audio = fh.read()
    payload = {"text": text, "format": "wav",
               "references": [{"audio": audio, "text": text}]}
    ppath = os.path.join(BASE, "work", "hack", "fix_finale.msgpack")
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

def chorus(v):
    up = grain_shift(v, 2 ** (7 / 1200))
    dn = grain_shift(v, 2 ** (-7 / 1200))
    n = len(v)
    fit = lambda x: x[:n] if len(x) >= n else np.pad(x, (0, n - len(x)))
    delayed = np.concatenate([np.zeros(int(0.09 * SR), dtype=np.float32), v])[:n]
    m = v * 0.9 + fit(up) * 0.42 + fit(dn) * 0.42 + delayed * 0.30
    return (m / max(1e-9, np.max(np.abs(m))) * 0.92).astype(np.float32)

for lid in ("l11", "l12"):
    text = NEW_TEXT[lid]
    cur = load_mono(os.path.join(BASE, "work", "voice4", "l12_solo.wav")) \
        if lid == "l12" else load_mono(os.path.join(BASE, "work", "voice4", f"{lid}.wav"))
    cur_g = garble(cur)
    rp = os.path.join(BASE, "work", "voice2", f"{lid}_fin.wav")
    print(lid, "raw http", gen_raw(text, rp))
    guide = pitch_correct(load_mono(rp))
    gp = os.path.join(BASE, "work", "guides_simple", f"{lid}_fin.wav")
    save_wav(gp, guide)
    best = None
    for roll in range(3):
        op = os.path.join(BASE, "work", "voice4", f"{lid}_fin{roll}.wav")
        code = hack(gp, text, op)
        x = load_mono(op)
        dur = len(x) / SR
        fw = flat_windows(x)
        tail_worst = max(fw[int(len(fw) * 0.6):]) if len(fw) > 3 else 1.0
        score = garble(x) + 0.5 * tail_worst
        ok = 2.2 <= dur <= 3.3
        print(f"  roll{roll}: http={code} dur={dur:.2f} garble={garble(x):.3f} "
              f"tail_worst={tail_worst:.2f} score={score:.3f} gate={'OK' if ok else 'FAIL'}")
        if ok and (best is None or score < best[0]):
            best = (score, op)
    if best is None:
        print(lid, "no candidate passed the gate - keeping the current stem")
        continue
    winner = load_mono(best[1])
    if lid == "l12":
        solo_old = os.path.join(BASE, "work", "voice4", "l12_solo.wav")
        os.replace(solo_old, os.path.join(BASE, "work", "voice4", "l12_solo_v1.wav"))
        save_wav(solo_old, winner)
        stem = os.path.join(BASE, "work", "voice4", "l12.wav")
        os.replace(stem, os.path.join(BASE, "work", "voice4", "l12_minna_v1.wav"))
        save_wav(stem, chorus(winner))
    else:
        stem = os.path.join(BASE, "work", "voice4", "l11.wav")
        os.replace(stem, os.path.join(BASE, "work", "voice4", "l11_v1.wav"))
        save_wav(stem, winner)
    print(lid, "installed:", os.path.basename(best[1]), f"(current garble was {cur_g:.3f})")

# l13: spoken line, plain take with the new text
rp = os.path.join(BASE, "work", "voice2", "l13_fin.wav")
print("l13 raw http", gen_raw(NEW_TEXT["l13"], rp))
x = load_mono(rp)
print(f"l13 take dur={len(x)/SR:.2f}s")
if 1.6 <= len(x) / SR <= 3.6:
    stem = os.path.join(BASE, "work", "voice4", "l13.wav")
    os.replace(stem, os.path.join(BASE, "work", "voice4", "l13_old.wav"))
    save_wav(stem, x)
    print("l13 installed")
else:
    print("l13 take out of range - keeping the current stem")

r = subprocess.run([sys.executable, os.path.join(BASE, "src", "mix_final.py"),
                    "voice4", "mix_release"], capture_output=True, text=True)
print("mix_final:", r.returncode)
mr = os.path.join(BASE, "out", "mix_release.wav")
subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", mr,
                "-af", "afade=t=out:st=58.6:d=1.4",
                os.path.join(BASE, "out", "mix_release_faded.wav")], check=True)
os.replace(mr, os.path.join(BASE, "out", "mix_release_nofade.wav"))
os.replace(os.path.join(BASE, "out", "mix_release_faded.wav"), mr)
mix = load_mono(mr)
seg = mix[int(44.4 * SR):]
save_wav(os.path.join(BASE, "out", "finale_full.wav"), seg.astype(np.float32))
subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", os.path.join(BASE, "out", "finale_full.wav"),
                os.path.join(BASE, "out", "finale_full.mp3")], check=True)
print("wrote finale_full (faded master)")
