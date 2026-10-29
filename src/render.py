# Render the Halloween video frames (1080x1920, 30fps) and encode with ffmpeg.
# Layers: pre-rendered parallax background (sky/far/mid/ground), a shaded
# moon layer swelling over 4 bars, pumpkins hopping on the beat, the Muse
# stage window dancing (bounce/sway/tilt/squash) around real captured
# footage, amplitude lip-sync mouth, bats flapping on 8th notes, fog,
# twinkling stars, a bar-pulsing title and lyric telops. Each scene element
# dances to its own "instrument part" of the 128 BPM beat.
# Usage: python3 render.py <seconds> <audio_wav> <out_mp4>
import json, math, os, random, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

BASE = os.path.join(os.path.dirname(__file__), "..")
W, H, FPS = 1080, 1920, 30
SECONDS = float(sys.argv[1]); AUDIO = sys.argv[2]; OUT = sys.argv[3]
random.seed(31)

BEAT = 60 / 128  # the song is 128 BPM
BAR = BEAT * 4

def _load_font(name, size):
    p = os.path.join(BASE, "assets", "fonts", name)
    return ImageFont.truetype(p, size) if os.path.exists(p) else None

_FONT_CACHE = {}
def font_title(size):
    key = ("t", size)
    if key not in _FONT_CACHE:
        _FONT_CACHE[key] = _load_font("MochiyPopOne-Regular.ttf", size) or font(size)
    return _FONT_CACHE[key]

def font_lyric(size):
    key = ("l", size)
    if key not in _FONT_CACHE:
        _FONT_CACHE[key] = _load_font("MPLUSRounded1c-Bold.ttf", size) or font(size)
    return _FONT_CACHE[key]

def font(size):
    for p in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
              "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()

# ================= background (pre-rendered once) =================
def build_background():
    img = Image.new("RGB", (W, H))
    px = img.load()
    top, mid, bot = (14, 10, 44), (44, 24, 86), (22, 13, 38)
    for y in range(H):
        t = y / H
        if t < 0.6:
            k = t / 0.6; c = tuple(int(top[i] + (mid[i] - top[i]) * k) for i in range(3))
        else:
            k = (t - 0.6) / 0.4; c = tuple(int(mid[i] + (bot[i] - mid[i]) * k) for i in range(3))
        for x in range(W):
            px[x, y] = c
    d = ImageDraw.Draw(img, "RGBA")
    for _ in range(190):  # stars
        x, y = random.randint(0, W), random.randint(0, 1200)
        r = random.choice([1, 1, 2, 2, 3]); a = random.randint(110, 255)
        d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 250, 220, a))
    # (the moon is drawn per-frame as its own layer: see MOON below)
    # far layer: mountains + castle silhouette (blurred for depth)
    far = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    fd = ImageDraw.Draw(far)
    fd.polygon([(0, 1180), (180, 1020), (360, 1150), (560, 980), (760, 1140),
                (940, 1030), (1080, 1160), (1080, 1400), (0, 1400)], fill=(38, 26, 74, 255))
    fd.rectangle([742, 838, 806, 990], fill=(38, 26, 74, 255))           # castle tower
    fd.polygon([(730, 838), (874, 838), (774, 762)], fill=(38, 26, 74, 255))
    fd.rectangle([770, 900, 786, 936], fill=(255, 196, 92, 255))          # lit window
    far = far.filter(ImageFilter.GaussianBlur(2.2))
    img.paste(Image.alpha_composite(img.convert("RGBA"), far).convert("RGB"), (0, 0))
    d = ImageDraw.Draw(img, "RGBA")
    # mid layer: haunted house (left) + dead tree (right)
    d.rectangle([58, 1080, 300, 1330], fill=(16, 12, 34, 255))
    d.polygon([(40, 1080), (318, 1080), (179, 962)], fill=(12, 9, 28, 255))
    d.rectangle([150, 1188, 208, 1330], fill=(10, 8, 22, 255))
    for wx, wy in [(92, 1128), (232, 1128)]:
        d.rectangle([wx, wy, wx + 34, wy + 46], fill=(255, 176, 66, 255))
        d.line([wx + 17, wy, wx + 17, wy + 46], fill=(16, 12, 34, 255), width=3)
    def branch(x, y, ang, ln, wdt):
        if ln < 14 or wdt <= 0:
            return
        x2, y2 = x + ln * math.cos(ang), y - ln * math.sin(ang)
        d.line([x, y, x2, y2], fill=(12, 9, 24, 255), width=int(wdt))
        branch(x2, y2, ang + 0.42, ln * 0.72, wdt - 2)
        branch(x2, y2, ang - 0.35, ln * 0.78, wdt - 2)
    d.line([962, 1330, 962, 1120], fill=(12, 9, 24, 255), width=20)
    branch(962, 1160, 0.5, 92, 9); branch(962, 1200, 2.6, 84, 9)
    # ground + perspective path
    d.rectangle([0, 1330, W, H], fill=(11, 9, 22, 255))
    d.polygon([(470, 1330), (610, 1330), (760, H), (320, H)], fill=(26, 20, 44, 255))
    for gy, gw in [(1408, 3), (1520, 4), (1680, 5), (1860, 6)]:  # path joints
        d.line([540 - gw * 14, gy, 540 + gw * 14, gy], fill=(40, 32, 64, 255), width=gw)
    # gravestones
    for gx, gy, gs in [(120, 1420, 1.0), (880, 1450, 0.8), (760, 1396, 0.6)]:
        d.rounded_rectangle([gx, gy, gx + 74 * gs, gy + 96 * gs], radius=30 * gs, fill=(52, 48, 74, 255))
        d.line([gx + 16 * gs, gy + 48 * gs, gx + 58 * gs, gy + 48 * gs], fill=(30, 27, 46, 255), width=4)
    # fence
    for fx in range(0, W, 90):
        if 430 < fx < 650:
            continue
        d.rectangle([fx, 1306, fx + 12, 1366], fill=(14, 11, 28, 255))
    d.rectangle([0, 1322, W, 1332], fill=(14, 11, 28, 255))
    return img

def pumpkin_sprite(s):
    im = Image.new("RGBA", (int(190 * s), int(160 * s)), (0, 0, 0, 0))
    dd = ImageDraw.Draw(im)
    cx, cy = 95 * s, 84 * s
    for k in (-1, 0, 1):
        dd.ellipse([cx + k * 44 * s - 52 * s, cy - 46 * s, cx + k * 44 * s + 52 * s, cy + 46 * s],
                   fill=(232, 116, 26, 255), outline=(176, 82, 14, 255), width=max(2, int(3 * s)))
    dd.rectangle([cx - 8 * s, cy - 66 * s, cx + 8 * s, cy - 44 * s], fill=(74, 110, 44, 255))
    g = (255, 196, 66, 255)
    dd.polygon([(cx - 30 * s, cy - 12 * s), (cx - 12 * s, cy - 12 * s), (cx - 21 * s, cy + 2 * s)], fill=g)
    dd.polygon([(cx + 12 * s, cy - 12 * s), (cx + 30 * s, cy - 12 * s), (cx + 21 * s, cy + 2 * s)], fill=g)
    dd.polygon([(cx - 32 * s, cy + 18 * s), (cx - 14 * s, cy + 12 * s), (cx - 4 * s, cy + 20 * s),
                (cx + 6 * s, cy + 12 * s), (cx + 16 * s, cy + 20 * s), (cx + 30 * s, cy + 13 * s),
                (cx + 22 * s, cy + 30 * s), (cx - 22 * s, cy + 30 * s)], fill=g)
    return im

def textfx(text, fnt, face_top, face_bot, extrude, depth=8, stroke=(52, 20, 84)):
    """Puffy pseudo-3D lettering, like the reference video's title cards:
    soft drop shadow, a stacked extrusion for thickness, a vertical
    gradient face, a dark rim, and a glossy highlight on the top half."""
    tmp = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
    bbox = tmp.textbbox((0, 0), text, font=fnt, stroke_width=5)
    w = bbox[2] - bbox[0] + 44
    h = bbox[3] - bbox[1] + depth + 40
    x0, y0 = 22 - bbox[0], 10 - bbox[1]
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    sh = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(sh).text((x0, y0 + depth + 8), text, font=fnt,
                            fill=(0, 0, 0, 130), stroke_width=5, stroke_fill=(0, 0, 0, 130))
    layer.alpha_composite(sh.filter(ImageFilter.GaussianBlur(6)))
    d = ImageDraw.Draw(layer)
    for i in range(depth, 0, -1):  # the extruded side of the letters
        d.text((x0, y0 + i), text, font=fnt, fill=extrude + (255,),
               stroke_width=5, stroke_fill=extrude + (255,))
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).text((x0, y0), text, font=fnt, fill=255,
                              stroke_width=5, stroke_fill=255)
    v = np.linspace(0, 1, h)[:, None]
    grad = np.zeros((h, w, 4), np.uint8)
    for c in range(3):
        grad[..., c] = (face_top[c] * (1 - v) + face_bot[c] * v).astype(np.uint8)
    grad[..., 3] = 255
    layer.paste(Image.fromarray(grad, "RGBA"), (0, 0), mask)
    # rim: draw it on its own layer and composite — drawing text with a
    # transparent fill directly onto the layer erases the glyph interior
    # (a PIL RGBA-draw quirk), which wiped the gradient face
    rim = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(rim).text((x0, y0), text, font=fnt, fill=(0, 0, 0, 0),
                             stroke_width=2, stroke_fill=stroke + (255,))
    layer.alpha_composite(rim)
    ma = np.asarray(mask).astype(float) / 255.0
    fall = np.clip(1 - np.linspace(0, 1, h)[:, None] / 0.55, 0, 1)
    gloss = Image.fromarray((ma * fall * 88).astype(np.uint8), "L")
    layer.paste(Image.new("RGBA", (w, h), (255, 255, 255, 255)), (0, 0), gloss)
    # the glyphs' visual centre inside the layer (the padding is asymmetric,
    # so callers that centre the layer itself misplace the text)
    centre = (22 + (bbox[2] - bbox[0]) / 2, 10 + (bbox[3] - bbox[1] + depth) / 2)
    return layer, centre

def moon_sprite():
    """A shaded moon with soft craters and a radial glow (the flat concentric
    rings looked crude). Sprite is 620x620; the disc centre is its centre."""
    S = 620
    yy, xx = np.mgrid[0:S, 0:S]
    r = np.sqrt((xx - S / 2) ** 2 + (yy - S / 2) ** 2)
    glow_a = (95 * np.exp(-((r / 175.0) ** 2))).astype(np.uint8)
    img = Image.fromarray(np.dstack([np.full((S, S), 255, np.uint8),
                                     np.full((S, S), 236, np.uint8),
                                     np.full((S, S), 180, np.uint8), glow_a]), "RGBA")
    R = 100
    disc = Image.new("RGBA", (2 * R, 2 * R), (0, 0, 0, 0))
    g = np.zeros((2 * R, 2 * R, 4), np.uint8)
    v = np.linspace(0, 1, 2 * R)[:, None]
    g[..., 0] = 255 - 14 * v
    g[..., 1] = 246 - 30 * v
    g[..., 2] = 205 - 52 * v
    g[..., 3] = 255
    disc = Image.fromarray(g, "RGBA")
    cd = ImageDraw.Draw(disc, "RGBA")
    for cx, cy, cr, a in [(62, 58, 21, 20), (140, 128, 15, 17), (104, 36, 10, 15),
                          (156, 70, 9, 14), (70, 140, 12, 16), (128, 168, 7, 13)]:
        cd.ellipse([cx - cr, cy - cr, cx + cr, cy + cr], fill=(226, 198, 132, a))
    disc = disc.filter(ImageFilter.GaussianBlur(6))
    # gentle terminator shading toward the lower-left
    sh = Image.new("L", (2 * R, 2 * R), 0)
    shd = ImageDraw.Draw(sh)
    shd.ellipse([-R * 1.15, R * 0.25, R * 0.9, R * 2.3], fill=64)
    sh = sh.filter(ImageFilter.GaussianBlur(28))
    disc = Image.composite(Image.new("RGBA", disc.size, (196, 168, 112, 255)), disc, sh)
    mask = Image.new("L", (2 * R, 2 * R), 0)
    ImageDraw.Draw(mask).ellipse([0, 0, 2 * R - 1, 2 * R - 1], fill=255)
    img.paste(disc, (S // 2 - R, S // 2 - R), mask)
    return img

MOON = moon_sprite()
MOON_C = (896, 178)  # disc centre in frame coords

def title_layer():
    """The two title lines as one transparent layer, pasted per frame with a
    bar-pulse scale (the title is the 'front vocal' of the scene band)."""
    layer = Image.new("RGBA", (W, 340), (0, 0, 0, 0))
    l1, _ = textfx("Trick or Treat!", font_title(94), (255, 205, 105), (255, 128, 36),
                   (96, 40, 130), depth=10)
    l2, _ = textfx("ハロウィンのうた", font_title(54), (255, 246, 224), (255, 214, 150),
                   (96, 40, 130), depth=7)
    layer.alpha_composite(l1, (int(W / 2 - l1.width / 2), 0))
    layer.alpha_composite(l2, (int(W / 2 - l2.width / 2), 148))
    return layer

TITLE = title_layer()

_TELOP_CACHE = {}
def telop_layer(lid):
    """A lyric line as a glossy pill: 3D lettering for the main line
    (kanji text) with small hiragana ruby over the segments that have a
    reading in TELOP_SEGMENTS. Cached per line."""
    if lid in _TELOP_CACHE:
        return _TELOP_CACHE[lid]
    segs = TELOP_SEGMENTS[lid]
    main = "".join(s for s, _ in segs)
    meas = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
    size = 40
    fnt = font_lyric(size)
    tw = int(meas.textlength(main, font=fnt))
    while tw > 944 and size > 26:  # long lines shrink to fit the frame
        size -= 2
        fnt = font_lyric(size)
        tw = int(meas.textlength(main, font=fnt))
    tx, _ = textfx(main, fnt, (255, 250, 235), (255, 226, 170), (70, 30, 105), depth=5)
    arr = np.asarray(tx)
    ys, xs = np.where(arr[..., 3] > 200)
    ink_top, ink_left = int(ys.min()), int(xs.min())
    ink_w = int(xs.max() - xs.min())
    if any(r for _, r in segs):
        pad = 30  # headroom for the ruby row
        comp = Image.new("RGBA", (tx.width, tx.height + pad), (0, 0, 0, 0))
        comp.alpha_composite(tx, (0, pad))
        rf = font_lyric(17)
        od = ImageDraw.Draw(comp)
        x = ink_left
        for s, r in segs:
            sw = meas.textlength(s, font=fnt)
            if r:
                od.text((x + sw / 2, ink_top + pad - 5), r, font=rf, anchor="mb",
                        fill=(255, 246, 224, 255), stroke_width=2,
                        stroke_fill=(40, 16, 66, 255))
            x += sw
        tx = comp
    w, h = ink_w + 120, 118
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    sh = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(sh).rounded_rectangle([10, 16, w - 10, h - 6], radius=30, fill=(0, 0, 0, 120))
    layer.alpha_composite(sh.filter(ImageFilter.GaussianBlur(5)))
    v = np.linspace(0, 1, h)[:, None]
    grad = np.zeros((h, w, 4), np.uint8)
    for c, (a, b) in enumerate(zip((52, 32, 94), (26, 15, 52))):
        grad[..., c] = (a * (1 - v) + b * v).astype(np.uint8)
    grad[..., 3] = 218
    pill = Image.fromarray(grad, "RGBA")
    pmask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(pmask).rounded_rectangle([8, 8, w - 8, h - 14], radius=30, fill=255)
    layer.paste(pill, (0, 0), pmask)
    d = ImageDraw.Draw(layer, "RGBA")
    d.rounded_rectangle([8, 8, w - 8, h - 14], radius=30, outline=(150, 118, 210, 130), width=2)
    d.line([26, 12, w - 26, 12], fill=(255, 255, 255, 60), width=3)
    # centre the ink (alpha>200 excludes the soft shadow), not the layer
    arr = np.asarray(tx)
    ys, xs = np.where(arr[..., 3] > 200)
    icx, icy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    layer.alpha_composite(tx, (int(w / 2 - icx), int((h - 6) / 2 - icy)))
    _TELOP_CACHE[lid] = layer
    return layer

# ================= avatar sprite (prep once) =================
def build_sprite(closed=False):
    S = 780
    av = Image.open(os.path.join(BASE, "assets", "muse_base.jpg")).convert("RGB").resize((S, S))
    # paint out the original smile with a heavily blurred copy of the face
    # itself (keeps the local tones/gradient, so no flat smudge ring remains)
    blurred = av.filter(ImageFilter.GaussianBlur(16))
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).ellipse([322, 210, 458, 270], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(8))
    av = Image.composite(blurred, av, mask).convert("RGBA")
    d = ImageDraw.Draw(av)
    hx, hy = S * 0.50, S * 0.055
    d.ellipse([hx - 205, hy + 58, hx + 205, hy + 108], fill=(38, 24, 74, 255))
    d.polygon([(hx - 128, hy + 82), (hx + 118, hy + 82), (hx + 58, hy - 178), (hx + 6, hy - 196)],
              fill=(48, 30, 96, 255))
    d.polygon([(hx - 118, hy + 58), (hx + 112, hy + 58), (hx + 104, hy + 84), (hx - 110, hy + 84)],
              fill=(255, 140, 40, 255))
    d.rectangle([hx - 26, hy + 56, hx + 26, hy + 86], outline=(255, 214, 90, 255), width=6)
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).ellipse([6, 6, S - 6, S - 6], fill=255)
    av.putalpha(mask)
    if closed:
        # blink variant: cover both eyes with blurred face, draw closed-eye arcs
        blurred = av.filter(ImageFilter.GaussianBlur(14))
        emask = Image.new("L", (S, S), 0)
        ed = ImageDraw.Draw(emask)
        for ex in (327, 449):
            ed.ellipse([ex - 21, 205 - 17, ex + 21, 205 + 17], fill=255)
        emask = emask.filter(ImageFilter.GaussianBlur(5))
        av = Image.composite(blurred, av, emask)
        d2 = ImageDraw.Draw(av)
        for ex in (327, 449):
            d2.arc([ex - 15, 205 - 13, ex + 15, 205 + 11], start=15, end=165,
                   fill=(70, 44, 36, 255), width=4)
    return av  # mouth centre in sprite coords: (390, 234); centre: (390, 390)

def mouth_patch(openness):
    # drawn at 3x and downscaled: at 1x the shapes alias into a blocky
    # mosaic on the captured face's smooth surface.
    SS = 3
    im = Image.new("RGBA", (76 * SS, 56 * SS), (0, 0, 0, 0))
    dd = ImageDraw.Draw(im)
    if openness < 0.10:   # closed: gentle smile arc
        dd.arc([20 * SS, 13 * SS, 56 * SS, 43 * SS], start=15, end=165,
               fill=(70, 44, 36, 255), width=4 * SS)
    else:
        hgt = (6 + 26 * openness) * SS
        dd.ellipse([18 * SS, 28 * SS - hgt / 2, 58 * SS, 28 * SS + hgt / 2], fill=(96, 44, 44, 255))
        dd.ellipse([18 * SS, 28 * SS + hgt * 0.10, 58 * SS, 28 * SS + hgt / 2], fill=(178, 84, 84, 255))
    return im.resize((76, 56), Image.LANCZOS)

# ================= per-frame =================
BG = build_background()
SPRITE = build_sprite()
SPRITE_CLOSED = build_sprite(closed=True)

# ================= captured-avatar frames (n1: real Muse footage) =================
USE_CAPTURE = os.environ.get("RENDER_SOURCE", "capture") != "still"
CAP_SCALE = 780 / 480  # capture is 480px, sprite canvas is 780px
CAP_MOUTH_POS = []  # per capture frame: (mouth_x, mouth_y, scale) in 780px coords

def draw_hat(d, hx, hy, k=1.0):
    d.ellipse([hx - 205 * k, hy + 58 * k, hx + 205 * k, hy + 108 * k], fill=(38, 24, 74, 255))
    d.polygon([(hx - 128 * k, hy + 82 * k), (hx + 118 * k, hy + 82 * k),
               (hx + 58 * k, hy - 178 * k), (hx + 6 * k, hy - 196 * k)], fill=(48, 30, 96, 255))
    d.polygon([(hx - 118 * k, hy + 58 * k), (hx + 112 * k, hy + 58 * k),
               (hx + 104 * k, hy + 84 * k), (hx - 110 * k, hy + 84 * k)], fill=(255, 140, 40, 255))
    d.rectangle([hx - 26 * k, hy + 56 * k, hx + 26 * k, hy + 86 * k],
                outline=(255, 214, 90, 255), width=max(2, int(6 * k)))

def detect_mouth(a):
    """Locate the mouth via the eyes: find the eye pair among the dark blobs,
    predict the mouth below the eye-midpoint, then snap to the nearest wide
    thin blob. Works for both captures (in the witch footage the hat and
    cloak merge into one giant dark blob, so the old 'widest blob' rule
    fails; the eyes stay separate on the face plate)."""
    from scipy import ndimage
    dark = a.sum(axis=2) < 330
    roi = np.zeros_like(dark)
    roi[110:300, 90:390] = True
    lab, _ = ndimage.label(dark & roi)
    blobs = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        h = sl[0].stop - sl[0].start
        w = sl[1].stop - sl[1].start
        if int((lab[sl] == i).sum()) < 30:
            continue
        blobs.append(((sl[1].start + sl[1].stop) / 2,
                      (sl[0].start + sl[0].stop) / 2, w, h))
    eyes = [b for b in blobs if 8 <= b[2] <= 34 and 8 <= b[3] <= 30]
    best = None
    for i in range(len(eyes)):
        for j in range(i + 1, len(eyes)):
            e1, e2 = eyes[i], eyes[j]
            if abs(e1[1] - e2[1]) <= 12:
                d = abs(e2[0] - e1[0])
                if 50 <= d <= 170 and (best is None or d > best[0]):
                    best = (d, e1, e2)
    if best is None:
        return None
    d, e1, e2 = best
    mid = ((e1[0] + e2[0]) / 2, (e1[1] + e2[1]) / 2)
    px, py = mid[0], mid[1] + 0.21 * d
    cands = [b for b in blobs
             if b[2] >= 14 and b[3] <= 14 and b[2] >= 1.8 * b[3]
             and abs(b[0] - px) <= 0.35 * d
             and 0.02 * d <= (b[1] - mid[1]) <= 0.48 * d]
    if cands:
        m = min(cands, key=lambda b: (b[0] - px) ** 2 + (b[1] - py) ** 2)
        return m[0], m[1], m[2] / 37.0
    return px, py, (d * 0.30) / 37.0  # mouth width is ~0.30 * eye distance

def track_mouths(raws):
    """Per-frame mouth position by direct detection. Frames where no eye
    pair can be found (the head is turned away, or mid-blink) return None:
    those frames keep the capture's original face, with no cover and no
    lip-sync patch — a wrong patch on the cheek looks far worse than a
    brief natural smile while the character looks around."""
    return [detect_mouth(a) for a in raws]

def wallpaper_sprite(size=780):
    """The stage window's backdrop: a deep-purple Halloween wallpaper
    (tone-on-tone moons, sparkles, dots and mini pumpkins). Chosen so the
    white ghosts and Muse's cream face both read against it."""
    yy, xx = np.mgrid[0:size, 0:size]
    r = np.sqrt((xx - size / 2) ** 2 + (yy - size / 2) ** 2) / (size * 0.72)
    base = np.zeros((size, size, 3), np.uint8)
    for c, (a, b) in enumerate(zip((108, 86, 162), (58, 44, 100))):
        base[..., c] = (a * (1 - np.clip(r, 0, 1)) + b * np.clip(r, 0, 1)).astype(np.uint8)
    im = Image.fromarray(base, "RGB").convert("RGBA")
    d = ImageDraw.Draw(im, "RGBA")
    rng = random.Random(7)
    for _ in range(46):  # sparkles and dots
        x, y = rng.randint(8, size - 8), rng.randint(8, size - 8)
        if rng.random() < 0.5:
            s = rng.randint(4, 8)
            d.polygon([(x, y - s), (x + 2, y - 2), (x + s, y), (x + 2, y + 2),
                       (x, y + s), (x - 2, y + 2), (x - s, y), (x - 2, y - 2)],
                      fill=(168, 148, 214, 110))
        else:
            d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(168, 148, 214, 90))
    for _ in range(9):  # crescent moons (disc minus an offset disc, via a mask)
        x, y, s = rng.randint(20, size - 40), rng.randint(20, size - 40), rng.randint(11, 17)
        mk = Image.new("L", (3 * s, 3 * s), 0)
        md = ImageDraw.Draw(mk)
        md.ellipse([2, 2, 2 * s - 2, 2 * s - 2], fill=110)
        md.ellipse([int(s * 0.75), 0, int(2.75 * s), int(1.8 * s)], fill=0)
        moonlet = Image.new("RGBA", (3 * s, 3 * s), (186, 168, 228, 255))
        im.paste(moonlet, (x, y), mk)
    for _ in range(7):  # mini pumpkins
        x, y = rng.randint(16, size - 44), rng.randint(16, size - 40)
        d.ellipse([x, y, x + 28, y + 22], fill=(196, 128, 84, 70))
        d.rectangle([x + 12, y - 4, x + 17, y + 2], fill=(120, 160, 90, 80))
    return im

WALLPAPER = wallpaper_sprite()

def key_over_wallpaper(fr_rgb):
    """Cut the captured performer out of its pale studio backdrop and set
    it on the wallpaper. The backdrop is light and neutral; the face is
    warmer (R-B larger), the costume dark - both survive the key. Holes
    are filled so any keyed specks inside the silhouette are restored."""
    from scipy import ndimage
    arr = np.asarray(fr_rgb).astype(int)
    lum = arr.mean(axis=2)
    warm = arr[..., 0] - arr[..., 2]
    bg = (lum > 205) & (warm < 12)
    subj = ndimage.binary_fill_holes(~bg)
    lab, n = ndimage.label(subj)
    if n > 1:  # keep the performer (and anything sizeable attached to the shot)
        sizes = ndimage.sum(subj, lab, range(1, n + 1))
        keep = [i + 1 for i, s in enumerate(sizes) if s > 0.02 * sizes.max()]
        subj = np.isin(lab, keep)
        subj = ndimage.binary_fill_holes(subj)
    mask = Image.fromarray((subj * 255).astype(np.uint8), "L").filter(
        ImageFilter.GaussianBlur(1.5))
    out = WALLPAPER.copy()
    out.paste(fr_rgb.convert("RGBA"), (0, 0), mask)
    return out

def build_capture_frames():
    import glob as _g
    cap_file = os.environ.get("CAPTURE_FILE", os.path.join(BASE, "assets", "muse_witch.mp4"))
    stem = os.path.splitext(os.path.basename(cap_file))[0]
    tmp = os.path.join(BASE, "work", f"capframes_{stem}")
    os.makedirs(tmp, exist_ok=True)
    if not _g.glob(os.path.join(tmp, "f*.png")):
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", cap_file,
                        "-vf", "scale=480:480",
                        os.path.join(tmp, "f%03d.png")], check=True)
    raws = [Image.open(p).convert("RGB") for p in sorted(_g.glob(os.path.join(tmp, "f*.png")))]
    tracked = track_mouths([np.asarray(r).astype(int) for r in raws])
    frames = []
    for raw, tpos in zip(raws, tracked):
        if tpos is not None:
            mx480, my480, sc = tpos
            mx, my = mx480 * CAP_SCALE, my480 * CAP_SCALE
            CAP_MOUTH_POS.append((mx, my, sc))  # 780px coords, as the loop expects
        else:
            CAP_MOUTH_POS.append(None)
        fr = raw.resize((780, 780), Image.LANCZOS)
        if tpos is not None:
            # cover the capture's smile by filling it with the face's own
            # skin tone, sampled from a ring around the mouth (a blurred
            # patch read as a mosaic smudge on the smooth face plate)
            x0, y0 = int(mx - 110 * sc), int(my - 80 * sc)
            x1, y1 = int(mx + 110 * sc), int(my + 80 * sc)
            reg = fr.crop((x0, y0, x1, y1))
            arr = np.asarray(reg).astype(float)
            cy2, cx2 = reg.height / 2, reg.width / 2
            yy2, xx2 = np.mgrid[0:reg.height, 0:reg.width]
            rr = ((xx2 - cx2) / (86 * sc)) ** 2 + ((yy2 - cy2) / (52 * sc)) ** 2
            ring = (rr > 1.15) & (rr < 2.1)
            skin = tuple(int(v) for v in np.median(arr[ring].reshape(-1, 3), axis=0)) \
                if ring.sum() > 50 else (243, 234, 214)
            fill = Image.new("RGB", reg.size, skin)
            m = Image.new("L", reg.size, 0)
            ImageDraw.Draw(m).ellipse([cx2 - 58 * sc, cy2 - 30 * sc,
                                       cx2 + 58 * sc, cy2 + 30 * sc], fill=255)
            m = m.filter(ImageFilter.GaussianBlur(5))
            fr.paste(Image.composite(fill, reg, m), (x0, y0))
        # no drawn hat: the witch costume is part of the captured footage.
        # Key the performer off the studio backdrop onto the wallpaper,
        # then show the whole thing in a rounded-rectangle stage window.
        fr = key_over_wallpaper(fr)
        d = ImageDraw.Draw(fr)
        d.rounded_rectangle([5, 5, 774, 774], radius=64, outline=(58, 32, 96, 255), width=16)
        d.rounded_rectangle([13, 13, 766, 766], radius=56, outline=(255, 160, 50, 255), width=7)
        mask = Image.new("L", (780, 780), 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, 779, 779], radius=64, fill=255)
        fr.putalpha(mask)
        frames.append(fr)
    return frames

CAP_FRAMES = build_capture_frames() if USE_CAPTURE else []
PUMPKINS = [(pumpkin_sprite(1.35), 26, 1494, 0.0), (pumpkin_sprite(1.08), 826, 1537, 1.57),
            (pumpkin_sprite(0.85), 892, 1484, 3.14)]  # bottoms at y<=1710: clear of the telop band
# ================= staging cast (the lyrics, illustrated) =================
def glow_sprite(size=256, rgb=(255, 170, 60)):
    yy, xx = np.mgrid[0:size, 0:size]
    r = np.sqrt((xx - size / 2) ** 2 + (yy - size / 2) ** 2) / (size / 2)
    a = (np.clip(1 - r, 0, 1) ** 2 * 255).astype(np.uint8)
    g = np.zeros((size, size, 4), np.uint8)
    g[..., 0], g[..., 1], g[..., 2], g[..., 3] = rgb[0], rgb[1], rgb[2], a
    return Image.fromarray(g, "RGBA")

GLOW = glow_sprite()

def ghost_sprite():
    im = Image.new("RGBA", (130, 150), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.pieslice([8, 4, 122, 116], start=180, end=360, fill=(246, 246, 255, 238))
    d.rectangle([8, 58, 122, 128], fill=(246, 246, 255, 238))
    for i, gx in enumerate((8, 38, 68, 96)):  # wavy hem
        d.pieslice([gx, 112, gx + 26, 144], start=0, end=180, fill=(246, 246, 255, 238))
    d.ellipse([42, 52, 56, 72], fill=(40, 30, 70, 255))
    d.ellipse([76, 52, 90, 72], fill=(40, 30, 70, 255))
    d.ellipse([58, 80, 72, 94], fill=(40, 30, 70, 255))
    d.ellipse([22, 74, 40, 86], fill=(255, 170, 190, 120))
    d.ellipse([92, 74, 110, 86], fill=(255, 170, 190, 120))
    return im

GHOST = ghost_sprite()

def cat_sprite(eyes_open=True):
    im = Image.new("RGBA", (150, 120), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    body = (18, 14, 30, 255)
    edge = (56, 46, 80, 255)
    d.polygon([(26, 44), (36, 2), (60, 30)], fill=edge)      # ears (rim)
    d.polygon([(124, 44), (114, 2), (90, 30)], fill=edge)
    d.polygon([(28, 42), (36, 4), (58, 30)], fill=body)
    d.polygon([(122, 42), (114, 4), (92, 30)], fill=body)
    d.polygon([(34, 34), (40, 16), (52, 30)], fill=(120, 70, 100, 255))
    d.polygon([(116, 34), (110, 16), (98, 30)], fill=(120, 70, 100, 255))
    d.ellipse([15, 23, 135, 121], fill=edge)                 # head (rim)
    d.ellipse([18, 26, 132, 118], fill=body)
    # mini witch hat, tilted on the left ear - the cat's Halloween
    # costume (user, 2026-10-29: 'give the cat a costume too')
    d.polygon([(8, 28), (72, 28), (47, -8)], fill=(58, 32, 96, 255))
    d.ellipse([2, 24, 78, 35], fill=(44, 24, 74, 255))
    d.polygon([(27, 16), (57, 16), (51, 7), (32, 7)], fill=(255, 140, 40, 255))
    d.rectangle([38, 8, 46, 15], fill=(255, 200, 80, 255))
    if eyes_open:
        d.ellipse([48, 62, 64, 82], fill=(190, 235, 120, 255))
        d.ellipse([88, 62, 104, 82], fill=(190, 235, 120, 255))
        d.ellipse([53, 65, 59, 79], fill=(10, 10, 16, 255))
        d.ellipse([93, 65, 99, 79], fill=(10, 10, 16, 255))
    else:
        d.line([48, 72, 64, 72], fill=(190, 235, 120, 255), width=3)
        d.line([88, 72, 104, 72], fill=(190, 235, 120, 255), width=3)
    d.arc([64, 82, 86, 98], start=20, end=160, fill=(230, 220, 235, 255), width=2)
    # orange bowtie at the neck
    d.polygon([(75, 110), (56, 102), (56, 118)], fill=(255, 140, 40, 255))
    d.polygon([(75, 110), (94, 102), (94, 118)], fill=(255, 140, 40, 255))
    d.ellipse([71, 106, 79, 114], fill=(200, 90, 20, 255))
    return im

CAT_OPEN = cat_sprite(True)
CAT_SHUT = cat_sprite(False)

def candy_house_sprite():
    im = Image.new("RGBA", (330, 220), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([70, 92, 260, 214], radius=10, fill=(166, 102, 54, 255))   # walls
    d.polygon([(46, 100), (284, 100), (228, 30), (102, 30)], fill=(240, 240, 246, 255))  # icing roof
    for i, gx in enumerate(range(58, 272, 24)):  # gumdrops on the roof edge
        d.ellipse([gx, 88, gx + 16, 108], fill=[(236, 92, 120), (120, 210, 140),
                                                (250, 200, 80), (130, 170, 250)][i % 4] + (255,))
    for gx in range(112, 224, 28):  # roof candies
        d.ellipse([gx, 44, gx + 13, 57], fill=(236, 92, 120, 255))
    d.rounded_rectangle([146, 148, 186, 214], radius=8, fill=(112, 62, 32, 255))   # door
    d.ellipse([176, 178, 183, 185], fill=(250, 210, 90, 255))
    for wx, wy in [(90, 118), (214, 118)]:  # warm windows
        d.rounded_rectangle([wx, wy, wx + 28, wy + 30], radius=5, fill=(255, 196, 92, 255))
        d.line([wx + 14, wy, wx + 14, wy + 30], fill=(166, 102, 54, 255), width=3)
    for px in (66, 258):  # candy-cane pillars
        for k in range(6):
            d.rectangle([px, 104 + k * 18, px + 9, 113 + k * 18],
                        fill=(236, 92, 120, 255) if k % 2 == 0 else (246, 246, 250, 255))
    d.line([292, 214, 292, 158], fill=(246, 246, 250, 255), width=5)  # lollipop
    d.ellipse([272, 118, 312, 158], fill=(160, 120, 240, 255))
    d.ellipse([282, 128, 302, 148], fill=(246, 246, 250, 255))
    d.ellipse([288, 134, 296, 142], fill=(236, 92, 120, 255))
    return im

CANDY_HOUSE = candy_house_sprite()

def candy_sprite(rgb):
    im = Image.new("RGBA", (58, 38), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.polygon([(4, 10), (18, 19), (4, 28)], fill=(246, 246, 250, 255))
    d.polygon([(54, 10), (40, 19), (54, 28)], fill=(246, 246, 250, 255))
    d.ellipse([15, 6, 43, 32], fill=rgb + (255,))
    d.ellipse([22, 11, 32, 19], fill=(255, 255, 255, 150))
    return im

CANDIES = [candy_sprite(c).resize((150, 98), Image.LANCZOS)
           for c in [(236, 92, 120), (120, 210, 140), (250, 200, 80),
                     (130, 170, 250), (240, 150, 90)]]  # phone-legible size

def witch_sprite():
    """A little witch flying in to join the party (l4). Deliberately NOT
    a dark silhouette: coloured hat and dress, a visible face and a pale
    halo, so she reads against the night sky and the moon alike."""
    im = Image.new("RGBA", (320, 210), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.line([26, 158, 268, 176], fill=(150, 100, 55, 255), width=9)      # broomstick
    for k in range(7):  # straw brush
        d.line([268, 160 + k * 3, 312, 148 + k * 6], fill=(236, 196, 96, 255), width=4)
    d.polygon([(96, 108), (168, 104), (196, 168), (74, 170)], fill=(122, 70, 176, 255))  # dress
    d.polygon([(96, 108), (168, 104), (150, 128), (112, 130)], fill=(238, 138, 62, 255))  # sash
    d.ellipse([104, 52, 164, 112], fill=(252, 240, 220, 255))           # face
    d.ellipse([120, 78, 128, 88], fill=(40, 30, 60, 255))               # eyes
    d.ellipse([142, 78, 150, 88], fill=(40, 30, 60, 255))
    d.arc([128, 86, 144, 100], start=15, end=165, fill=(40, 30, 60, 255), width=2)
    d.ellipse([108, 90, 120, 98], fill=(255, 168, 178, 170))            # cheeks
    d.ellipse([148, 90, 160, 98], fill=(255, 168, 178, 170))
    d.polygon([(100, 66), (112, 96), (100, 104)], fill=(238, 138, 62, 255))  # hair
    d.polygon([(168, 66), (156, 96), (168, 104)], fill=(238, 138, 62, 255))
    d.ellipse([72, 52, 196, 74], fill=(88, 52, 142, 255))               # hat brim
    d.polygon([(92, 56), (172, 56), (140, 4), (122, 2)], fill=(88, 52, 142, 255))  # hat cone
    d.rectangle([104, 46, 160, 58], fill=(245, 152, 62, 255))           # hat band
    d.rectangle([126, 47, 140, 57], outline=(255, 214, 92, 255), width=3)  # buckle
    return im

WITCH = witch_sprite().transpose(Image.FLIP_LEFT_RIGHT)  # brush trails behind
WITCH_GLOW = glow_sprite(256, (206, 196, 255))

def _lerp(a, b, k):
    return a + (b - a) * k

def _smooth(a, b, x):
    k = min(1.0, max(0.0, (x - a) / (b - a)))
    return k * k * (3 - 2 * k)

ENVELOPE = np.load(os.path.join(BASE, "work", "envelope.npy"))
TELOP_SEGMENTS = {
    "l0": [("ハッピーハロウィン! Museだよ。", None), ("今夜", "こんや"), ("は", None),
           ("歌", "うた"), ("っちゃうぞ!", None)],
    "l1": [("トリック・オア・トリート、お", None), ("菓子", "かし"), ("を ちょうだい", None)],
    "l2": [("くれなきゃ イタズラ、しちゃうぞ ほら", None)],
    "l3": [("カボチャランタン、ゆらゆら ", None), ("揺", "ゆ"), ("れて", None)],
    "l4": [("お", None), ("化", "ば"), ("けも ", None), ("魔女", "まじょ"), ("も、", None),
           ("集", "あつ"), ("まる", None), ("夜", "よる")],
    "l5": [("ハロウィン ハロウィン、", None), ("月", "つき"), ("の", None), ("夜", "よ"),
           ("のパーティー", None)],
    "l6": [("Trick or Treat", "トリック・オア・トリート"), ("、", None), ("皆", "みんな"),
           ("で", None), ("歌", "うた"), ("おう", None)],
    "l7": [("真っ暗", "まっくら"), (" ", None), ("闇夜", "やみよ"), ("を ", None),
           ("灯", "とも"), ("す ランタン", None)],
    "l8": [("蝙蝠", "こうもり"), (" パタパタ ", None), ("空", "そら"), ("を ", None),
           ("回", "まわ"), ("る", None)],
    "l9": [("黒猫", "くろねこ"), (" ひょっこり ", None), ("顔", "かお"), ("を ", None),
           ("出", "だ"), ("して", None)],
    "l10": [("皆", "みんな"), (" ", None), ("集", "あつ"), ("まれ お", None), ("菓子", "かし"),
            ("の ", None), ("家", "いえ"), ("へ", None)],
    "l11": [("ハロウィン ハロウィン、", None), ("踊", "おど"), ("ろう、", None), ("朝", "あさ"),
            ("まで", None)],
    "l12": [("Trick or Treat", "トリック・オア・トリート"), ("、", None), ("笑顔", "えがお"),
            ("で ", None), ("歌", "うた"), ("おう", None)],
    "l13": [("ハッピーハロウィン! また ", None), ("来年", "らいねん"), ("!", None)],
}
_TL = json.load(open(os.path.join(BASE, "work", "timeline.json")))
SLOTS = {e["id"]: e["start"] for e in _TL}
DURS = {e["id"]: e["end"] - e["start"] for e in _TL}

def current_line(t):
    for lid, start in SLOTS.items():
        if start <= t <= start + DURS[lid] + 0.25:
            return lid
    return None

FRAMES = int(SECONDS * FPS)
START_S = float(os.environ.get("RENDER_START", "0"))
START_F = int(START_S * FPS)
FDIR = os.environ.get("RENDER_FDIR", os.path.join(BASE, "work", "frames"))
os.makedirs(FDIR, exist_ok=True)
S0 = 780
CX, CY = 540, 988  # avatar centre in frame coords (raised so the dancing
                    # window's bottom corners clear the pumpkins)

BLACK_IMG = Image.new("RGB", (W, H), (0, 0, 0))

for f in range(FRAMES):
    if os.path.exists(os.path.join(FDIR, f"f{f:04d}.png")):
        continue  # resume support: keep frames from an interrupted run
    t = START_S + f / FPS
    img = BG.copy()
    d = ImageDraw.Draw(img, "RGBA")
    # twinkling stars
    rng = random.Random(1000 + f)
    for _ in range(14):
        x, y = rng.randint(0, W), rng.randint(0, 1100)
        a = rng.randint(60, 200)
        d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(255, 252, 230, a))
    # the moon is the pad: a slow glow swell over 4 bars
    swell = 1 + 0.02 * math.sin(2 * math.pi * t / (BAR * 4))
    mw = int(620 * swell)
    moon = MOON.resize((mw, mw), Image.LANCZOS)
    img.paste(moon, (int(MOON_C[0] - mw / 2), int(MOON_C[1] - mw / 2)), moon)
    # l4: a little witch flies across the moon, coming to join the party
    wp = (t - (SLOTS["l4"] + 0.3)) / 3.3
    if 0 <= wp <= 1:
        wx = -330 + wp * (W + 660)
        wy = 158 - math.sin(wp * math.pi) * 42 + math.sin(t * 6) * 4
        halo = WITCH_GLOW.resize((400, 400), Image.LANCZOS)
        halo.putalpha(halo.getchannel("A").point(lambda v: int(v * 0.38)))
        img.paste(halo, (int(wx + 160 - 200), int(wy + 105 - 200)), halo)
        img.paste(WITCH, (int(wx), int(wy)), WITCH)
    # l7: the night deepens; only the lanterns and windows stay lit
    dim_k = 0.30 * _smooth(SLOTS["l7"], SLOTS["l7"] + 1.0, t) * (1 - _smooth(SLOTS["l5"] - 1.0, SLOTS["l5"], t))
    if dim_k > 0.003:
        img = Image.blend(img, BLACK_IMG, dim_k)
        d = ImageDraw.Draw(img, "RGBA")
        for wxc, wyc in [(109, 1151), (249, 1151)]:  # the haunted house windows glow
            g = GLOW.resize((120, 120), Image.LANCZOS)
            g.putalpha(g.getchannel("A").point(lambda v: int(v * dim_k / 0.30 * 0.85)))
            img.paste(g, (wxc - 60, wyc - 60), g)
            d.rounded_rectangle([wxc - 17, wyc - 23, wxc + 17, wyc + 23], radius=4,
                                fill=(255, 205, 110, int(110 * dim_k / 0.30)))
    # l9: the black cat peeks out from behind the right-hand pumpkins
    # (drawn before them, so they hide it while it rises). It keeps
    # living after the entrance: a breathing bob, a slow head tilt and
    # a repeated peek - duck down and pop back every ~6 s (user,
    # 2026-10-29: 'the cat should pop in and out like the ghosts move')
    raw = min(1.0, max(0.0, (t - SLOTS["l9"]) / 0.8))
    if raw > 0:
        eob = 1 + 2.70158 * (raw - 1) ** 3 + 1.70158 * (raw - 1) ** 2  # easeOutBack
        # peek cycle: a quick duck, a hidden beat, then a pop back up
        # with overshoot - a peek ("hyokko!"), not a smooth squat
        ph3 = (t - SLOTS["l9"]) % 6.3
        dip = 0.0
        if ph3 > 5.15:
            u = ph3 - 5.15
            if u < 0.16:
                dip = (u / 0.16) * 0.88
            elif u < 0.62:
                dip = 0.88
            elif u < 1.0:
                v = (u - 0.62) / 0.38
                dip = 0.88 * (1 - (1 + 2.70158 * (v - 1) ** 3 + 1.70158 * (v - 1) ** 2))
        dip = min(0.92, max(-0.08, dip))
        cat = CAT_SHUT if (t % 3.7) < 0.11 else CAT_OPEN
        tilt = math.sin(t * 0.9) * 3.0 * raw
        if abs(tilt) > 0.2:
            cat = cat.rotate(tilt, resample=Image.BICUBIC, expand=True)
        ccy = 1562 - 144 * eob * (1 - dip) + math.sin(t * 1.8) * 6 * raw
        # warm halo behind the cat so it separates from the dark
        halo = GLOW.resize((300, 300), Image.LANCZOS)
        halo.putalpha(halo.getchannel("A").point(lambda v: int(v * 0.30 * raw)))
        img.paste(halo, (972 - 150, int(ccy - 165)), halo)
        img.paste(cat, (int(972 - cat.width / 2), int(ccy - cat.height / 2)), cat)
    # pumpkins are the rhythm section: they hop on the beat (kick), the
    # right-hand ones answer a half-beat later, with a landing squash.
    # From l3 they are lit lanterns and sway like pendulums.
    lan = _smooth(SLOTS["l3"], SLOTS["l3"] + 0.8, t)
    for spr0, px_, py_, ph in PUMPKINS:
        spr = spr0
        if lan > 0.01:
            flick = lan * (0.80 + 0.20 * math.sin(t * 7.3 + ph * 3.0)) * (1 + 1.1 * dim_k)
            gs = int(spr.width * 2.3)
            g = GLOW.resize((gs, gs), Image.LANCZOS)
            g.putalpha(g.getchannel("A").point(lambda v: int(v * min(1.0, flick) * 0.75)))
            img.paste(g, (int(px_ + spr.width / 2 - gs / 2), int(py_ + spr.height * 0.42 - gs / 2)), g)
            ang = lan * math.sin(t * 1.5 + ph) * 4.5
            spr = spr0.rotate(ang, resample=Image.BICUBIC, expand=True,
                              center=(spr0.width / 2, spr0.height - 6))
        hop = abs(math.sin(math.pi * t / BEAT + ph))
        off = -hop * 17
        sq = 1 - 0.10 * (1 - hop)
        ps = spr.resize((int(spr.width / sq), int(spr.height * sq)), Image.LANCZOS)
        img.paste(ps, (px_ - (ps.width - spr.width) // 2, int(py_ + off + spr.height - ps.height)), ps)
    # ---- dancing stage window (the footage performs; the code dances) ----
    bounce = -abs(math.sin(math.pi * t)) * 30
    sway = math.sin(2 * math.pi * t / 4) * 34
    tilt = math.sin(2 * math.pi * t / 2) * 5.0 + math.sin(2 * math.pi * t / 2 + 1.3) * 1.2
    if 37.9 <= t < 39.9:  # chorus flourish: a happy rock, not a forced spin
        u = (t - 37.9) / 2.0
        tilt += 8 * math.sin(2 * math.pi * u * 2) * (1 - 0.4 * abs(2 * u - 1))
        bounce *= 1.7
    if SLOTS["l2"] <= t < SLOTS["l2"] + 2.0:  # l2 mischief: a playful extra tilt
        u2 = t - SLOTS["l2"]
        tilt += 2.6 * math.sin(2 * math.pi * u2 / 0.94) * _smooth(0.0, 0.3, u2) * (1 - _smooth(1.7, 2.0, u2))
    squash = abs(math.sin(math.pi * t)) ** 1.6  # snappier landings
    breath = math.sin(2 * math.pi * t / 2.9)    # slow breathing drift
    pulse = abs(math.sin(math.pi * t))
    sx = 1 + 0.045 * squash - 0.003 * breath
    sy = 1 - 0.09 * squash + 0.006 * breath
    win = S0 * (1 + 0.02 * pulse)
    if USE_CAPTURE:
        half = (len(CAP_FRAMES) - 1) / 24.0
        cyc = 2 * half
        ct = t % cyc
        ci = int(ct * 24) if ct <= half else int((cyc - ct) * 24)
        ci = max(0, min(len(CAP_FRAMES) - 1, ci))
        base = CAP_FRAMES[ci]
    else:
        base = SPRITE_CLOSED if (t % 3.7) < 0.11 else SPRITE
    spr = base.resize((int(win * sx), int(win * sy)), Image.LANCZOS)
    # bake the mouth into the sprite BEFORE rotation, so mouth and body
    # always rotate together (fixes the counter-rotation bug)
    o = float(ENVELOPE[min(START_F + f, len(ENVELOPE) - 1)])
    mpos = CAP_MOUTH_POS[ci] if USE_CAPTURE else (390, 234, 1.0)
    if mpos is not None:  # None = head turned/blinking in the footage: leave its face alone
        mx, my, msc = mpos
        mp = mouth_patch(o)
        if USE_CAPTURE:
            k = CAP_SCALE * msc
            mp = mp.resize((int(mp.width * k), int(mp.height * k)), Image.LANCZOS)
        spr.paste(mp, (int(mx * (win / S0) * sx - mp.width / 2), int(my * (win / S0) * sy - mp.height / 2)), mp)
    spr = spr.rotate(tilt, resample=Image.BICUBIC, expand=True)
    cx, cy = CX + sway, CY + bounce + breath * 4
    img.paste(spr, (int(cx - spr.width / 2), int(cy - spr.height / 2)), spr)
    d = ImageDraw.Draw(img, "RGBA")
    # l10: the candy house appears at the foot of the stage
    hp = _smooth(SLOTS["l10"], SLOTS["l10"] + 1.2, t)
    if hp > 0:
        house = CANDY_HOUSE
        if hp < 1:
            house = CANDY_HOUSE.copy()
            house.putalpha(house.getchannel("A").point(lambda v: int(v * hp)))
        img.paste(house, (540 - 165, 1692 - 220 + int((1 - hp) * 46)), house)
    # l4 onward: ghosts float in and stay; they drift to the candy house
    # at l10 and slowly circle the stage through the chorus
    GHOST_HOVER = [(150, 815), (905, 765), (830, 520)]
    GHOST_GATHER = [(320, 1215), (700, 1185), (520, 1125)]
    for gi in range(3):
        ep = _smooth(SLOTS["l4"] + gi * 0.35, SLOTS["l4"] + 1.4 + gi * 0.35, t)
        if ep <= 0:
            continue
        hx, hy = GHOST_HOVER[gi]
        startx = -150 if gi != 1 else W + 20
        gx = _lerp(startx, hx, ep)
        gy = _lerp(hy + 50, hy, ep)
        gk = _smooth(SLOTS["l10"], SLOTS["l10"] + 1.45, t)
        gx, gy = _lerp(gx, GHOST_GATHER[gi][0], gk), _lerp(gy, GHOST_GATHER[gi][1], gk)
        ok_ = _smooth(SLOTS["l5"], SLOTS["l5"] + 1.0, t) * (1 - _smooth(52.5, 53.5, t))
        if ok_ > 0:
            ang = t * 0.55 + gi * 2.094
            gx = _lerp(gx, 540 + math.cos(ang) * 465, ok_)
            gy = _lerp(gy, 1060 + math.sin(ang) * 320, ok_)
        gx += math.sin(t * 1.3 + gi * 2.1) * 7
        gy += math.sin(t * 2.0 + gi * 2.1) * 11
        img.paste(GHOST, (int(gx - 65), int(gy - 75)), GHOST)
    # candies drift around the stage at l1 and through the chorus
    ca = max(_smooth(SLOTS["l1"], SLOTS["l1"] + 0.6, t) * (1 - _smooth(SLOTS["l2"] - 0.6, SLOTS["l2"], t)),
             _smooth(SLOTS["l5"], SLOTS["l5"] + 0.6, t) * (1 - _smooth(52.9, 53.5, t)))
    if ca > 0.01:
        for ci2, cand in enumerate(CANDIES):
            bx2, by2 = [(235, 755), (845, 745), (175, 1150), (905, 1110), (330, 1580)][ci2]
            c2 = cand.rotate(math.sin(t * 0.9 + ci2 * 1.7) * 14, resample=Image.BICUBIC, expand=True)
            if ca < 1:
                c2.putalpha(c2.getchannel("A").point(lambda v: int(v * ca)))
            img.paste(c2, (int(bx2 + math.sin(t * 1.1 + ci2 * 2.3) * 10 - c2.width / 2),
                           int(by2 + math.sin(t * 1.7 + ci2 * 1.3) * 13 - c2.height / 2)), c2)
    # fog bands drifting
    for i, (fy, sp, ph) in enumerate([(520, 14, 0.0), (1480, 9, 2.4)]):
        fx = ((t * sp + ph * 160) % (W + 700)) - 350
        d.ellipse([fx - 420, fy, fx + 420, fy + 90], fill=(220, 220, 255, 16))
    # bats are the hi-hat: wings flap on 8th notes, bodies drift on sine
    # paths; drawn larger and in a lighter violet so they read against the sky
    for bi, (sp, by, ph) in enumerate([(46, 118, 0.0), (34, 238, 300.0), (58, 352, 640.0)]):
        bx = ((t * sp + ph) % (W + 300)) - 150
        byy = by + math.sin(t * 2.2 + bi * 2.0) * 16
        ow = _smooth(SLOTS["l8"], SLOTS["l8"] + 0.6, t) * (1 - _smooth(SLOTS["l9"] - 0.6, SLOTS["l9"], t))
        if ow > 0:  # l8: the bats circle the moon, the tree, the stage
            (ocx, ocy), (orx, ory) = [((896, 178), (255, 95)), ((375, 305), (215, 85)),
                                      ((540, 570), (335, 105))][bi]
            oa = 2 * math.pi * (t - SLOTS["l8"]) / DURS["l8"] + bi * 2.094
            bx = _lerp(bx, ocx + math.cos(oa) * orx, ow)
            byy = _lerp(byy, ocy + math.sin(oa) * ory, ow)
        flap = math.sin(2 * math.pi * t / (BEAT / 2) + bi * 1.7) * 12
        s = 2.1
        pts = [(bx - 46 * s, byy - flap), (bx - 26 * s, byy - 18 * s - flap * 0.4), (bx - 10 * s, byy - 4 * s),
               (bx, byy - 16 * s), (bx + 10 * s, byy - 4 * s), (bx + 26 * s, byy - 18 * s - flap * 0.4),
               (bx + 46 * s, byy - flap), (bx + 30 * s, byy + 4 * s), (bx + 15 * s, byy - 2 * s),
               (bx, byy + 12 * s), (bx - 15 * s, byy - 2 * s), (bx - 30 * s, byy + 4 * s)]
        d.polygon(pts, fill=(64, 50, 110, 255))
    # outro: dawn breaks (they danced until morning) - a warm wash that
    # rises from the bottom of the frame
    dk = _smooth(51.5, 59.0, t)
    if dk > 0:
        arr = np.asarray(img).astype(np.float32)
        rows = (np.linspace(0.18, 1.0, H)[:, None, None] * dk * 0.55).astype(np.float32)
        arr = arr * (1 - rows) + np.array([255, 168, 118], np.float32) * rows
        img = Image.fromarray(arr.astype(np.uint8))
        # keep the jack-o'-lanterns lit through the dawn: re-assert
        # their glow over the wash (user, 2026-10-29 - 'the lanterns
        # at the bottom get buried and I could not spot them')
        lan = _smooth(SLOTS["l3"], SLOTS["l3"] + 0.8, t)
        for spr0, px_, py_, ph in PUMPKINS:
            flick = lan * (0.80 + 0.20 * math.sin(t * 7.3 + ph * 3.0)) * dk
            gs = int(spr0.width * 2.3)
            g = GLOW.resize((gs, gs), Image.LANCZOS)
            g.putalpha(g.getchannel("A").point(lambda v: int(v * min(1.0, flick) * 0.7)))
            img.paste(g, (int(px_ + spr0.width / 2 - gs / 2),
                          int(py_ + spr0.height * 0.42 - gs / 2)), g)
        # the gravestones wash out to pale blobs at dawn and stop
        # reading as stones (user, 2026-10-29) - redraw them dark
        dg = ImageDraw.Draw(img)
        for gx, gy, gs2 in [(120, 1420, 1.0), (880, 1450, 0.8), (760, 1396, 0.6)]:
            dg.rounded_rectangle([gx, gy, gx + 74 * gs2, gy + 96 * gs2],
                                 radius=30 * gs2, fill=(52, 48, 74))
            dg.line([gx + 16 * gs2, gy + 48 * gs2, gx + 58 * gs2, gy + 48 * gs2],
                    fill=(30, 27, 46), width=4)
    # title is the front vocal: a scale pulse on each bar start + a small
    # bounce every 2 beats
    bar_phase = (t % BAR) / BAR
    tpulse = 1 + 0.05 * max(0.0, 1 - (t % BAR) / 0.30) + 0.008 * abs(math.sin(math.pi * t / (BEAT * 2)))
    tw2, th2 = int(W * tpulse), int(340 * tpulse)
    tlay = TITLE.resize((tw2, th2), Image.LANCZOS)
    img.paste(tlay, (int(W / 2 - tw2 / 2), 322), tlay)  # below the moon/bat zone
    # lyric telop: a glossy pill with 3D lettering
    line = current_line(t)
    if line:
        tl = telop_layer(line)
        img.paste(tl, (int(W / 2 - tl.width / 2), 1768), tl)
    # finale fade-out, 58.6-60.0s (matches the audio afade in the master)
    if t >= 58.6:
        fk = min(1.0, (t - 58.6) / 1.4)
        if fk > 0:
            img = Image.blend(img, BLACK_IMG, fk)
    img.save(os.path.join(FDIR, f"f{f:04d}.png"))
    if f % 90 == 0:
        print("frame", f, "/", FRAMES, flush=True)

subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", str(FPS),
                "-i", os.path.join(FDIR, "f%04d.png"), "-i", AUDIO,
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                "-c:a", "aac", "-shortest", OUT], check=True)
print("wrote", OUT)
