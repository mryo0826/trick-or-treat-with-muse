# Test frame for the Halloween video (1080x1920, vertical).
# Night scene + Muse avatar (circle-masked) wearing a drawn witch hat,
# with an open mouth overlay to preview the amplitude lip-sync idea.
import os, math, random
from PIL import Image, ImageDraw, ImageFilter, ImageFont

BASE = os.path.join(os.path.dirname(__file__), "..")
W, H = 1080, 1920
random.seed(31)

def font(size, bold=True):
    cands = ["/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
             "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]
    for p in cands:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()

# ---- background: vertical gradient night sky --------------------------------
img = Image.new("RGB", (W, H))
px = img.load()
top, mid, bot = (16, 12, 48), (43, 24, 84), (24, 14, 40)
for y in range(H):
    t = y / H
    if t < 0.6:
        k = t / 0.6
        c = tuple(int(top[i] + (mid[i] - top[i]) * k) for i in range(3))
    else:
        k = (t - 0.6) / 0.4
        c = tuple(int(mid[i] + (bot[i] - mid[i]) * k) for i in range(3))
    for x in range(0, W, 4):
        for dx in range(4):
            px[x + dx, y] = c
d = ImageDraw.Draw(img, "RGBA")

# stars
for _ in range(170):
    x, y = random.randint(0, W), random.randint(0, 1250)
    r = random.choice([1, 1, 2, 2, 3])
    a = random.randint(110, 255)
    d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 250, 220, a))

# moon with glow (top right)
mx, my, mr = 838, 268, 118
glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
gd = ImageDraw.Draw(glow)
gd.ellipse([mx - mr * 2.1, my - mr * 2.1, mx + mr * 2.1, my + mr * 2.1],
           fill=(255, 236, 170, 60))
img.paste(Image.alpha_composite(img.convert("RGBA"), glow).convert("RGB"), (0, 0))
d = ImageDraw.Draw(img, "RGBA")
d.ellipse([mx - mr, my - mr, mx + mr, my + mr], fill=(252, 240, 190, 255))
for cx, cy, cr in [(800, 230, 20), (880, 300, 14), (845, 205, 10)]:
    d.ellipse([cx - cr, cy - cr, cx + cr, cy + cr], fill=(238, 222, 165, 255))

# bats (simple two-arc silhouettes)
def bat(cx, cy, s):
    pts = [(cx - 46 * s, cy), (cx - 26 * s, cy - 18 * s), (cx - 10 * s, cy - 4 * s),
           (cx, cy - 16 * s), (cx + 10 * s, cy - 4 * s), (cx + 26 * s, cy - 18 * s),
           (cx + 46 * s, cy), (cx + 24 * s, cy + 6 * s), (cx, cy + 12 * s),
           (cx - 24 * s, cy + 6 * s)]
    d.polygon(pts, fill=(10, 8, 24, 255))
bat(210, 300, 1.15); bat(330, 190, 0.8); bat(140, 470, 0.65)

# ground
d.rectangle([0, 1560, W, H], fill=(13, 10, 26, 255))
d.ellipse([-200, 1500, W + 200, 1660], fill=(13, 10, 26, 255))

# pumpkins
def pumpkin(cx, cy, s):
    for k in (-1, 0, 1):
        d.ellipse([cx + k * 44 * s - 52 * s, cy - 46 * s, cx + k * 44 * s + 52 * s, cy + 46 * s],
                  fill=(232, 116, 26, 255), outline=(176, 82, 14, 255), width=max(2, int(3 * s)))
    d.rectangle([cx - 8 * s, cy - 66 * s, cx + 8 * s, cy - 44 * s], fill=(74, 110, 44, 255))
    glowc = (255, 196, 66, 255)
    d.polygon([(cx - 30 * s, cy - 12 * s), (cx - 12 * s, cy - 12 * s), (cx - 21 * s, cy + 2 * s)], fill=glowc)
    d.polygon([(cx + 12 * s, cy - 12 * s), (cx + 30 * s, cy - 12 * s), (cx + 21 * s, cy + 2 * s)], fill=glowc)
    d.polygon([(cx - 32 * s, cy + 18 * s), (cx - 14 * s, cy + 12 * s), (cx - 4 * s, cy + 20 * s),
               (cx + 6 * s, cy + 12 * s), (cx + 16 * s, cy + 20 * s), (cx + 30 * s, cy + 13 * s),
               (cx + 22 * s, cy + 30 * s), (cx - 22 * s, cy + 30 * s)], fill=glowc)
pumpkin(168, 1700, 1.0); pumpkin(938, 1745, 0.8); pumpkin(858, 1648, 0.55)

# ---- Muse avatar in a circle mask ------------------------------------------
av = Image.open(os.path.join(BASE, "assets", "muse_base.jpg")).convert("RGB")
S = 780
av = av.resize((S, S))
mask = Image.new("L", (S, S), 0)
md = ImageDraw.Draw(mask)
md.ellipse([6, 6, S - 6, S - 6], fill=255)
ax, ay = (W - S) // 2, 800                     # paste position
ring = Image.new("RGBA", (W, H), (0, 0, 0, 0))
rd = ImageDraw.Draw(ring)
rd.ellipse([ax - 10, ay - 10, ax + S + 10, ay + S + 10], outline=(255, 176, 66, 255), width=10)
img.paste(av, (ax, ay), mask)
img.paste(ring, (0, 0), ring)
d = ImageDraw.Draw(img, "RGBA")

# witch hat (drawn) - positioned over the head (head top ~5.5% of avatar img)
hx, hy = ax + S * 0.50, ay + S * 0.055
d.ellipse([hx - 205, hy + 58, hx + 205, hy + 108], fill=(38, 24, 74, 255))           # brim
d.polygon([(hx - 128, hy + 82), (hx + 118, hy + 82), (hx + 58, hy - 178), (hx + 6, hy - 196)],
          fill=(48, 30, 96, 255))                                                     # cone
d.polygon([(hx - 118, hy + 58), (hx + 112, hy + 58), (hx + 104, hy + 84), (hx - 110, hy + 84)],
          fill=(255, 140, 40, 255))                                                   # band
d.rectangle([hx - 26, hy + 56, hx + 26, hy + 86], outline=(255, 214, 90, 255), width=6)  # buckle

# open mouth overlay (lip-sync preview) - mouth ~ (50%, 30.5%) of avatar img
mox, moy = ax + S * 0.50, ay + S * 0.305
d.ellipse([mox - 30, moy - 6, mox + 30, moy + 40], fill=(96, 44, 44, 255))
d.ellipse([mox - 30, moy + 12, mox + 30, moy + 40], fill=(178, 84, 84, 255))

# ---- titles ------------------------------------------------------------------
f1 = font(96); f2 = font(54)
d.text((W / 2, 120), "Trick or Treat!", font=f1, anchor="ma", fill=(255, 170, 60, 255),
       stroke_width=3, stroke_fill=(60, 20, 90, 255))
d.text((W / 2, 244), "ハロウィンのうた", font=f2, anchor="ma", fill=(255, 240, 210, 255))
d.text((W / 2, 1812), "Muse が歌います♪", font=font(46), anchor="ma", fill=(230, 220, 255, 255))

out = os.path.join(BASE, "out", "test_frame.png")
img.save(out)
print("wrote", out)
