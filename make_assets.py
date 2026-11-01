"""
Генератор картинок для игры «Горный байкер».

Рисует все PNG-спрайты (байк, гонщик, горы, земля, деревья, камни, флаги)
со светом, тенями и фактурой и кладёт их в папку assets/.
Запускается автоматически, если папки assets нет. Любую картинку можно
потом открыть и поправить в Krita / GIMP / Photoshop — игра подхватит её.

Нужны: pip install numpy pillow
"""
import json
import math
import os
import random

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "assets")
VERSION = "6"
HS = 3                      # во сколько раз детали байка/гонщика больше игрового размера

# numpy нужен только для (пере)генерации картинок, в рантайме импортировать его необязательно
np = None
_L = None
_Hh = None


def _np():
    """Лениво импортирует numpy и подготавливает векторы света. Вызывается в generate_all()."""
    global np, _L, _Hh
    if np is None:
        import numpy
        np = numpy
        _L = np.array([0.50, -0.70, 0.50], np.float32)
        _L /= np.linalg.norm(_L)
        _Hh = _L + np.array([0, 0, 1], np.float32)
        _Hh /= np.linalg.norm(_Hh)
    return np


# ======================= шум =======================
def value_noise(h, w, cy, cx, rng):
    g = rng.random((cy, cx)).astype(np.float32)
    ys = np.arange(h, dtype=np.float32) * cy / h
    xs = np.arange(w, dtype=np.float32) * cx / w
    iy = ys.astype(int)
    ix = xs.astype(int)
    fy = ys - iy
    fx = xs - ix
    fy = fy * fy * (3 - 2 * fy)
    fx = fx * fx * (3 - 2 * fx)
    iy1 = (iy + 1) % cy
    ix1 = (ix + 1) % cx
    iy = iy % cy
    ix = ix % cx
    a = g[iy][:, ix]
    b = g[iy][:, ix1]
    c = g[iy1][:, ix]
    d = g[iy1][:, ix1]
    top = a + (b - a) * fx[None, :]
    bot = c + (d - c) * fx[None, :]
    return top + (bot - top) * fy[:, None]


def fbm(h, w, cells, octaves, rng, persistence=0.5):
    cy, cx = cells
    total = np.zeros((h, w), np.float32)
    amp, norm = 1.0, 0.0
    for o in range(octaves):
        total += amp * value_noise(h, w, cy * 2 ** o, cx * 2 ** o, rng)
        norm += amp
        amp *= persistence
    return total / norm


def noise1d(w, cells, rng):
    return value_noise(1, w, 1, cells, rng)[0]


def circ_smooth(a, k):
    half = k // 2
    return sum(np.roll(a, i) for i in range(-half, half + 1)) / (2 * half + 1)


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def lerp(a, b, t):
    return a + (b - a) * t


# ======================= холст с освещением =======================
class Canvas:
    """Рисует закруглённые детали по полям расстояний и красит их с учётом света."""

    def __init__(self, w, h, ss=6):
        self.w, self.h, self.ss = w, h, ss
        self.W, self.H = w * ss, h * ss
        ys, xs = np.mgrid[0:self.H, 0:self.W].astype(np.float32)
        self.px = (xs + 0.5) / ss
        self.py = (ys + 0.5) / ss
        self.img = np.zeros((self.H, self.W, 4), np.float32)

    # --- фигуры: возвращают (расстояние до края, локальный радиус) ---
    def capsule(self, ax, ay, bx, by, ra, rb=None):
        rb = ra if rb is None else rb
        pax, pay = self.px - ax, self.py - ay
        bax, bay = bx - ax, by - ay
        h = np.clip((pax * bax + pay * bay) / (bax * bax + bay * bay + 1e-9), 0, 1)
        d = np.hypot(pax - bax * h, pay - bay * h)
        r = ra + (rb - ra) * h
        return d - r, r

    def circle(self, cx, cy, R):
        d = np.hypot(self.px - cx, self.py - cy)
        return d - R, np.full_like(d, R)

    def ring(self, cx, cy, R, w):
        d = np.abs(np.hypot(self.px - cx, self.py - cy) - R)
        return d - w, np.full_like(d, w)

    def rbox(self, cx, cy, hw, hh, rad, ang=0.0):
        dx, dy = self.px - cx, self.py - cy
        c, s = math.cos(ang), math.sin(ang)
        xr = dx * c + dy * s
        yr = -dx * s + dy * c
        qx = np.abs(xr) - (hw - rad)
        qy = np.abs(yr) - (hh - rad)
        d = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - rad
        return d, np.full_like(d, min(hw, hh))

    @staticmethod
    def union(*shapes):
        d, r = shapes[0]
        for d2, r2 in shapes[1:]:
            m = d2 < d
            d = np.where(m, d2, d)
            r = np.where(m, r2, r)
        return d, r

    # --- покраска ---
    def add(self, shape, base, ambient=0.42, spec=0.25, shin=26.0, alpha=1.0, flat=False):
        d, r = shape
        inside = (d < 0).astype(np.float32)
        if not inside.any():
            return
        base = np.asarray(base, np.float32)
        if base.ndim == 1:
            base = np.broadcast_to(base, (self.H, self.W, 3))
        if flat:
            col = base
        else:
            rr = np.maximum(r, 0.05)
            u = np.clip(1.0 + d / rr, 0, 1)
            hgt = np.sqrt(np.clip(1 - u * u, 0, 1))
            gy, gx = np.gradient(hgt)
            k = rr * self.ss
            nx, ny = -gx * k, -gy * k
            nz = np.ones_like(hgt)
            inv = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz)
            nx, ny, nz = nx * inv, ny * inv, nz * inv
            diff = np.clip(nx * _L[0] + ny * _L[1] + nz * _L[2], 0, 1)
            sp = np.clip(nx * _Hh[0] + ny * _Hh[1] + nz * _Hh[2], 0, 1) ** shin
            shade = ambient + (1 - ambient) * diff
            occl = 0.80 + 0.20 * np.clip(hgt * 2.0, 0, 1)
            col = base * (shade * occl)[..., None] + 255.0 * spec * sp[..., None]
            col = np.clip(col, 0, 255)
        a = (inside * alpha)[..., None]
        a0 = self.img[..., 3:4]
        new_a = a + a0 * (1 - a)
        rgb = (col * a + self.img[..., :3] * a0 * (1 - a)) / np.maximum(new_a, 1e-6)
        self.img[..., :3] = np.where(new_a > 1e-6, rgb, 0)
        self.img[..., 3:4] = new_a

    def to_image(self, scale):
        k = self.ss // scale
        a = self.img[..., 3:4]
        pm = self.img[..., :3] * a
        H2, W2 = self.H // k, self.W // k
        pm = pm.reshape(H2, k, W2, k, 3).mean((1, 3))
        a = a.reshape(H2, k, W2, k, 1).mean((1, 3))
        rgb = np.where(a > 1e-4, pm / np.maximum(a, 1e-4), 0)
        out = np.concatenate([rgb, a * 255.0], -1)
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGBA")


def down(img, k):
    """Уменьшение PIL-картинки в k раз без тёмных краёв (с предумножением альфы)."""
    arr = np.asarray(img, np.float32)
    h, w = arr.shape[0] // k, arr.shape[1] // k
    arr = arr[:h * k, :w * k]
    a = arr[..., 3:4] / 255.0
    pm = arr[..., :3] * a
    pm = pm.reshape(h, k, w, k, 3).mean((1, 3))
    a = a.reshape(h, k, w, k, 1).mean((1, 3))
    rgb = np.where(a > 1e-4, pm / np.maximum(a, 1e-4), 0)
    out = np.concatenate([rgb, a * 255.0], -1)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGBA")


# ======================= описание деталей (общее для игры) =======================
# pivot — точка вращения в «обычных» пикселях спрайта, nominal — длина детали
PARTS = {
    "wheel":     dict(size=(40, 40), pivot=(20, 20)),
    "frame":     dict(size=(80, 58), pivot=(34, 36)),
    "rear_arm":  dict(size=(44, 14), pivot=(4, 7), nominal=36),
    "seatstay":  dict(size=(50, 12), pivot=(4, 6), nominal=42),
    "shock":     dict(size=(38, 12), pivot=(3, 6), nominal=32),
    "fork_up":   dict(size=(30, 10), pivot=(3, 5), nominal=24),
    "fork_low":  dict(size=(34, 14), pivot=(3, 6), nominal=24),
    "crank":     dict(size=(15, 8), pivot=(3, 4), nominal=9),
    "pedal":     dict(size=(14, 8), pivot=(7, 4)),
    "thigh":     dict(size=(36, 14), pivot=(6, 7), nominal=25),
    "shin":      dict(size=(38, 14), pivot=(6, 7), nominal=26),
    "knee":      dict(size=(18, 18), pivot=(9, 9)),
    "boot":      dict(size=(28, 16), pivot=(6, 8)),
    "torso":     dict(size=(52, 32), pivot=(9, 16), nominal=31),
    "upper_arm": dict(size=(30, 12), pivot=(5, 6), nominal=21),
    "forearm":   dict(size=(32, 12), pivot=(5, 6), nominal=22),
    "elbow":     dict(size=(14, 14), pivot=(7, 7)),
    "glove":     dict(size=(16, 14), pivot=(4, 7)),
    "head":      dict(size=(38, 38), pivot=(17, 19)),
}

# палитра
C_FRAME = (98, 102, 114)
C_ORANGE = (214, 96, 34)
C_STEEL = (168, 170, 178)
C_BLACK = (26, 26, 30)
C_GOLD = (206, 160, 62)
C_JERSEY = (96, 84, 76)
C_RED = (160, 46, 34)
C_ARMOR = (122, 126, 138)
C_PANTS = (58, 62, 74)
C_MUD = (86, 64, 44)


def save_part(name, cv):
    img = cv.to_image(HS)
    img.save(os.path.join(OUT, name + ".png"))


# ======================= байк =======================
def make_wheel():
    cv = Canvas(40, 40, 6)
    cx = cy = 20.0
    cv.add(cv.ring(cx, cy, 13.4, 2.8), (22, 22, 25), ambient=0.38, spec=0.18)
    knobs = []
    for k in range(22):
        a = k * 2 * math.pi / 22
        for side in (-1, 1):
            aa = a + side * 0.065
            c, s = math.cos(aa), math.sin(aa)
            knobs.append(cv.capsule(cx + c * 15.0, cy + s * 15.0, cx + c * 17.2, cy + s * 17.2, 0.85, 1.0))
    cv.add(cv.union(*knobs), (46, 46, 52), ambient=0.40, spec=0.22)
    cv.add(cv.ring(cx, cy, 9.8, 1.35), (112, 114, 124), ambient=0.40, spec=0.55, shin=40)
    for k in range(16):
        a = k * 2 * math.pi / 16
        sg = 1 if k % 2 else -1
        x0, y0 = cx + math.cos(a) * 2.8, cy + math.sin(a) * 2.8
        x1, y1 = cx + math.cos(a + sg * 0.30) * 9.5, cy + math.sin(a + sg * 0.30) * 9.5
        cv.add(cv.capsule(x0, y0, x1, y1, 0.28), (170, 172, 182), ambient=0.5, spec=0.5, shin=30)
    cv.add(cv.ring(cx, cy, 5.7, 0.85), (150, 126, 102), ambient=0.45, spec=0.4)
    arms = [cv.capsule(cx, cy, cx + math.cos(a) * 5.4, cy + math.sin(a) * 5.4, 0.7)
            for a in (0.3, 1.55, 2.8, 4.05, 5.3)]
    cv.add(cv.union(*arms), (140, 118, 96), ambient=0.45, spec=0.3)
    cv.add(cv.circle(cx, cy, 2.5), (206, 208, 214), ambient=0.35, spec=0.7, shin=36)
    cv.add(cv.circle(cx, cy, 1.0), (40, 40, 46), ambient=0.6, spec=0.1)
    rng = random.Random(5)
    for _ in range(9):
        a = rng.random() * 2 * math.pi
        rr = rng.uniform(10.5, 16.2)
        R = rng.uniform(0.9, 1.9)
        cv.add(cv.circle(cx + math.cos(a) * rr, cy + math.sin(a) * rr, R), C_MUD, ambient=0.5, spec=0.08)
    save_part("wheel", cv)


def make_frame():
    cv = Canvas(*PARTS["frame"]["size"], 6)
    OX, OY = PARTS["frame"]["pivot"]

    def P(x, y):
        return x + OX, y + OY

    steel = C_FRAME
    tubes = [
        cv.capsule(*P(-17, -14), *P(23, -13), 2.7, 2.5),          # верхняя труба
        cv.capsule(*P(-2, 12), *P(-17.5, -14), 2.7, 2.5),         # подседельная
        cv.capsule(*P(-2, 12), *P(23.5, -8), 3.6, 3.0),           # нижняя труба
        cv.capsule(*P(24, -18), *P(25, -5), 3.4),                 # рулевая
        cv.capsule(*P(-4, 4), *P(14, -9), 1.6, 1.4),              # усиливающая косынка
    ]
    cv.add(cv.union(*tubes), steel, ambient=0.40, spec=0.5, shin=34)
    # оранжевая наклейка на нижней трубе
    cv.add(cv.capsule(*P(3, 6.5), *P(18, -4.5), 0.9), C_ORANGE, ambient=0.6, spec=0.2)
    # бело-серая надпись-полоска на верхней трубе
    cv.add(cv.capsule(*P(-8, -14.2), *P(10, -13.6), 0.55), (200, 196, 188), ambient=0.7, spec=0.1)
    # каретка и звезда
    cv.add(cv.circle(*P(-2, 12), 4.3), (96, 98, 108), ambient=0.4, spec=0.6, shin=30)
    cv.add(cv.ring(*P(-2, 12), 7.6, 0.9), (176, 178, 186), ambient=0.4, spec=0.5)
    cv.add(cv.ring(*P(-2, 12), 5.4, 0.7), (120, 120, 130), ambient=0.4, spec=0.3)
    teeth = [cv.circle(P(-2, 12)[0] + math.cos(a) * 8.5, P(-2, 12)[1] + math.sin(a) * 8.5, 0.6)
             for a in np.linspace(0, 2 * math.pi, 26, endpoint=False)]
    cv.add(cv.union(*teeth), (170, 172, 180), ambient=0.5, spec=0.3)
    # узлы сварки
    for (x, y) in ((-17, -14), (24, -17), (24, -6), (-2, 12)):
        cv.add(cv.circle(*P(x, y), 3.2), (92, 94, 104), ambient=0.4, spec=0.5)
    # подседельный штырь и седло
    cv.add(cv.capsule(*P(-18, -14), *P(-20, -21), 1.8), C_STEEL, ambient=0.4, spec=0.7)
    cv.add(cv.union(cv.rbox(*P(-20.5, -23), 8.5, 2.4, 2.2, -0.05),
                    cv.capsule(*P(-28, -22.5), *P(-26, -20.5), 1.6)),
           (26, 26, 30), ambient=0.45, spec=0.35)
    # вынос, руль, ручка, рычаг тормоза
    cv.add(cv.capsule(*P(24, -19), *P(27, -23.5), 2.3), (36, 36, 42), ambient=0.4, spec=0.5)
    cv.add(cv.capsule(*P(24.5, -26.5), *P(34, -24), 1.5), (30, 30, 36), ambient=0.4, spec=0.5)
    cv.add(cv.capsule(*P(30, -25), *P(37, -23.6), 2.1), (34, 34, 38), ambient=0.45, spec=0.2)
    for gx in (31.5, 33.5, 35.5):
        cv.add(cv.capsule(*P(gx, -26.6), *P(gx + 0.3, -21.8), 0.35), (16, 16, 18), ambient=0.6, flat=True)
    cv.add(cv.capsule(*P(33.5, -24.5), *P(41, -21), 0.9), (170, 170, 178), ambient=0.4, spec=0.7)
    # номерная табличка и цифра
    cv.add(cv.rbox(*P(34, -13.5), 5.2, 5.8, 1.4, 0.06), (218, 210, 184), ambient=0.6, spec=0.2)
    dark = (34, 28, 24)
    cv.add(cv.capsule(*P(30.7, -17.3), *P(37.2, -17.0), 0.6), dark, flat=True)
    cv.add(cv.capsule(*P(37.2, -17.0), *P(33.2, -9.6), 0.7), dark, flat=True)
    cv.add(cv.capsule(*P(31.5, -13.5), *P(35.5, -13.4), 0.5), dark, flat=True)
    cv.add(cv.circle(*P(29.3, -16.2), 0.6), (150, 150, 156), flat=True)
    # грязь на раме
    for (x, y, R) in ((-8, 13, 1.6), (4, 15, 1.2), (14, 3, 1.1), (-22, -10, 1.0), (20, -13, 0.9)):
        cv.add(cv.circle(*P(x, y), R), C_MUD, ambient=0.5, spec=0.05)
    save_part("frame", cv)


def limb(name, length_px, ra, rb, base, deco=None, ambient=0.42, spec=0.3, shin=24, pivot_pad=None):
    w, h = PARTS[name]["size"]
    px, py = PARTS[name]["pivot"]
    cv = Canvas(w, h, 6)
    cv.add(cv.capsule(px, py, px + length_px, py, ra, rb), base, ambient=ambient, spec=spec, shin=shin)
    if deco:
        deco(cv, px, py)
    save_part(name, cv)


def make_chassis_parts():
    # заднее перо
    def deco_arm(cv, px, py):
        cv.add(cv.capsule(px + 4, py - 0.4, px + 26, py - 0.2, 0.7), C_ORANGE, ambient=0.6, spec=0.2)
        cv.add(cv.circle(px, py, 3.7), (100, 102, 112), ambient=0.4, spec=0.6)
        cv.add(cv.rbox(px + 33.5, py + 0.5, 3.8, 2.8, 1.0), (30, 30, 36), ambient=0.4, spec=0.4)
        cv.add(cv.rbox(px + 32, py + 4.6, 3.0, 2.2, 0.8), (182, 70, 34), ambient=0.5, spec=0.5)  # суппорт
    limb("rear_arm", 36, 3.0, 2.2, C_FRAME, deco_arm, spec=0.5, shin=34)
    limb("seatstay", 42, 1.9, 1.5, C_FRAME, spec=0.5, shin=34)

    # амортизатор с красной пружиной
    w, h = PARTS["shock"]["size"]
    cv = Canvas(w, h, 6)
    cv.add(cv.capsule(3, 6, 12, 6, 2.4), (22, 22, 26), ambient=0.4, spec=0.5, shin=30)
    cv.add(cv.capsule(10, 6, 14, 6, 1.1), C_STEEL, ambient=0.4, spec=0.7)
    coils = [cv.capsule(15 + i * 2.3, 2.1, 16.2 + i * 2.3, 9.9, 0.85) for i in range(7)]
    cv.add(cv.union(*coils), (204, 52, 34), ambient=0.4, spec=0.55, shin=24)
    cv.add(cv.capsule(31, 6, 35, 6, 2.2), (22, 22, 26), ambient=0.4, spec=0.5)
    cv.add(cv.circle(3, 6, 1.4), (140, 140, 150), ambient=0.4, spec=0.6)
    save_part("shock", cv)

    # вилка: золотая стойка и тёмная нога
    limb("fork_up", 24, 2.25, 2.25, C_GOLD, ambient=0.34, spec=0.8, shin=36)

    def deco_low(cv, px, py):
        cv.add(cv.capsule(px + 5, py - 0.3, px + 15, py - 0.3, 0.8), C_ORANGE, ambient=0.6, spec=0.2)
        cv.add(cv.rbox(px + 17.5, py + 3.4, 3.3, 2.5, 0.9), (182, 70, 34), ambient=0.5, spec=0.5)   # суппорт
        cv.add(cv.circle(px + 24, py, 2.4), (96, 98, 108), ambient=0.4, spec=0.6)                  # дропаут
        cv.add(cv.ring(px + 1.4, py, 2.9, 0.65), (14, 14, 16), ambient=0.4, spec=0.3)               # сальник
    limb("fork_low", 24, 2.75, 2.5, (36, 36, 42), deco_low, spec=0.45, shin=30)

    # шатун и педаль
    limb("crank", 9, 1.5, 1.3, (160, 162, 172), spec=0.6, shin=30)
    w, h = PARTS["pedal"]["size"]
    cv = Canvas(w, h, 6)
    cv.add(cv.rbox(7, 4, 5.6, 1.5, 0.9), (30, 30, 34), ambient=0.45, spec=0.4)
    for x in (3, 5.5, 8.5, 11):
        cv.add(cv.circle(x, 2.4, 0.35), (170, 170, 178), flat=True)
    save_part("pedal", cv)


# ======================= гонщик =======================
def make_rider():
    # бедро и голень (штаны)
    def deco_thigh(cv, px, py):
        for t in (0.35, 0.62):
            cv.add(cv.capsule(px + 24 * t, py - 2.8, px + 24 * t + 1.6, py + 2.8, 0.35), (30, 31, 36), ambient=0.6, flat=True)
        cv.add(cv.capsule(px + 3, py - 3.8, px + 22, py - 3.0, 0.5), (96, 98, 108), ambient=0.6, spec=0.1, alpha=0.5)
    limb("thigh", 25, 5.3, 4.3, C_PANTS, deco_thigh, ambient=0.45, spec=0.12, shin=14)

    def deco_shin(cv, px, py):
        cv.add(cv.capsule(px + 3, py - 0.4, px + 22, py - 0.2, 2.7, 2.3), C_ARMOR, ambient=0.34, spec=0.65, shin=30)
        for t in (0.25, 0.6):
            cv.add(cv.capsule(px + 3 + 19 * t, py - 3.2, px + 3.6 + 19 * t, py + 2.8, 0.5), (24, 24, 28), flat=True)
        cv.add(cv.capsule(px + 4, py - 1.6, px + 20, py - 1.4, 0.5), (150, 152, 164), ambient=0.6, spec=0.2, alpha=0.7)
        cv.add(cv.capsule(px + 23.5, py, px + 28.5, py, 4.0, 4.1), (24, 24, 28), ambient=0.4, spec=0.3)   # верх ботинка
    limb("shin", 26, 4.3, 3.5, C_PANTS, deco_shin, ambient=0.45, spec=0.12, shin=14)

    # наколенник
    cv = Canvas(18, 18, 6)
    cv.add(cv.circle(9, 9, 6.3), C_ARMOR, ambient=0.32, spec=0.7, shin=28)
    cv.add(cv.ring(9, 9, 4.0, 0.5), (50, 52, 60), ambient=0.5, spec=0.2)
    cv.add(cv.capsule(3.5, 13.5, 14.5, 13.5, 0.7), (28, 28, 32), ambient=0.6, flat=True)
    cv.add(cv.capsule(5.5, 4.8, 10, 4.2, 0.6), (190, 192, 204), flat=True, alpha=0.8)
    save_part("knee", cv)

    # ботинок
    cv = Canvas(28, 16, 6)
    cv.add(cv.union(cv.capsule(6, 8, 19, 8.6, 3.4), cv.circle(20.2, 8.8, 3.3)), (40, 40, 46), ambient=0.38, spec=0.4, shin=22)
    cv.add(cv.capsule(4, 11.4, 24, 11.6, 1.2), (14, 14, 16), ambient=0.5, spec=0.15)
    for x in (10, 14):
        cv.add(cv.capsule(x, 4.2, x + 0.6, 11.4, 0.65), (176, 178, 188), ambient=0.4, spec=0.7)
    cv.add(cv.capsule(8, 6, 17, 6.4, 0.45), (210, 70, 40), flat=True, alpha=0.85)
    cv.add(cv.circle(22.5, 10.2, 1.3), C_MUD, ambient=0.5, spec=0.05)
    save_part("boot", cv)

    # торс: куртка, нагрудник, рюкзак-гидропак
    cv = Canvas(52, 32, 6)
    px, py = PARTS["torso"]["pivot"]
    cv.add(cv.rbox(px + 14, py - 7.6, 10.5, 2.9, 1.8), (52, 56, 66), ambient=0.4, spec=0.3)          # рюкзак (спина)
    cv.add(cv.capsule(px + 5, py - 9.4, px + 22, py - 9.5, 0.5), (110, 114, 126), ambient=0.6, flat=True, alpha=0.8)
    cv.add(cv.capsule(px, py, px + 31, py, 6.6, 5.7), C_JERSEY, ambient=0.42, spec=0.2, shin=16)   # корпус
    for t in (0.34, 0.54, 0.74):                                                                      # пластины нагрудника
        cv.add(cv.rbox(px + 31 * t, py + 3.0, 3.4, 2.8, 1.0), C_ARMOR, ambient=0.34, spec=0.6, shin=26)
    cv.add(cv.capsule(px + 1, py - 0.6, px + 30, py - 0.6, 0.85), C_RED, ambient=0.6, spec=0.15)     # красная полоса
    cv.add(cv.capsule(px + 1, py + 1.2, px + 30, py + 1.2, 0.3), (120, 122, 132), flat=True, alpha=0.7)
    cv.add(cv.circle(px + 31.5, py - 0.2, 5.9), C_JERSEY, ambient=0.42, spec=0.2)                    # плечо
    cv.add(cv.circle(px + 32.6, py - 2.6, 3.0), C_ARMOR, ambient=0.34, spec=0.5)                     # наплечник
    cv.add(cv.capsule(px + 2, py + 6.8, px + 27, py + 6.4, 0.55), (70, 72, 82), flat=True, alpha=0.8) # шов
    save_part("torso", cv)

    # рука
    def deco_ua(cv, px, py):
        cv.add(cv.capsule(px + 3, py - 0.2, px + 16, py - 0.2, 0.7), C_RED, ambient=0.6, spec=0.15)
    limb("upper_arm", 21, 3.9, 3.4, C_JERSEY, deco_ua, ambient=0.40, spec=0.18, shin=16)

    def deco_fa(cv, px, py):
        cv.add(cv.capsule(px + 19.2, py, px + 22.0, py, 3.5, 3.5), (30, 30, 34), ambient=0.4, spec=0.2)   # манжета
    limb("forearm", 22, 3.3, 2.8, C_JERSEY, deco_fa, ambient=0.40, spec=0.18, shin=16)

    cv = Canvas(14, 14, 6)
    cv.add(cv.circle(7, 7, 4.8), C_ARMOR, ambient=0.32, spec=0.7, shin=28)
    cv.add(cv.ring(7, 7, 2.9, 0.45), (50, 52, 60), ambient=0.5)
    save_part("elbow", cv)

    # перчатка с защитой костяшек
    cv = Canvas(16, 14, 6)
    cv.add(cv.capsule(4, 7, 10, 7, 3.3), (28, 28, 32), ambient=0.4, spec=0.25)
    cv.add(cv.rbox(10, 5, 3.2, 2.1, 0.8, 0.25), C_ARMOR, ambient=0.34, spec=0.6)
    cv.add(cv.union(cv.capsule(9, 8.4, 12.4, 9.4, 1.0), cv.capsule(8, 9.2, 11.4, 10.6, 0.95)), (22, 22, 26), ambient=0.4, spec=0.2)
    cv.add(cv.capsule(3, 6.6, 6, 6.6, 0.6), C_RED, flat=True)
    save_part("glove", cv)

    # шлем-интеграл
    cv = Canvas(38, 38, 6)
    cx, cy = 17.0, 19.0
    cv.add(cv.capsule(cx - 10, cy + 7.5, cx - 3, cy + 9.5, 3.0), C_ORANGE, ambient=0.40, spec=0.35)        # шейный брейс
    cv.add(cv.circle(cx, cy, 11.2), (36, 37, 42), ambient=0.40, spec=0.45, shin=22)                          # скорлупа
    cv.add(cv.capsule(cx + 3.5, cy + 4.5, cx + 12.5, cy + 3.2, 4.2, 3.4), (28, 29, 33), ambient=0.38, spec=0.4)  # подбородок
    cv.add(cv.capsule(cx + 12.6, cy + 3.2, cx + 12.8, cy + 7.6, 1.8), (20, 20, 24), ambient=0.4, spec=0.3)  # вентиляция
    cv.add(cv.capsule(cx - 5, cy - 8.5, cx + 14.5, cy - 11.0, 1.0, 1.8), (22, 22, 26), ambient=0.4, spec=0.4)  # козырёк
    cv.add(cv.capsule(cx - 9, cy - 3, cx + 3, cy - 10.4, 0.95), (170, 52, 36), ambient=0.6, spec=0.25)       # полоса
    cv.add(cv.rbox(cx + 6.4, cy - 0.8, 6.8, 4.4, 1.9, 0.12), (12, 12, 14), ambient=0.4, spec=0.3)            # рамка очков
    cv.add(cv.rbox(cx + 6.8, cy - 0.8, 5.5, 3.2, 1.5, 0.12), (214, 126, 38), ambient=0.34, spec=0.9, shin=18)  # янтарная линза
    cv.add(cv.capsule(cx + 3.2, cy - 2.2, cx + 8.0, cy - 2.8, 0.45), (255, 232, 180), flat=True, alpha=0.85)
    cv.add(cv.capsule(cx - 11, cy - 0.4, cx - 3, cy - 0.2, 1.3), (14, 14, 16), ambient=0.4, spec=0.1)        # ремешок очков
    for (vx, vy) in ((cx - 5.5, cy - 6.2), (cx - 2.2, cy - 8.2), (cx + 1.0, cy - 9.2)):
        cv.add(cv.rbox(vx, vy, 1.3, 0.6, 0.5, -0.5), (8, 8, 10), flat=True)
    cv.add(cv.circle(cx - 6.5, cy - 7.5, 1.4), C_MUD, ambient=0.5, spec=0.05)
    save_part("head", cv)


def generate_characters():
    make_wheel()
    make_frame()
    make_chassis_parts()
    make_rider()


# ======================= окружение =======================
def save_img(img, name):
    img.save(os.path.join(OUT, name + ".png"))


def arr_to_img(arr, mode="RGB"):
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode)


def make_sky(W=960, H=540):
    rng = np.random.default_rng(11)
    ys = np.linspace(0, 1, H, dtype=np.float32)
    st = [0.0, 0.40, 0.68, 0.88, 1.0]
    sc = [(30, 34, 44), (74, 72, 80), (150, 116, 92), (214, 152, 96), (232, 172, 108)]
    col = np.stack([np.interp(ys, st, [c[k] for c in sc]) for k in range(3)], -1).astype(np.float32)
    col = np.repeat(col[:, None, :], W, 1)
    xs = np.arange(W, dtype=np.float32)[None, :]
    yy = np.arange(H, dtype=np.float32)[:, None]
    sx, sy = W * 0.74, 215.0
    dist = np.hypot(xs - sx, yy - sy)
    glow = np.clip(np.exp(-(dist / 170) ** 1.3) * 0.7 + np.exp(-(dist / 38) ** 2), 0, 1)[..., None]
    col = col * (1 - glow) + np.array([255, 176, 96], np.float32) * glow
    core = np.clip(1 - (dist / 24) ** 4, 0, 1)[..., None]
    col = col * (1 - core) + np.array([255, 228, 172], np.float32) * core
    n1 = fbm(H, W, (3, 4), 5, rng)
    n2 = fbm(H, W, (6, 9), 4, rng)
    band = (1 - smoothstep(0.30, 0.80, ys))[:, None]
    dens = smoothstep(0.46, 0.74, n1 * 0.65 + n2 * 0.35) * band
    lit = np.clip(1 - dist / 720, 0, 1)
    detail = 0.35 + 0.65 * n2
    cc = lerp(np.array([40, 38, 50], np.float32), np.array([224, 140, 92], np.float32), ((lit * detail) ** 0.9)[..., None])
    k = (dens * 0.9)[..., None]
    col = col * (1 - k) + cc * k
    col += (rng.random((H, W, 1)) - 0.5) * 5
    save_img(arr_to_img(col), "sky")


def make_clouds():
    for i in range(4):
        rng = np.random.default_rng(100 + i)
        w, h = 380, 120
        n = fbm(h, w, (2, 3), 5, rng)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        ell = 1 - np.clip(((xx - w / 2) / (w * 0.5)) ** 2 + ((yy - h * 0.55) / (h * 0.45)) ** 2, 0, 1)
        dens = smoothstep(0.30, 0.66, n * ell * 1.7) * np.clip(ell * 1.6, 0, 1)
        under = smoothstep(0.30, 0.95, yy / h + 0.30 * (n - 0.5))
        col = lerp(np.array([56, 52, 62], np.float32), np.array([206, 130, 94], np.float32), under[..., None])
        arr = np.concatenate([col, (dens * 150)[..., None]], -1)
        save_img(arr_to_img(arr, "RGBA"), f"cloud_{i}")


def make_mountains(name, w, h, seed, amp, c_top, c_bot, haze, cells, snow=False):
    rng = np.random.default_rng(seed)
    r1 = noise1d(w, cells, rng)
    r2 = noise1d(w, cells * 2 + 1, rng)
    r3 = noise1d(w, cells * 7, rng)
    ridge = 0.54 * (1 - np.abs(2 * r1 - 1)) + 0.26 * (1 - np.abs(2 * r2 - 1)) + 0.20 * r3
    ridge = circ_smooth(ridge, 7)
    top = (h - 6) - ridge * amp
    yy = np.arange(h, dtype=np.float32)[:, None]
    depth = yy - top[None, :]
    alpha = np.clip(depth + 0.5, 0, 1)
    slope = np.gradient(top)
    facet = np.clip(0.55 + slope * 0.60, 0.12, 1.0)
    facet = np.convolve(facet, np.ones(5) / 5, "same")
    gul = fbm(h, w, (7, w // 30), 3, rng)
    gul2 = fbm(h, w, (5, w // 60), 3, rng)
    shade = (0.35 + 0.65 * facet[None, :]) * (0.78 + 0.5 * gul) * (0.86 + 0.28 * gul2)
    t = np.clip(depth / (h * 0.9), 0, 1)[..., None]
    base = lerp(np.array(c_top, np.float32), np.array(c_bot, np.float32), t)
    col = base * np.clip(shade, 0, 1.7)[..., None]
    hz = (np.clip((yy - h * 0.45) / (h * 0.55), 0, 1) ** 1.3 * 0.55)[..., None]
    col = col * (1 - hz) + np.array(haze, np.float32) * hz
    if snow:
        snow_level = h - 6 - amp * 0.52
        sd = np.where(top < snow_level, (snow_level - top) * 0.92 + 6 * (noise1d(w, w // 9, rng) - 0.5), 0)
        sd = np.maximum(sd, 0)
        mask = (depth < sd[None, :] * (0.75 + 0.5 * gul2)) & (depth > 0)
        sshade = 0.45 + 0.65 * facet[None, :] * (0.8 + 0.4 * gul)
        sc = np.array([226, 220, 226], np.float32) * np.clip(sshade, 0, 1.3)[..., None]
        sc = sc * (1 - hz * 0.6) + np.array(haze, np.float32) * hz * 0.6
        col = np.where(mask[..., None], sc, col)
    rim = (np.clip(1 - depth / 3.0, 0, 1) * facet[None, :])[..., None]
    col = col + rim * np.array([90, 50, 20], np.float32)
    out = np.concatenate([col, (alpha * 255)[..., None]], -1)
    save_img(arr_to_img(out, "RGBA"), name)


def make_hills(w=1920, h=240, seed=21):
    rng = np.random.default_rng(seed)
    r1 = noise1d(w, 8, rng)
    r2 = noise1d(w, 24, rng)
    ridge = 0.65 * r1 + 0.35 * r2
    base_top = (h - 12) - ridge * 62
    tw = 11
    n = w // tw
    th1 = rng.random(n) * 15 + 6
    th2 = rng.random(n) * 15 + 5
    xs = np.arange(w)
    tri1 = 1 - np.abs(((xs % tw) / tw) - 0.5) * 2
    tri2 = 1 - np.abs((((xs + 5) % w) % tw) / tw - 0.5) * 2
    p1 = th1[(xs // tw) % n] * tri1 ** 0.85
    p2 = th2[(((xs + 5) % w) // tw) % n] * tri2 ** 0.85
    top = base_top - np.maximum(p1, p2 * 0.9)
    yy = np.arange(h, dtype=np.float32)[:, None]
    depth = yy - top[None, :]
    alpha = np.clip(depth + 0.5, 0, 1)
    nz = fbm(h, w, (6, 90), 3, rng)
    t = np.clip(depth / 160, 0, 1)[..., None]
    col = lerp(np.array([30, 38, 38], np.float32), np.array([12, 16, 18], np.float32), t) * (0.75 + 0.5 * nz)[..., None]
    hz = (np.clip(1 - depth / 90, 0, 1) ** 1.5 * 0.40)[..., None]
    col = col * (1 - hz) + np.array([118, 88, 70], np.float32) * hz
    out = np.concatenate([col, (alpha * 255)[..., None]], -1)
    save_img(arr_to_img(out, "RGBA"), "hills")


def make_dirt(w=504, h=480):
    rng = np.random.default_rng(31)
    yy = np.linspace(0, 1, h, dtype=np.float32)
    st = [0.0, 0.10, 0.45, 1.0]
    sc = [(126, 100, 74), (104, 82, 60), (72, 56, 42), (30, 24, 20)]
    base = np.stack([np.interp(yy, st, [c[k] for c in sc]) for k in range(3)], -1)[:, None, :]
    n = fbm(h, w, (8, 6), 5, rng)
    n2 = fbm(h, w, (24, 18), 3, rng)
    strata = 0.5 + 0.5 * np.sin(yy[:, None] * h / 9.0 + n * 7.0)
    tex = 0.66 + 0.60 * n + 0.22 * (n2 - 0.5) + 0.10 * (strata - 0.5)
    col = base * tex[..., None]
    ss = 3
    ov = Image.new("RGBA", (w * ss, h * ss), (0, 0, 0, 0))
    r = random.Random(4)
    d = ImageDraw.Draw(ov)
    for _ in range(130):
        cx = r.random() * w * ss
        cy = (r.random() ** 1.6) * h * ss
        rx = r.uniform(1.8, 9.5) * ss
        ry = rx * r.uniform(0.55, 0.85)
        g = r.randint(58, 118)
        tone = (g + 16 + r.randint(-4, 8), g + 2, g - 12 - r.randint(0, 10))
        for ox in (-w * ss, 0, w * ss):
            x = cx + ox
            d.ellipse([x - rx + 1.5 * ss, cy - ry + 2 * ss, x + rx + 1.5 * ss, cy + ry + 2 * ss], fill=(10, 8, 6, 150))
            d.ellipse([x - rx, cy - ry, x + rx, cy + ry], fill=tone + (255,))
            d.ellipse([x - rx * 0.6, cy - ry * 0.75, x + rx * 0.2, cy - ry * 0.05],
                      fill=(min(255, tone[0] + 42), min(255, tone[1] + 38), min(255, tone[2] + 30), 255))
            d.arc([x - rx, cy - ry, x + rx, cy + ry], 200, 340, fill=(20, 16, 12, 220), width=ss)
    for _ in range(8):
        x = r.random() * w * ss
        y = r.random() * 130 * ss
        pts = [(x, y)]
        for _k in range(6):
            x += r.uniform(-9, 9) * ss
            y += r.uniform(5, 14) * ss
            pts.append((x, y))
        for ox in (-w * ss, 0, w * ss):
            d.line([(px + ox, py) for px, py in pts], fill=(30, 23, 18, 165), width=int(1.3 * ss))
    ov = down(ov, ss)
    img = arr_to_img(col, "RGB").convert("RGBA")
    img.alpha_composite(ov)
    save_img(img.convert("RGB"), "dirt")


def make_grass(w=504, h=44):
    ss = 4
    img = Image.new("RGBA", (w * ss, h * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    r = random.Random(7)
    pal = [(46, 42, 22), (70, 64, 34), (98, 90, 48), (132, 118, 64), (166, 148, 80)]
    blades = []
    for _ in range(1350):
        blades.append((r.random() * w * ss, r.uniform(5, 21) * ss, r.uniform(-9, 9) * ss,
                       r.uniform(0.9, 2.1) * ss, r.random()))
    blades.sort(key=lambda b: b[1])
    base_y = (h - 6) * ss
    for x, ht, bend, wd, sh in blades:
        for ox in (-w * ss, 0, w * ss):
            bx = x + ox
            tipx, tipy = bx + bend, base_y - ht
            ci = min(3, int(sh * 4))
            d.polygon([(bx - wd / 2, base_y + 2 * ss), (bx + wd / 2, base_y + 2 * ss), (tipx, tipy)], fill=pal[ci] + (255,))
            mx, my = bx + bend * 0.5, base_y - ht * 0.5
            d.polygon([(mx - wd * 0.28, my), (mx + wd * 0.28, my), (tipx, tipy)], fill=pal[min(4, ci + 1)] + (255,))
    soil_top = (h - 8) * ss
    for x in range(0, w * ss, 3):
        jitter = r.uniform(-1.4, 1.4) * ss
        d.rectangle([x, soil_top + jitter, x + 3, h * ss], fill=(58 + (x % 7), 46, 28, 255))
    d.rectangle([0, soil_top + 3 * ss, w * ss, h * ss], fill=(70, 54, 34, 255))
    save_img(down(img, ss), "grass")


def make_rocks():
    sizes = [(64, 44), (84, 58), (56, 40), (100, 66), (72, 50), (46, 34)]
    for i, (w, h) in enumerate(sizes):
        ss = 4
        r = random.Random(40 + i)
        img = Image.new("RGBA", (w * ss, h * ss), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        cx, yb = w * ss / 2, (h - 6) * ss
        rx, ry = w * 0.46 * ss, (h - 12) * ss * 0.95
        d.ellipse([cx - rx * 1.05, yb - 2 * ss, cx + rx * 1.05, yb + 5 * ss], fill=(0, 0, 0, 110))
        pts = []
        for k in range(12):
            a = math.pi + k * math.pi / 11
            pts.append((cx + math.cos(a) * rx * r.uniform(0.86, 1.08), yb + math.sin(a) * ry * r.uniform(0.78, 1.05)))
        pts += [(cx + rx * 1.02, yb + 3 * ss), (cx - rx * 1.02, yb + 3 * ss)]
        c0 = (cx + r.uniform(-0.1, 0.2) * rx, yb - ry * 0.42)
        base_c = (112 + r.randint(-10, 10), 106 + r.randint(-10, 10), 100 + r.randint(-10, 10))
        light_ang = -math.pi / 3
        for k in range(len(pts)):
            p, q = pts[k], pts[(k + 1) % len(pts)]
            mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
            ang = math.atan2(my - c0[1], mx - c0[0])
            sh = 0.5 + 0.5 * math.cos(ang - light_ang)
            col = tuple(min(255, int(base_c[c] * (0.34 + 0.95 * sh) + (22, 10, 0)[c] * sh)) for c in range(3))
            d.polygon([c0, p, q], fill=col + (255,))
        for _k in range(7):
            p = pts[r.randrange(len(pts))]
            d.line([c0, p], fill=(24, 22, 22, 150), width=ss)
        for _k in range(3):
            x = cx + r.uniform(-0.6, 0.6) * rx
            y = yb - ry * r.uniform(0.2, 0.8)
            d.line([(x, y), (x + r.uniform(-8, 8) * ss, y + r.uniform(4, 11) * ss)], fill=(18, 16, 16, 220), width=int(1.4 * ss))
        for _k in range(5):
            x = cx + r.uniform(-0.7, 0.7) * rx
            y = yb - ry * r.uniform(0.45, 0.95)
            rr = r.uniform(2, 6) * ss
            d.ellipse([x - rr, y - rr * 0.5, x + rr, y + rr * 0.5], fill=(100, 96, 50, 130))
        d.line(pts[:-2], fill=(20, 18, 18, 230), width=int(1.6 * ss))
        save_img(down(img, ss), f"rock_{i}")


def make_pine(h, layer, seed):
    w = int(h * 0.6)
    ss = 4
    r = random.Random(seed)
    img = Image.new("RGBA", (w * ss, h * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if layer == 0:
        dark, mid, rim = (24, 32, 34), (36, 48, 50), (110, 82, 60)
    else:
        dark, mid, rim = (14, 28, 20), (28, 50, 32), (112, 92, 54)
    cx = w * ss / 2
    tw = max(2, w // 14) * ss
    d.rectangle([cx - tw / 2, (h - h // 8) * ss, cx + tw / 2, h * ss], fill=(40, 30, 24, 255))
    tiers = 6
    body = h * 0.93 * ss
    for i in range(tiers):
        top = i * body / (tiers + 0.9)
        bot = top + body * 0.34
        half = (w * ss / 2) * (0.26 + 0.74 * (i + 1) / tiers) * 0.97
        left, right = [], []
        for k in range(1, 8):
            t = k / 7
            y = top + (bot - top) * t ** 0.92
            zig = r.uniform(-1.5, 1.5) * ss * (1 if k % 2 else -1)
            left.append((cx - half * t - zig, y))
            right.append((cx + half * t + zig, y))
        scallop = []
        n = 9
        for k in range(n + 1):
            t = k / n
            x = cx - half + 2 * half * t
            scallop.append((x, bot + (3 if k % 2 else -1.0) * ss + r.uniform(-1, 1) * ss))
        poly = [(cx, top)] + left + scallop + right[::-1]
        d.polygon(poly, fill=dark + (255,))
        d.polygon([(cx, top)] + [(cx + (x - cx) * 0.62, y) for (x, y) in right[:6]] +
                  [(cx + half * 0.5, bot), (cx, bot - 2 * ss)], fill=mid + (255,))
        for _k in range(70):
            t = r.random()
            y = top + (bot - top) * t
            spread = half * t
            x = cx + r.uniform(-spread, spread)
            c = mid if x > cx else dark
            d.line([(x, y), (x + r.uniform(-4, 4) * ss, y + r.uniform(2, 6) * ss)], fill=c + (255,), width=ss)
        d.line([(cx, top)] + right[:], fill=rim + (255,), width=int(1.3 * ss))
    save_img(down(img, ss), f"pine_{layer}_{h}")


def make_dead_tree(h, seed):
    w = int(h * 0.7)
    ss = 4
    r = random.Random(seed)
    img = Image.new("RGBA", (w * ss, h * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cx = w * ss / 2
    col, hi = (34, 27, 25), (112, 84, 58)
    d.polygon([(cx - 5 * ss, h * ss), (cx + 5 * ss, h * ss), (cx + 1.8 * ss, h * 0.28 * ss), (cx - 1.8 * ss, h * 0.28 * ss)], fill=col + (255,))
    d.line([(cx + 3 * ss, h * ss), (cx + 1.2 * ss, h * 0.3 * ss)], fill=hi + (255,), width=ss)

    def branch(x, y, ang, ln, wd, depth):
        x2, y2 = x + math.cos(ang) * ln, y + math.sin(ang) * ln
        d.line([(x, y), (x2, y2)], fill=col + (255,), width=max(1, int(wd)))
        if depth > 0:
            branch(x2, y2, ang + r.uniform(0.35, 0.8), ln * 0.66, wd * 0.7, depth - 1)
            branch(x2, y2, ang - r.uniform(0.35, 0.8), ln * 0.62, wd * 0.7, depth - 1)

    for i in range(7):
        y = (h - h * (0.25 + 0.1 * i)) * ss
        side = -1 if i % 2 == 0 else 1
        ang = -math.pi / 2 + side * r.uniform(0.7, 1.1)
        branch(cx, y, ang, w * ss * (0.20 + 0.16 * (1 - i / 8)), 3.4 * ss * (1 - i * 0.07), 2)
    save_img(down(img, ss), f"dead_{h}")


def make_flags():
    w, h = 60, 112
    ss = 4
    for name, cloth in (("red", (150, 36, 30)), ("gold", (208, 160, 50))):
        for f in range(4):
            r = random.Random(60 + f)
            img = Image.new("RGBA", (w * ss, h * ss), (0, 0, 0, 0))
            d = ImageDraw.Draw(img)
            px = 7 * ss
            d.rectangle([px - 2.2 * ss, 8 * ss, px + 2.2 * ss, h * ss], fill=(98, 90, 84, 255))
            d.rectangle([px + 0.6 * ss, 8 * ss, px + 2.2 * ss, h * ss], fill=(150, 140, 128, 255))
            d.ellipse([px - 3.4 * ss, 4.6 * ss, px + 3.4 * ss, 11.4 * ss], fill=(176, 100, 56, 255))
            for st in range(8):
                d.ellipse([px - 7 * ss + st * 2 * ss, (h - 4) * ss, px - 3 * ss + st * 2 * ss, (h + 2) * ss], fill=(70, 66, 62, 255))
            phase = f * math.pi / 2
            x0 = px + 2.2 * ss
            ncol = 48
            for cidx in range(ncol):
                u = cidx / ncol
                wob = math.sin(phase + u * 4.4) * 3.2 * u * ss
                ht = 30 * (1 - 0.16 * u) * ss
                torn = 0.0
                if u > 0.5:
                    torn = (r.random() * 6 + (3 if cidx % 3 == 0 else 0)) * (u - 0.4) * ss
                top = 12 * ss + wob
                bot = top + ht - torn
                shd = 0.72 + 0.34 * math.cos(phase + u * 4.4)
                c = tuple(min(255, int(cloth[k] * shd)) for k in range(3))
                d.rectangle([x0 + cidx * ss, top, x0 + (cidx + 1) * ss, bot], fill=c + (255,))
                mid = top + ht * 0.5
                d.rectangle([x0 + cidx * ss, mid - 1.6 * ss, x0 + (cidx + 1) * ss, mid + 1.6 * ss],
                            fill=tuple(min(255, int(v * shd)) for v in (206, 198, 186)) + (255,))
            d.line([(x0, 12 * ss), (x0 + ncol * ss, 12 * ss + math.sin(phase + 4.4) * 3.2 * ss)], fill=(30, 18, 14, 255), width=int(0.9 * ss))
            save_img(down(img, ss), f"flag_{name}_{f}")


def generate_environment():
    make_sky()
    make_clouds()
    make_mountains("mtn_far", 1920, 340, 5, 200, (132, 112, 124), (78, 70, 86), (196, 150, 120), 5, snow=True)
    make_mountains("mtn_mid", 1920, 300, 9, 135, (84, 76, 90), (52, 48, 62), (176, 134, 106), 8)
    make_hills()
    make_dirt()
    make_grass()
    make_rocks()
    for layer in (0, 1):
        for i, h in enumerate((58, 78, 100, 126, 152)):
            make_pine(h, layer, 100 + layer * 10 + i)
    for i, h in enumerate((70, 96, 124)):
        make_dead_tree(h, 200 + i)
    make_flags()


def generate_all():
    _np()  # numpy подключается только здесь, для генерации картинок
    os.makedirs(OUT, exist_ok=True)
    generate_characters()
    generate_environment()
    with open(os.path.join(OUT, "VERSION"), "w") as f:
        f.write(VERSION)


def ensure_assets():
    """Создаёт картинки, если их ещё нет или версия генератора поменялась."""
    vf = os.path.join(OUT, "VERSION")
    ok = os.path.exists(vf) and open(vf).read().strip() == VERSION
    if not ok:
        print("Рисую картинки для игры (один раз, около минуты)…")
        generate_all()
        print("Готово.")


if __name__ == "__main__":
    generate_all()
    print("all assets ok")