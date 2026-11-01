"""
Горный байкер — 2D-игра с физикой: едем на горном велосипеде по холмам,
делаем кувырки и стараемся не удариться головой.

Установка: pip install pygame-ce pymunk numpy pillow
Запуск:    python mountain_rider.py

Управление (клавиатура):
  D / стрелка вправо  — газ
  A / стрелка влево   — тормоз
  W / стрелка вверх   — наклон назад (нос вверх)
  S / стрелка вниз    — наклон вперёд (нос вниз)
  R — после падения: с последней контрольной точки
  N — после падения: заново с начала
  F2 — переключить качество графики (высокое / низкое)
  ESC — выход
На сенсорном экране работают экранные кнопки.
"""

import math
import os
import random
import sys

import pygame
import pymunk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_assets as MA  # noqa: E402

HS = MA.HS

# ---------- Настройки ----------
W, H = 960, 540
FPS = 60
SUBSTEPS = 3
GRAVITY_BASE = 1100
GRAVITY_CUT = -220            # купленная сниженная гравитация (1100 -> 880)
GRAVITY_COST = 320

WHEEL_R = 17
CHASSIS_SIZE = (70, 12)
TORQUE_BASE = 42000
WHEEL_SPEED_BASE = 60       # рад/с
LEAN_TORQUE = 52000
CHECKPOINT_EVERY = 900     # пикселей
PX_PER_METER = 40
SEGMENT = 24               # длина кусочка земли для физики
HEAD_LOCAL = (8, -46)      # положение головы в системе рамы
FORK_SETTLED = 72.5        # длина вилки в покое с учётом провисания пружины

# Магазин улучшений (покупается за очки)
SPEED_PER_LVL = 8            # +макс. скорость колёс за уровень
SPEED_COST = 40              # базовая цена; растёт на 12 за уровень
SPEED_MAX = 5
ACCEL_PER_LVL = 9000         # +момент двигателя за уровень
ACCEL_COST = 35              # базовая цена; растёт на 12 за уровень
ACCEL_MAX = 5
DIST_POINTS_PER_M = 0.1      # очки за метр дистанции
FLIP_BONUS = 25              # очки за кувырок
CHECKPOINT_BONUS = 40        # очки за пройденную контрольную точку

# Админ-панель (отладка, F1)
ADMIN_G_MIN = 300
ADMIN_G_STEP = 50
ADMIN_SPEED_MIN = 20
ADMIN_SPEED_STEP = 5
ADMIN_TORQUE_MIN = 5000
ADMIN_TORQUE_STEP = 5000
ADMIN_LEAN_MIN = 5000
ADMIN_LEAN_STEP = 5000
ADMIN_POINTS_STEP = 100

UI_TEXT = (228, 222, 210)
AMBER = (240, 172, 60)

# Цвета байка и гонщика (суровый стиль)
FRAME = (54, 56, 62)
FRAME_EDGE = (12, 12, 14)
ACCENT = (206, 92, 34)
STEEL = (150, 152, 158)
BLACK = (20, 20, 22)
JERSEY = (46, 48, 54)
JERSEY_DARK = (26, 27, 31)
ARMOR = (78, 80, 90)
PANTS = (42, 44, 50)
PANTS_FAR = (26, 27, 31)
HELMET = (34, 35, 40)
GLOVE = (24, 24, 26)
MUD = (80, 60, 42)


# ---------- Вспомогательные функции ----------
def hrand(i, salt=0):
    """Детерминированный «случайный» шум 0..1 по номеру (для деревьев, камней)."""
    n = (int(i) * 374761393 + salt * 668265263) & 0xFFFFFFFF
    n = ((n ^ (n >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((n ^ (n >> 16)) & 0xFFFF) / 65535.0


def lerp_color(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def lerp_pt(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def ik(a, b, l1, l2, flip):
    """Двухзвенная обратная кинематика: возвращает (сустав, конец)."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    d = max(0.001, min(math.hypot(dx, dy), l1 + l2 - 0.01))
    cos_a = max(-1.0, min(1.0, (l1 * l1 + d * d - l2 * l2) / (2 * l1 * d)))
    ang_a = math.acos(cos_a)
    base = math.atan2(dy, dx)
    ang = base + (ang_a if flip else -ang_a)
    joint = (a[0] + l1 * math.cos(ang), a[1] + l1 * math.sin(ang))
    end = (a[0] + math.cos(base) * d, a[1] + math.sin(base) * d)
    return joint, end


# ---------- Рельеф ----------
def terrain_y(x):
    """Высота земли в точке x (ось Y направлена вниз). Чем дальше, тем круче."""
    base = 400.0
    t = max(0.0, x - 500.0)
    ramp = min(1.0, t / 7000.0)
    amp = 18 + 50 * ramp
    y = (
        amp * math.sin(t / 170.0)
        + amp * 0.25 * math.sin(t / 80.0 + 1.3)
        + 30 * ramp * math.sin(t / 420.0)
    )
    blend = min(1.0, max(0.0, (x - 350.0) / 300.0))  # плавный старт на ровном месте
    return base + y * blend - t * 0.02 * blend       # лёгкий общий подъём в гору


# ---------- Байк (физика) ----------
class Bike:
    def __init__(self, space, x):
        self.space = space
        y = terrain_y(x) - 24 - WHEEL_R - 6
        mass_c = 4.0
        self.chassis = pymunk.Body(mass_c, pymunk.moment_for_box(mass_c, CHASSIS_SIZE))
        self.chassis.position = (x, y)
        box = pymunk.Poly.create_box(self.chassis, CHASSIS_SIZE)
        box.friction = 0.6
        box.filter = pymunk.ShapeFilter(group=1)
        items = [self.chassis, box]

        self.wheels = []
        for dx in (-36, 36):
            body = pymunk.Body(1.2, pymunk.moment_for_circle(1.2, 0, WHEEL_R))
            body.position = (x + dx, y + 24)
            shape = pymunk.Circle(body, WHEEL_R)
            shape.friction = 1.6
            shape.filter = pymunk.ShapeFilter(group=1)
            groove = pymunk.GrooveJoint(self.chassis, body, (dx, 8), (dx, 36), (0, 0))
            spring = pymunk.DampedSpring(self.chassis, body, (dx, 8), (0, 0), 16, 700, 45)
            items += [body, shape, groove, spring]
            self.wheels.append(body)
        self.rear, self.front = self.wheels
        space.add(*items)
        self.items = items

    def remove(self):
        self.space.remove(*self.items)

    def head_pos(self):
        return self.chassis.local_to_world(HEAD_LOCAL)


# ---------- Картинки (загружаются из папки assets) ----------
SLICE = 3                    # ширина вертикальных полосок земли
FAR_PARTS = ("thigh", "shin", "knee", "boot", "crank", "pedal")


class Art:
    """Загружает нарисованные спрайты и готовит их к быстрой отрисовке."""

    def __init__(self):
        MA.ensure_assets()

        def img(name, alpha=True):
            s = pygame.image.load(os.path.join(MA.OUT, name + ".png"))
            return s.convert_alpha() if alpha else s.convert()

        self.sky = img("sky", False)
        self.clouds = [img(f"cloud_{i}") for i in range(4)]
        self.mtn_far, self.mtn_mid, self.hills = img("mtn_far"), img("mtn_mid"), img("hills")
        self.fog = self.make_fog()

        dirt = img("dirt", False)
        self.dirt_slices = [dirt.subsurface((i * SLICE, 0, SLICE, dirt.get_height())).copy()
                            for i in range(dirt.get_width() // SLICE)]
        grass = img("grass")
        self.grass_slices = [grass.subsurface((i * SLICE, 0, SLICE, grass.get_height())).copy()
                             for i in range(grass.get_width() // SLICE)]
        self.grass_h = grass.get_height()

        self.rocks = [img(f"rock_{i}") for i in range(6)]
        self.trees = [[img(f"pine_{layer}_{h}") for h in (58, 78, 100, 126, 152)] for layer in (0, 1)]
        self.dead = [img(f"dead_{h}") for h in (70, 96, 124)]
        self.flags = {n: [img(f"flag_{n}_{f}") for f in range(4)] for n in ("red", "gold")}

        self.parts = {name: img(name) for name in MA.PARTS}
        self.far_parts = {}
        for name in FAR_PARTS:
            far = self.parts[name].copy()
            far.fill((118, 116, 124, 255), special_flags=pygame.BLEND_RGBA_MULT)
            self.far_parts[name] = far
        self.wheel_frames = self.bake_wheel(self.parts["wheel"])

        self.vignette = self.make_vignette()
        self.shadow = self.make_shadow()
        self.dust = self.make_dust()

    @staticmethod
    def bake_wheel(hi, steps=72):
        size = hi.get_width()
        out = []
        for i in range(steps):
            rot = pygame.transform.rotate(hi, -i * 360.0 / steps)
            rw, rh = rot.get_size()
            crop = rot.subsurface(((rw - size) // 2, (rh - size) // 2, size, size))
            out.append(pygame.transform.smoothscale(crop, (size // HS, size // HS)))
        return out

    @staticmethod
    def make_fog():
        surf = pygame.Surface((W, H), pygame.SRCALPHA)
        center, spread = 395, 150
        for y in range(H):
            d = abs(y - center) / spread
            a = int(80 * max(0.0, 1 - d) ** 1.4)
            if a > 0:
                pygame.draw.line(surf, (176, 140, 112, a), (0, y), (W, y))
        return surf

    @staticmethod
    def make_vignette():
        small = pygame.Surface((48, 27), pygame.SRCALPHA)
        for y in range(27):
            for x in range(48):
                nx = (x / 47 - 0.5) * 2
                ny = (y / 26 - 0.5) * 2
                d = math.hypot(nx, ny) / 1.414
                a = int(170 * max(0.0, (d - 0.38) / 0.62) ** 1.6)
                small.set_at((x, y), (6, 5, 8, min(255, a)))
        return pygame.transform.smoothscale(small, (W, H))

    @staticmethod
    def make_shadow():
        surf = pygame.Surface((130, 22), pygame.SRCALPHA)
        for k in range(11):
            rect = pygame.Rect(k * 5, k, 130 - k * 10, 22 - k * 2)
            pygame.draw.ellipse(surf, (0, 0, 0, 12 + k * 9), rect)
        return surf

    @staticmethod
    def make_dust():
        sizes = [4, 6, 8, 11, 14, 18]
        alphas = [14, 36, 60, 84, 108, 130]
        out = []
        for s in sizes:
            row = []
            for a in alphas:
                spr = pygame.Surface((2 * s, 2 * s), pygame.SRCALPHA)
                pygame.draw.circle(spr, (112, 94, 78, a // 3), (s, s), s)
                pygame.draw.circle(spr, (120, 102, 84, a * 2 // 3), (s, s), int(s * 0.7))
                pygame.draw.circle(spr, (132, 114, 94, a), (s, s), max(1, int(s * 0.4)))
                row.append(spr)
            out.append(row)
        return out


# ---------- Игра ----------
class Game:
    def __init__(self):
        pygame.init()
        if "ANDROID_ARGUMENT" in os.environ:
            flags = pygame.FULLSCREEN
        else:
            flags = pygame.SCALED | pygame.RESIZABLE
        self.screen = pygame.display.set_mode((W, H), flags)
        pygame.display.set_caption("Горный байкер")
        self.clock = pygame.time.Clock()
        names = "arial,dejavusans,liberationsans"
        self.font = pygame.font.SysFont(names, 22, bold=True)
        self.big = pygame.font.SysFont(names, 46, bold=True)
        self.huge = pygame.font.SysFont(names, 30, bold=True)
        self.small = pygame.font.SysFont(names, 15, bold=True)
        self.art = Art()
        self.hq = True
        self.time = 0.0
        self.fingers = {}
        self.best = 0.0
        self.total_flips = 0
        self.checkpoint_x = 150.0
        self.points = 0
        self.upgrades = {"speed": 0, "accel": 0, "gravity": 0}
        self.shop_open = False
        self.admin_open = False
        self.admin = {"gravity": None, "max_speed": None, "torque": None, "lean": None}
        self.new_run(150.0)

    # --- запуск/перезапуск ---
    def new_run(self, start_x):
        self.space = pymunk.Space()
        self.space.gravity = (0, self.eff_gravity())
        self.segments = []
        self.built_to = start_x - 600
        self.bike = Bike(self.space, start_x)
        self.start_x = 150.0
        self.cam = pygame.Vector2(start_x - W * 0.35, self.bike.chassis.position.y - H * 0.6)
        self.state = "play"
        self.in_air = False
        self.air_time = 0.0
        self.air_start_angle = 0.0
        self.flip_popup = ("", 0.0)
        self.particles = []
        self.rider_lean = 0.0
        self.last_dist = self.distance()
        self.build_terrain()

    def restart_from_checkpoint(self):
        self.new_run(self.checkpoint_x)

    def restart_all(self):
        self.checkpoint_x = 150.0
        self.total_flips = 0
        self.new_run(150.0)

    # --- рельеф ---
    def build_terrain(self):
        target = self.bike.chassis.position.x + 1600
        while self.built_to < target:
            x0, x1 = self.built_to, self.built_to + SEGMENT
            seg = pymunk.Segment(
                self.space.static_body, (x0, terrain_y(x0)), (x1, terrain_y(x1)), 3
            )
            seg.friction = 1.3
            self.space.add(seg)
            self.segments.append((x1, seg))
            self.built_to = x1
        limit = self.bike.chassis.position.x - 1500
        while self.segments and self.segments[0][0] < limit:
            self.space.remove(self.segments.pop(0)[1])

    # --- физика ---
    def wheel_on_ground(self, w):
        return terrain_y(w.position.x) - (w.position.y + WHEEL_R) < 10

    def spawn_dust(self, x, y, vx, count=1, power=1.0):
        if not self.hq or len(self.particles) > 140:
            return
        for _ in range(count):
            life = random.uniform(0.45, 0.9)
            self.particles.append([
                x + random.uniform(-4, 4), y + random.uniform(-3, 1),
                vx + random.uniform(-45, 15) * power, random.uniform(-80, -15) * power,
                life, life, random.randint(0, 5),
            ])

    def step(self, gas, brake, lean_back, lean_fwd, dt=1.0 / FPS):
        if self.state != "play":
            return
        b = self.bike
        sub = dt / SUBSTEPS
        for _ in range(SUBSTEPS):
            if gas and b.rear.angular_velocity < self.max_wheel_speed():
                tq = self.engine_torque()
                b.rear.torque = tq
                b.chassis.torque -= tq * 0.3
            if brake:
                b.rear.angular_velocity *= 0.90
                b.front.angular_velocity *= 0.90
            if lean_back:
                b.chassis.torque -= self.lean_torque()
            if lean_fwd:
                b.chassis.torque += self.lean_torque()
            self.space.step(sub)

        # поза гонщика
        target = (1.0 if lean_fwd else 0.0) - (1.0 if lean_back else 0.0)
        self.rider_lean += (target - self.rider_lean) * min(1.0, 8 * dt)

        # Падение: голова коснулась земли
        hx, hy = b.head_pos()
        if hy + 9 > terrain_y(hx):
            self.crash()
            return

        # Пыль из-под колёс
        for wheel, chance in ((b.rear, 1.0), (b.front, 0.35)):
            if self.wheel_on_ground(wheel):
                speed = abs(wheel.angular_velocity)
                if speed > 12 and random.random() < min(1.0, speed / 45) * chance:
                    self.spawn_dust(
                        wheel.position.x - 4, wheel.position.y + WHEEL_R - 2,
                        -b.chassis.velocity.x * 0.12,
                    )

        # Воздух и кувырки
        grounded = self.wheel_on_ground(b.rear) or self.wheel_on_ground(b.front)
        if not grounded and not self.in_air:
            self.in_air = True
            self.air_time = 0.0
            self.air_start_angle = b.chassis.angle
        elif not grounded:
            self.air_time += dt
        elif self.in_air:
            self.in_air = False
            delta = abs(b.chassis.angle - self.air_start_angle)
            flips = int((delta + 1.2) // (2 * math.pi))
            if flips >= 1 and self.air_time > 0.3:
                self.total_flips += flips
                self.points += flips * FLIP_BONUS
                self.flip_popup = (f"Кувырок x{flips}!  +{flips * FLIP_BONUS}", 1.5)
            if self.air_time > 0.4:
                for wheel in (b.rear, b.front):
                    self.spawn_dust(wheel.position.x, wheel.position.y + WHEEL_R, 0, 8, 1.6)

        # Очки за дистанцию и контрольные точки
        d = self.distance()
        self.points += (d - self.last_dist) * DIST_POINTS_PER_M
        self.last_dist = d
        x = b.chassis.position.x
        if x > self.checkpoint_x + CHECKPOINT_EVERY:
            self.checkpoint_x += CHECKPOINT_EVERY * ((x - self.checkpoint_x) // CHECKPOINT_EVERY)
            self.points += CHECKPOINT_BONUS
        self.best = max(self.best, d)

        self.build_terrain()

    def crash(self):
        self.state = "crashed"
        hx, hy = self.bike.head_pos()
        self.spawn_dust(hx, hy + 6, 0, 12, 1.8)

    def distance(self):
        return max(0.0, (self.bike.chassis.position.x - self.start_x) / PX_PER_METER)

    def speed_kmh(self):
        return self.bike.chassis.velocity.length / PX_PER_METER * 3.6

    # --- улучшения ---
    def max_wheel_speed(self):
        if self.admin["max_speed"] is not None:
            return self.admin["max_speed"]
        return WHEEL_SPEED_BASE + self.upgrades["speed"] * SPEED_PER_LVL

    def engine_torque(self):
        if self.admin["torque"] is not None:
            return self.admin["torque"]
        return TORQUE_BASE + self.upgrades["accel"] * ACCEL_PER_LVL

    def eff_gravity(self):
        if self.admin["gravity"] is not None:
            return self.admin["gravity"]
        return GRAVITY_BASE + (GRAVITY_CUT if self.upgrades["gravity"] else 0)

    def lean_torque(self):
        if self.admin["lean"] is not None:
            return self.admin["lean"]
        return LEAN_TORQUE

    def upgrade_cost(self, key):
        if key == "gravity":
            return GRAVITY_COST
        if key == "speed":
            return SPEED_COST + self.upgrades["speed"] * 12
        return ACCEL_COST + self.upgrades["accel"] * 12

    def can_upgrade(self, key):
        if key == "gravity":
            return self.upgrades["gravity"] == 0 and self.points >= GRAVITY_COST
        limit = SPEED_MAX if key == "speed" else ACCEL_MAX
        return self.upgrades[key] < limit and self.points >= self.upgrade_cost(key)

    def buy(self, key):
        if not self.can_upgrade(key):
            return
        self.points -= self.upgrade_cost(key)
        self.upgrades[key] += 1

    def shop_items(self):
        return [
            ("speed", "Скорость",
             f"Макс. скорость колеса +{SPEED_PER_LVL}",
             f"ур. {self.upgrades['speed']}/{SPEED_MAX}"),
            ("accel", "Ускорение",
             f"Сила двигателя +{ACCEL_PER_LVL // 1000} тыс.",
             f"ур. {self.upgrades['accel']}/{ACCEL_MAX}"),
            ("gravity", "Гравитация",
             "1100 → 880, разовая",
             "★ куплено" if self.upgrades["gravity"] else "не куплено"),
        ]

    def update_particles(self, dt):
        for p in self.particles:
            p[0] += p[2] * dt
            p[1] += p[3] * dt
            p[2] *= 0.97
            p[3] *= 0.96
            p[4] -= dt
        self.particles = [p for p in self.particles if p[4] > 0]

    # --- ввод ---
    def buttons(self):
        return {
            "lean_back": pygame.Rect(20, H - 110, 84, 84),
            "lean_fwd": pygame.Rect(114, H - 110, 84, 84),
            "brake": pygame.Rect(W - 214, H - 110, 94, 84),
            "gas": pygame.Rect(W - 110, H - 110, 94, 84),
        }

    def touch_points(self):
        points = list(self.fingers.values())
        if pygame.mouse.get_pressed()[0]:
            points.append(pygame.mouse.get_pos())
        return points

    def read_inputs(self):
        k = pygame.key.get_pressed()
        gas = k[pygame.K_RIGHT] or k[pygame.K_d]
        brake = k[pygame.K_LEFT] or k[pygame.K_a]
        lean_back = k[pygame.K_UP] or k[pygame.K_w]
        lean_fwd = k[pygame.K_DOWN] or k[pygame.K_s]
        rects = self.buttons()
        for p in self.touch_points():
            if rects["gas"].collidepoint(p):
                gas = True
            if rects["brake"].collidepoint(p):
                brake = True
            if rects["lean_back"].collidepoint(p):
                lean_back = True
            if rects["lean_fwd"].collidepoint(p):
                lean_fwd = True
        return gas, brake, lean_back, lean_fwd

    # ================= ОТРИСОВКА =================
    def sx(self, x):
        return x - self.cam.x

    def sy(self, y):
        return y - self.cam.y

    # --- фон ---
    def draw_tile(self, img, factor, y):
        w = img.get_width()
        x = -((self.cam.x * factor) % w)
        self.screen.blit(img, (x, y))
        if x + w < W:
            self.screen.blit(img, (x + w, y))

    def draw_background(self):
        art = self.art
        self.screen.blit(art.sky, (0, 0))
        if self.hq:
            span = W + 700
            for i, c in enumerate(art.clouds):
                x = (i * 410 - self.cam.x * 0.04 - self.time * (3 + i * 1.2)) % span - 360
                y = 20 + (i * 71) % 130 - self.cam.y * 0.01
                self.screen.blit(c, (x, y))
        dy = -self.cam.y * 0.35
        self.draw_tile(art.mtn_far, 0.06, 6 + dy * 0.08)
        self.draw_tile(art.mtn_mid, 0.14, 62 + dy * 0.16)
        if self.hq:
            self.screen.blit(art.fog, (0, 0))
        hy = 186 + dy * 0.30
        self.draw_tile(art.hills, 0.30, hy)
        pygame.draw.rect(self.screen, (13, 17, 19), (0, hy + art.hills.get_height() - 1, W, H))

    def draw_trees(self):
        art = self.art
        slot = 120
        k0 = int((self.cam.x - 200) // slot)
        k1 = int((self.cam.x + W + 200) // slot) + 1
        for layer in (0, 1):
            sprites = art.trees[layer]
            for k in range(k0, k1):
                if k * slot < 900:
                    continue
                if hrand(k, 10 + layer) > (0.34 if layer == 0 else 0.30):
                    continue
                wx = k * slot + hrand(k, 20 + layer) * slot
                if layer == 1 and hrand(k, 200) < 0.3:
                    spr = art.dead[int(hrand(k, 201) * 3) % 3]
                else:
                    spr = sprites[int(hrand(k, 30 + layer) * len(sprites)) % len(sprites)]
                px = self.sx(wx) - spr.get_width() / 2
                py = self.sy(terrain_y(wx)) - spr.get_height() + 8
                self.screen.blit(spr, (px, py))

    # --- земля ---
    def draw_terrain(self):
        art = self.art
        n_d, n_g = len(art.dirt_slices), len(art.grass_slices)
        x0 = int(self.cam.x // SLICE) * SLICE - SLICE
        cam_x, cam_y = self.cam.x, self.cam.y
        screen = self.screen
        cols = []
        for wx in range(x0, int(cam_x) + W + 2 * SLICE, SLICE):
            sxp = int(round(wx - cam_x))
            ty = int(round(terrain_y(wx + 1.5) - cam_y))
            cols.append((wx, sxp, ty))
            screen.blit(art.dirt_slices[(wx // SLICE) % n_d], (sxp, ty))

        # валуны на поверхности (они стоят на земле, до травы)
        bslot = 150
        for k in range(int(cam_x // bslot) - 1, int((cam_x + W) // bslot) + 2):
            if hrand(k, 90) > 0.55 or k * bslot < 700:
                continue
            wx = k * bslot + hrand(k, 91) * bslot
            spr = art.rocks[int(hrand(k, 92) * 6) % 6]
            screen.blit(spr, (self.sx(wx) - spr.get_width() / 2, self.sy(terrain_y(wx)) - spr.get_height() + 10))

        gy = art.grass_h - 8
        for wx, sxp, ty in cols:
            screen.blit(art.grass_slices[(wx // SLICE) % n_g], (sxp, ty - gy + 2))

    def draw_foreground_grass(self):
        if not self.hq:
            return
        art = self.art
        n_g = len(art.grass_slices)
        gy = art.grass_h - 8
        for wx in range(int(self.cam.x // 12) * 12, int(self.cam.x) + W + 12, 12):
            r = hrand(wx // 12, 100)
            if r > 0.32:
                continue
            ty = int(round(terrain_y(wx) - self.cam.y))
            sxp = int(round(wx - self.cam.x))
            # короткий пучок травы перед байком: берём 4 полоски и опускаем чуть ниже
            for j in range(4):
                self.screen.blit(art.grass_slices[((wx // SLICE) + j) % n_g], (sxp + j * SLICE, ty - gy + 12))

    def draw_checkpoints(self):
        art = self.art
        first = int((self.cam.x - self.start_x) // CHECKPOINT_EVERY)
        frame = int(self.time * 7) % 4
        for i in range(max(1, first), first + 3):
            wx = self.start_x + i * CHECKPOINT_EVERY
            if wx < self.cam.x - 100 or wx > self.cam.x + W + 100:
                continue
            reached = wx <= self.checkpoint_x
            spr = art.flags["gold" if reached else "red"][frame]
            self.screen.blit(spr, (self.sx(wx) - 7, self.sy(terrain_y(wx)) - spr.get_height() + 6))

    # --- байк и гонщик из нарисованных деталей ---
    def draw_part(self, name, pos, angle, length=None, far=False):
        meta = MA.PARTS[name]
        hi = (self.art.far_parts if far and name in self.art.far_parts else self.art.parts)[name]
        px, py = meta["pivot"][0] * HS, meta["pivot"][1] * HS
        if length is not None:
            f = length / meta["nominal"]
            if abs(f - 1) > 0.02:
                hi = pygame.transform.smoothscale(hi, (max(2, int(hi.get_width() * f)), hi.get_height()))
                px *= f
        rot = pygame.transform.rotate(hi, -math.degrees(angle))
        vx, vy = px - hi.get_width() / 2, py - hi.get_height() / 2
        c, s = math.cos(angle), math.sin(angle)
        rx, ry = vx * c - vy * s, vx * s + vy * c
        sw = max(1, round(rot.get_width() / HS))
        sh = max(1, round(rot.get_height() / HS))
        small = pygame.transform.smoothscale(rot, (sw, sh))
        self.screen.blit(small, (round(pos[0] - rx / HS - sw / 2), round(pos[1] - ry / HS - sh / 2)))

    def bone(self, name, a, b, far=False):
        ang = math.atan2(b[1] - a[1], b[0] - a[0])
        self.draw_part(name, a, ang, length=math.hypot(b[0] - a[0], b[1] - a[1]), far=far)
        return ang

    def fork_compression(self):
        """Сжатие вилки 0..1: насколько переднее колесо ушло вверх к раме."""
        b = self.bike
        local_front = b.chassis.world_to_local(b.front.position)
        d_now = math.hypot(local_front[0] - HEAD_LOCAL[0], local_front[1] - HEAD_LOCAL[1])
        return max(0.0, min(1.0, (FORK_SETTLED - d_now) / 10.0))

    def draw_bike(self):
        b = self.bike
        ch = b.chassis
        s = self.screen
        art = self.art
        a0 = ch.angle

        def P(local):
            wx, wy = ch.local_to_world(local)
            return (self.sx(wx), self.sy(wy))

        def blit_wheel(body):
            idx = int((body.angle % (2 * math.pi)) / (2 * math.pi) * len(art.wheel_frames)) % len(art.wheel_frames)
            fr = art.wheel_frames[idx]
            s.blit(fr, (round(self.sx(body.position.x) - fr.get_width() / 2),
                        round(self.sy(body.position.y) - fr.get_height() / 2)))

        # тень на земле
        ground = terrain_y(ch.position.x)
        height = max(0.0, ground - (ch.position.y + 41))
        alpha = int(max(0, 230 - height * 2.4))
        if alpha > 4:
            art.shadow.set_alpha(alpha)
            s.blit(art.shadow, (self.sx(ch.position.x) - 65, self.sy(ground) - 6))

        rear = (self.sx(b.rear.position.x), self.sy(b.rear.position.y))
        front = (self.sx(b.front.position.x), self.sy(b.front.position.y))

        bb, head_t, seat_top = (-2, 12), (24, -9), (-18, -14)
        crank = b.rear.angle * 0.5
        ped_n = (bb[0] + 9 * math.cos(crank), bb[1] + 9 * math.sin(crank))
        ped_f = (bb[0] - 9 * math.cos(crank), bb[1] - 9 * math.sin(crank))
        lean = self.rider_lean
        hip_l = (-14, -14)
        sho_l = (6 + 5 * lean, -38 - 1 * abs(lean))
        head_l = (sho_l[0] + 2, sho_l[1] - 11)
        grip_l = (31, -23)
        bbp, headp, seatp = P(bb), P(head_t), P(seat_top)

        # --- дальняя нога ---
        hip = P(hip_l)
        knee_f, foot_f = ik(hip, P(ped_f), 25, 26, False)
        self.draw_part("crank", bbp, math.atan2(P(ped_f)[1] - bbp[1], P(ped_f)[0] - bbp[0]), 9, far=True)
        self.draw_part("pedal", P(ped_f), a0, far=True)
        ang = self.bone("thigh", hip, knee_f, far=True)
        self.draw_part("knee", knee_f, ang, far=True)
        self.bone("shin", knee_f, foot_f, far=True)
        self.draw_part("boot", foot_f, a0, far=True)

        # --- колёса и подвеска ---
        blit_wheel(b.rear)
        blit_wheel(b.front)
        self.bone("seatstay", seatp, rear)
        self.bone("rear_arm", bbp, rear)
        # цепь
        pygame.draw.line(s, (58, 52, 46), P((bb[0], bb[1] - 7.5)), (rear[0], rear[1] - 3), 1)
        pygame.draw.line(s, (58, 52, 46), P((bb[0], bb[1] + 7.5)), (rear[0], rear[1] + 3), 1)
        pygame.draw.circle(s, (150, 150, 158), (round(rear[0]), round(rear[1])), 3, 1)

        # --- рама, амортизатор, вилка ---
        self.draw_part("frame", P((0, 0)), a0)
        piv = (bbp[0] + (rear[0] - bbp[0]) * 0.5, bbp[1] + (rear[1] - bbp[1]) * 0.5)
        self.bone("shock", P((-9, -11)), piv)
        comp = self.fork_compression()
        dx, dy = front[0] - headp[0], front[1] - headp[1]
        dist = math.hypot(dx, dy) or 1.0
        ux, uy = dx / dist, dy / dist
        fang = math.atan2(dy, dx)
        self.draw_part("fork_up", headp, fang)
        low_top = (front[0] - ux * 24, front[1] - uy * 24)
        self.draw_part("fork_low", low_top, fang)

        # --- ближняя нога ---
        self.draw_part("crank", bbp, math.atan2(P(ped_n)[1] - bbp[1], P(ped_n)[0] - bbp[0]), 9)
        self.draw_part("pedal", P(ped_n), a0)
        knee_n, foot_n = ik(hip, P(ped_n), 25, 26, False)
        ang = self.bone("thigh", hip, knee_n)
        self.bone("shin", knee_n, foot_n)
        self.draw_part("knee", knee_n, ang)
        self.draw_part("boot", foot_n, a0)

        # --- торс, рука, голова ---
        sh = P(sho_l)
        self.bone("torso", hip, sh)
        elbow, hand = ik(sh, P(grip_l), 21, 22, True)
        self.bone("upper_arm", sh, elbow)
        fa = self.bone("forearm", elbow, hand)
        self.draw_part("elbow", elbow, fa)
        self.draw_part("glove", hand, fa)
        self.draw_part("head", P(head_l), a0 + lean * 0.06)

    # --- интерфейс ---
    def draw_button(self, rect, label, kind):
        pressed = any(rect.collidepoint(p) for p in self.touch_points())
        surf = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(surf, (10, 10, 12, 205 if pressed else 140), surf.get_rect(), border_radius=14)
        pygame.draw.rect(surf, AMBER + (230 if pressed else 120,), surf.get_rect(), 2, border_radius=14)
        self.screen.blit(surf, rect.topleft)
        cx, cy = rect.center
        if kind == "back":
            pygame.draw.polygon(self.screen, UI_TEXT, [(cx - 14, cy + 8), (cx + 14, cy + 8), (cx, cy - 14)])
        elif kind == "fwd":
            pygame.draw.polygon(self.screen, UI_TEXT, [(cx - 14, cy - 8), (cx + 14, cy - 8), (cx, cy + 14)])
        if kind == "text":
            big = self.font.render(label, True, AMBER if pressed else UI_TEXT)
            self.screen.blit(big, (cx - big.get_width() // 2, cy - big.get_height() // 2))
        else:
            txt = self.small.render(label, True, UI_TEXT)
            self.screen.blit(txt, (cx - txt.get_width() // 2, rect.bottom - 22))

    def panel(self, rect):
        surf = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(surf, (8, 8, 10, 175), surf.get_rect(), border_radius=10)
        pygame.draw.rect(surf, AMBER + (90,), surf.get_rect(), 1, border_radius=10)
        pygame.draw.rect(surf, AMBER + (200,), (0, 8, 3, rect.height - 16))
        self.screen.blit(surf, rect.topleft)

    def text(self, txt, font, pos, color=UI_TEXT, anchor="left"):
        img = font.render(txt, True, color)
        shadow = font.render(txt, True, (0, 0, 0))
        x, y = pos
        if anchor == "right":
            x -= img.get_width()
        elif anchor == "center":
            x -= img.get_width() // 2
        self.screen.blit(shadow, (x + 1, y + 2))
        self.screen.blit(img, (x, y))

    def draw_particles(self):
        for p in self.particles:
            ai = min(5, int(p[4] / p[5] * 6))
            spr = self.art.dust[p[6]][ai]
            self.screen.blit(spr, (self.sx(p[0]) - spr.get_width() // 2, self.sy(p[1]) - spr.get_height() // 2))

    # --- магазин ---
    def draw_shop(self):
        if not self.shop_open:
            return
        ov = pygame.Surface((W, H), pygame.SRCALPHA)
        ov.fill((6, 6, 8, 130))
        self.screen.blit(ov, (0, 0))
        rect = pygame.Rect(W // 2 - 280, 80, 560, 340)
        self.panel(rect)
        self.text("МАГАЗИН", self.huge, (rect.x + 20, rect.y + 12), AMBER)
        self.text(f"Очки: {self.points}", self.font, (rect.right - 20, rect.y + 16), AMBER, "right")
        y = rect.y + 64
        self.shop_btns = {}
        for key, name, effect, status in self.shop_items():
            row = pygame.Rect(rect.x + 20, y, rect.width - 40, 60)
            pygame.draw.rect(self.screen, (20, 22, 28), row, border_radius=8)
            pygame.draw.rect(self.screen, AMBER + (70,), row, 1, border_radius=8)
            self.text(name, self.font, (row.x + 14, row.y + 8), UI_TEXT)
            self.text(f"{effect}   {status}" if status else effect, self.small,
                      (row.x + 14, row.y + 34), (184, 178, 168))
            buy = pygame.Rect(row.right - 116, row.y + 14, 102, 32)
            self.shop_btns[key] = buy
            if key == "gravity" and self.upgrades["gravity"]:
                label, can = "куплено", False
            elif key in ("speed", "accel") and self.upgrades[key] >= SPEED_MAX:
                label, can = "MAX", False
            else:
                label, can = f"{self.upgrade_cost(key)} оч.", self.points >= self.upgrade_cost(key)
            if can:
                pygame.draw.rect(self.screen, (52, 60, 44), buy, border_radius=7)
                pygame.draw.rect(self.screen, (110, 160, 80), buy, 1, border_radius=7)
            else:
                pygame.draw.rect(self.screen, (44, 44, 50), buy, border_radius=7)
            self.text(label, self.small, buy.center, AMBER if can else (150, 150, 158), "center")
            y += 68
        self.text("Кликните «Магазин» или M для закрытия", self.small,
                  (rect.x + 20, rect.bottom - 26), (140, 136, 130))

    def process_shop_tap(self, p):
        for key, rect in getattr(self, "shop_btns", {}).items():
            if rect.collidepoint(p):
                self.buy(key)
                return
        panel = pygame.Rect(W // 2 - 280, 80, 560, 340)
        if not panel.collidepoint(p):
            self.shop_open = False

    # --- админ-панель ---
    def admin_adjust(self, key, delta):
        if key == "points":
            self.points = max(0, self.points + delta)
            return
        cur = {"gravity": self.eff_gravity(), "max_speed": self.max_wheel_speed(),
               "torque": self.engine_torque(), "lean": self.lean_torque()}[key]
        mmin = {"gravity": ADMIN_G_MIN, "max_speed": ADMIN_SPEED_MIN,
                "torque": ADMIN_TORQUE_MIN, "lean": ADMIN_LEAN_MIN}[key]
        step = {"gravity": ADMIN_G_STEP, "max_speed": ADMIN_SPEED_STEP,
                "torque": ADMIN_TORQUE_STEP, "lean": ADMIN_LEAN_STEP}[key]
        self.admin[key] = max(mmin, cur + delta)
        if key == "gravity" and self.space is not None:
            self.space.gravity = (0, self.admin["gravity"])

    def admin_reset(self):
        self.admin = {"gravity": None, "max_speed": None, "torque": None, "lean": None}
        if self.space is not None:
            self.space.gravity = (0, self.eff_gravity())

    def draw_admin(self):
        if not self.admin_open:
            return
        ov = pygame.Surface((W, H), pygame.SRCALPHA)
        ov.fill((6, 6, 8, 140))
        self.screen.blit(ov, (0, 0))
        rect = pygame.Rect(W // 2 - 300, 50, 600, 420)
        self.panel(rect)
        self.text("АДМИН-ПАНЕЛЬ (F1)", self.huge, (rect.x + 20, rect.y + 12), (255, 120, 90))
        items = [
            ("gravity", "Гравитация", f"{self.eff_gravity():.0f}"),
            ("max_speed", "Макс. скорость", f"{self.max_wheel_speed():.0f}"),
            ("torque", "Ускорение (момент)", f"{self.engine_torque():.0f}"),
            ("lean", "Наклон (LEAN)", f"{self.lean_torque():.0f}"),
            ("points", "Очки", f"{self.points}"),
        ]
        y = rect.y + 62
        self.admin_btns = {}
        for key, name, val in items:
            row = pygame.Rect(rect.x + 20, y, rect.width - 40, 52)
            pygame.draw.rect(self.screen, (20, 22, 28), row, border_radius=8)
            pygame.draw.rect(self.screen, (120, 70, 60), row, 1, border_radius=8)
            self.text(name, self.font, (row.x + 14, row.y + 12), UI_TEXT)
            self.text(val, self.font, (row.x + 250, row.y + 12), AMBER)
            inc = pygame.Rect(row.right - 74, row.y + 10, 60, 32)
            dec = pygame.Rect(row.right - 142, row.y + 10, 60, 32)
            self.admin_btns[(key, "inc")] = inc
            self.admin_btns[(key, "dec")] = dec
            for r, mark in ((dec, "-"), (inc, "+")):
                pygame.draw.rect(self.screen, (46, 46, 54), r, border_radius=7)
                self.text(mark, self.big, r.center, (240, 172, 60), "center")
            y += 60
        self.admin_reset_rect = pygame.Rect(rect.x + 20, y + 6, rect.width - 40, 44)
        pygame.draw.rect(self.screen, (70, 40, 40), self.admin_reset_rect, border_radius=8)
        pygame.draw.rect(self.screen, (200, 90, 70), self.admin_reset_rect, 1, border_radius=8)
        self.text("СБРОСИТЬ НАСТРОЙКИ", self.font, self.admin_reset_rect.center, (240, 200, 180), "center")

    def process_admin_tap(self, p):
        for (key, op), rect in getattr(self, "admin_btns", {}).items():
            if rect.collidepoint(p):
                delta = ADMIN_POINTS_STEP if key == "points" else 0
                if key != "points":
                    delta = {"gravity": ADMIN_G_STEP, "max_speed": ADMIN_SPEED_STEP,
                             "torque": ADMIN_TORQUE_STEP, "lean": ADMIN_LEAN_STEP}[key]
                self.admin_adjust(key, delta if op == "inc" else -delta)
                return
        if getattr(self, "admin_reset_rect", None) and self.admin_reset_rect.collidepoint(p):
            self.admin_reset()
            return
        panel = pygame.Rect(W // 2 - 300, 50, 600, 420)
        if not panel.collidepoint(p):
            self.admin_open = False

    def draw(self, dt):
        self.time += dt
        self.update_particles(dt)

        # камера с упреждением вперёд
        vx = self.bike.chassis.velocity.x
        target = pygame.Vector2(
            self.bike.chassis.position.x - W * 0.35 + max(-60, min(140, vx * 0.12)),
            self.bike.chassis.position.y - H * 0.6,
        )
        self.cam += (target - self.cam) * min(1.0, 6 * dt)

        self.draw_background()
        self.draw_trees()
        self.draw_terrain()
        self.draw_checkpoints()
        self.draw_particles()
        self.draw_bike()
        self.draw_foreground_grass()
        if self.hq:
            self.screen.blit(self.art.vignette, (0, 0))

        # HUD
        self.panel(pygame.Rect(10, 8, 270, 58))
        self.text(f"{self.distance():.0f} м", self.huge, (24, 12), AMBER)
        self.text(f"{self.speed_kmh():.0f} км/ч", self.small, (26, 46), (190, 184, 172))
        self.text(f"Очки: {self.points}", self.small, (26, 72), AMBER)

        self.shop_btn = pygame.Rect(W // 2 - 66, 8, 132, 34)
        self.draw_button(self.shop_btn, "Магазин (M)", "text")

        self.admin_btn = pygame.Rect(W // 2 + 70, 8, 120, 34)
        self.draw_button(self.admin_btn, "Админ (F1)", "text")

        self.panel(pygame.Rect(W - 250, 8, 240, 58))
        self.text(f"Кувырки: {self.total_flips}", self.font, (W - 22, 12), anchor="right")
        self.text(f"Рекорд: {self.best:.0f} м", self.small, (W - 22, 44), AMBER, anchor="right")

        text, t = self.flip_popup
        if t > 0:
            self.text(text, self.big, (W // 2, 92), AMBER, "center")
            self.flip_popup = (text, t - dt)

        rects = self.buttons()
        self.draw_button(rects["lean_back"], "назад", "back")
        self.draw_button(rects["lean_fwd"], "вперёд", "fwd")
        self.draw_button(rects["brake"], "тормоз", "text")
        self.draw_button(rects["gas"], "ГАЗ", "text")

        if self.state == "crashed":
            ov = pygame.Surface((W, H), pygame.SRCALPHA)
            ov.fill((24, 0, 0, 165))
            self.screen.blit(ov, (0, 0))
            self.text("Упал!", self.big, (W // 2, H // 2 - 100), (214, 70, 50), "center")
            self.text(f"Дистанция: {self.distance():.0f} м", self.font, (W // 2, H // 2 - 40), anchor="center")
            self.text("Тап / R — с контрольной точки", self.font, (W // 2, H // 2), anchor="center")
            self.text("N — заново с начала", self.font, (W // 2, H // 2 + 34), anchor="center")

        self.draw_shop()
        self.draw_admin()

        pygame.display.flip()

    # --- главный цикл ---
    def run(self):
        while True:
            dt = min(self.clock.tick(FPS) / 1000.0, 1 / 30)
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()
                    if e.key == pygame.K_F2:
                        self.hq = not self.hq
                    if e.key == pygame.K_m:
                        self.shop_open = not self.shop_open
                        self.admin_open = False
                    if e.key == pygame.K_F1:
                        self.shop_open = False
                        self.admin_open = not self.admin_open
                    if self.state == "crashed":
                        if e.key == pygame.K_r:
                            self.restart_from_checkpoint()
                        elif e.key == pygame.K_n:
                            self.restart_all()
                elif e.type == pygame.FINGERDOWN:
                    p = (e.x * W, e.y * H)
                    self.fingers[e.finger_id] = p
                    if self.admin_open:
                        self.process_admin_tap(p)
                    elif self.shop_open:
                        self.process_shop_tap(p)
                    elif self.state == "crashed":
                        self.restart_from_checkpoint()
                    elif getattr(self, "admin_btn", None) and self.admin_btn.collidepoint(p):
                        self.shop_open = False
                        self.admin_open = True
                    elif getattr(self, "shop_btn", None) and self.shop_btn.collidepoint(p):
                        self.shop_open = True
                elif e.type == pygame.FINGERMOTION:
                    self.fingers[e.finger_id] = (e.x * W, e.y * H)
                elif e.type == pygame.FINGERUP:
                    self.fingers.pop(e.finger_id, None)
                elif e.type == pygame.MOUSEBUTTONDOWN:
                    p = e.pos
                    if self.admin_open:
                        self.process_admin_tap(p)
                    elif self.shop_open:
                        self.process_shop_tap(p)
                    elif self.state == "crashed":
                        self.restart_from_checkpoint()
                    elif getattr(self, "admin_btn", None) and self.admin_btn.collidepoint(p):
                        self.shop_open = False
                        self.admin_open = True
                    elif getattr(self, "shop_btn", None) and self.shop_btn.collidepoint(p):
                        self.shop_open = True

            if not self.shop_open and not self.admin_open:
                self.step(*self.read_inputs(), dt=1.0 / FPS)
            self.draw(dt)


if __name__ == "__main__":
    Game().run()
