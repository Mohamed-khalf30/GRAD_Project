"""
carla_raw_imu.py  (v3)
-----------------------
تحكم:
  W/UP    throttle
  S/DN    brake
  A/LT    steer left
  D/RT    steer right
  SPACE   handbrake
  R       save + respawn (محظور لو الموتوسيكل وقع)
  ESC     quit

اعدادات الوقت:
  - مدة الـ run:        20 ثانية
  - warmup:             اول 5 ثواني مش بتتسجل
  - pre-collision buf:  آخر 5 ثواني قبل الاصطدام
  - post-collision buf: 2 ثانية بعد الاصطدام

CSV columns:
  run_id, timestamp_ms,
  acc_x, acc_y, acc_z,
  gyro_x, gyro_y, gyro_z,
  label,        (0=normal / 1=collision)
  severity      (none / minor / moderate / severe)
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import math
import carla
import csv
import os
import random
import numpy as np
from collections import deque
import pygame


OUTPUT_DIR         = "carla_raw_imu"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOTAL_RUNS         = 200
RUN_DURATION       = 20.0
FIXED_DELTA        = 0.05

WARMUP_SEC         = 5.0
PRE_COLLISION_SEC  = 5.0
POST_COLLISION_SEC = 2.0
SAMPLE_RATE        = int(1.0 / FIXED_DELTA)
PRE_SAMPLES        = int(PRE_COLLISION_SEC  * SAMPLE_RATE)
POST_SAMPLES       = int(POST_COLLISION_SEC * SAMPLE_RATE)


NO_COL_SAMPLES     = PRE_SAMPLES + POST_SAMPLES             # 140

NUM_NPC_VEHICLES   = 10
NUM_NPC_WALKERS    = 5

CAM_W    = 640
CAM_H    = 360
PANEL_W  = 380
SCREEN_W = CAM_W + PANEL_W
SCREEN_H = CAM_H

RENDER_EVERY = 3

MOTORCYCLE_MODELS = [
    "vehicle.kawasaki.ninja",
    "vehicle.yamaha.yzf",
    "vehicle.harley-davidson.low_rider",
]

# عتبات الـ impulse magnitude لتصنيف الخطورة
IMPULSE_MINOR    = 500
IMPULSE_MODERATE = 2000

CSV_HEADER = [
    "run_id", "timestamp_ms",
    "acc_x", "acc_y", "acc_z",
    "gyro_x", "gyro_y", "gyro_z",
    "label",
]

# ══════════════════════════════════════════
#  CARLA init
# ══════════════════════════════════════════
client = carla.Client("localhost", 2000)
client.set_timeout(60.0)
world  = client.get_world()

settings = world.get_settings()
settings.synchronous_mode    = True
settings.fixed_delta_seconds = FIXED_DELTA
world.apply_settings(settings)

blueprint_library = world.get_blueprint_library()
traffic_manager   = client.get_trafficmanager(8000)
traffic_manager.set_synchronous_mode(True)
traffic_manager.set_global_distance_to_leading_vehicle(2.0)
traffic_manager.global_percentage_speed_difference(0)
traffic_manager.set_hybrid_physics_mode(True)      # الـ NPCs البعيدة مش بتتحسب physics
traffic_manager.set_hybrid_physics_radius(50.0)    # نصف قطر 50 متر حواليك بس

_stats = {"total": 0, "collision": 0,}


def cleanup():
    s = world.get_settings()
    s.synchronous_mode    = False
    s.fixed_delta_seconds = None
    world.apply_settings(s)
    traffic_manager.set_synchronous_mode(False)


# ══════════════════════════════════════════
#  Pygame
# ══════════════════════════════════════════
pygame.init()
screen   = pygame.display.set_mode((SCREEN_W, SCREEN_H))
pygame.display.set_caption("CARLA IMU Recorder v3")
clock_pg = pygame.time.Clock()

try:
    F_TITLE = pygame.font.SysFont("Courier New", 16, bold=True)
    F_LBL   = pygame.font.SysFont("Courier New", 13, bold=True)
    F_VAL   = pygame.font.SysFont("Courier New", 13)
    F_MINI  = pygame.font.SysFont("Courier New", 11)
    F_BIG   = pygame.font.SysFont("Courier New", 26, bold=True)
except Exception:
    F_TITLE = F_LBL = F_VAL = F_MINI = F_BIG = pygame.font.Font(None, 16)

BG     = (8,   10,  16)
PANEL  = (13,  17,  26)
BORDER = (30,  40,  58)
TXT    = (195, 210, 230)
DIM    = (80,  95,  115)
CYAN   = (0,   195, 255)
PURPLE = (175,  80, 255)
GREEN  = (45,  225, 100)
RED    = (255,  55,  55)
ORANGE = (255, 155,   0)
YELLOW = (255, 230,   0)

cam_surf  = None
cam_ready = False

# لون وعنوان كل مستوى خطورة
SEVERITY_STYLE = {
    "none":     (GREEN,  "  DRIVING  "),
    "minor":    (YELLOW, "  MINOR COLLISION  "),
    "moderate": (ORANGE, "  MODERATE COLLISION  "),
    "severe":   (RED,    "  SEVERE COLLISION  "),
    "warmup":   (DIM,    "  WARMUP  "),
    "fallen":   (PURPLE, "  FALLEN - LOCKED  "),
}


# ══════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════
def classify_severity(impulse):
    mag = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
    if mag < IMPULSE_MINOR:
        return "minor", mag
    elif mag < IMPULSE_MODERATE:
        return "moderate", mag
    else:
        return "severe", mag


def get_motorcycle_bp():
    for m in MOTORCYCLE_MODELS:
        bp = blueprint_library.filter(m)
        if bp:
            return random.choice(list(bp))
    return random.choice(list(blueprint_library.filter("vehicle.*")))


def spawn_npcs():
    npcs_v, walkers, ctrls = [], [], []
    spawn_pts = world.get_map().get_spawn_points()
    random.shuffle(spawn_pts)
    veh_bps = [bp for bp in blueprint_library.filter("vehicle.*")
                if int(bp.get_attribute("number_of_wheels")) == 4]
    for sp in spawn_pts[:NUM_NPC_VEHICLES]:
        try:
            v = world.try_spawn_actor(random.choice(veh_bps), sp)
            if v:
                v.set_autopilot(True, 8000)
                npcs_v.append(v)
        except Exception:
            pass
    ctrl_bp = blueprint_library.find("controller.ai.walker")
    w_bps   = list(blueprint_library.filter("walker.pedestrian.*"))
    for _ in range(NUM_NPC_WALKERS):
        try:
            w = world.try_spawn_actor(
                random.choice(w_bps),
                carla.Transform(world.get_random_location_from_navigation())
            )
            if w:
                walkers.append(w)
        except Exception:
            pass
    world.tick()
    for w in walkers:
        try:
            c = world.spawn_actor(ctrl_bp, carla.Transform(), attach_to=w)
            ctrls.append(c)
        except Exception:
            pass
    world.tick()
    for c in ctrls:
        try:
            c.start()
            c.go_to_location(world.get_random_location_from_navigation())
            c.set_max_speed(1.4)
        except Exception:
            pass
    return npcs_v, walkers, ctrls


def destroy_npcs(npcs_v, walkers, ctrls):
    for c in ctrls:
        try: c.stop()
        except Exception: pass
    for actor in npcs_v + walkers + ctrls:
        try: actor.destroy()
        except Exception: pass


def spawn_sensors(vehicle, on_cam_cb, on_imu_cb, on_col_cb):
    cam_bp = blueprint_library.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(CAM_W))
    cam_bp.set_attribute("image_size_y", str(CAM_H))
    cam_bp.set_attribute("fov", "90")
    cam = world.spawn_actor(
        cam_bp,
        carla.Transform(carla.Location(x=-4.5, z=2.4), carla.Rotation(pitch=-12)),
        attach_to=vehicle,
    )
    cam.listen(on_cam_cb)
    imu = world.spawn_actor(
        blueprint_library.find("sensor.other.imu"),
        carla.Transform(), attach_to=vehicle,
    )
    imu.listen(on_imu_cb)
    col = world.spawn_actor(
        blueprint_library.find("sensor.other.collision"),
        carla.Transform(), attach_to=vehicle,
    )
    col.listen(on_col_cb)
    return [cam, imu, col]


def destroy_sensors(sensors):
    for s in sensors:
        try: s.stop()
        except Exception: pass
        try: s.destroy()
        except Exception: pass


def add_motorcycle_noise(ax, ay, az, gx, gy, gz, throttle):
    engine_scale = 0.05 + throttle * 0.12
    ax += np.random.normal(0, 0.015)
    ay += np.random.normal(0, engine_scale * 0.6)
    az += np.random.normal(0, engine_scale)
    if random.random() < 0.01:
        az += np.random.normal(0, 2.5)
        ay += np.random.normal(0, 1.2)
    gx += np.random.normal(0, 0.008)
    gy += np.random.normal(0, 0.008)
    gz += np.random.normal(0, 0.005)
    return ax, ay, az, gx, gy, gz


def get_manual_control(steer_prev):
    keys       = pygame.key.get_pressed()
    throttle   = 0.6 if (keys[pygame.K_w] or keys[pygame.K_UP])   else 0.0
    brake      = 0.8 if (keys[pygame.K_s] or keys[pygame.K_DOWN]) else 0.0
    hand_brake = bool(keys[pygame.K_SPACE])
    if keys[pygame.K_a] or keys[pygame.K_LEFT]:
        steer = max(steer_prev - 0.05, -1.0)
    elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
        steer = min(steer_prev + 0.05,  1.0)
    else:
        steer = steer_prev * 0.85
    return throttle, brake, steer, hand_brake


def save_run(run_id, ring_buf, post_buf, severity_val):
    had_collision = severity_val != "none"

    if had_collision:
        all_rows = list(ring_buf) + post_buf
        tag      = f"collision_{severity_val}"
    else:
        full = list(ring_buf)
        if len(full) > NO_COL_SAMPLES:
            start    = random.randint(0, len(full) - NO_COL_SAMPLES)
            all_rows = full[start: start + NO_COL_SAMPLES]
        else:
            all_rows = full
        tag = "no_collision"

    if not all_rows:
        print(f"  [RUN {run_id}] empty — skipped")
        return

    out_path = os.path.join(OUTPUT_DIR, f"run_{run_id:04d}_{tag}.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        for row in all_rows:
            writer.writerow([run_id] + row)

    print(f"  [SAVED] {os.path.basename(out_path)} | rows={len(all_rows)}"
            f" | pre={len(ring_buf)} post={len(post_buf)}"
            f" | severity={severity_val}")

    _stats["total"] += 1
    if had_collision:
        _stats["collision"] += 1
        _stats[severity_val] += 1

    rate = _stats["collision"] / max(_stats["total"], 1)
    if _stats["total"] >= 10 and rate < 0.40:
        print(f"  IMBALANCE WARNING: collision rate={rate:.0%}"
                f" ({_stats['collision']}/{_stats['total']})"
                f" | minor={_stats['minor']}"
                f" moderate={_stats['moderate']}"
                f" severe={_stats['severe']}")


# ══════════════════════════════════════════
#  Drawing
# ══════════════════════════════════════════
def draw_bar(surf, x, y, w, h, val, lo, hi, col, lbl):
    norm = np.clip((val - lo) / (hi - lo + 1e-9), 0, 1)
    pygame.draw.rect(surf, BORDER, (x, y, w, h))
    pygame.draw.rect(surf, col,    (x, y, int(norm * w), h))
    z = x + int((0 - lo) / (hi - lo + 1e-9) * w)
    pygame.draw.line(surf, DIM, (z, y), (z, y + h), 1)
    surf.blit(F_MINI.render(f"{lbl}: {val:+.3f}", True, TXT), (x+3, y+1))


def draw_sparkline(surf, x, y, w, h, hist, col):
    d = list(hist)
    if len(d) < 2:
        return
    lo = min(d); hi = max(d); rng = hi - lo or 1.0
    pts = [(x + int(i / (len(d)-1) * w),
            y + h - int((v - lo) / rng * h)) for i, v in enumerate(d)]
    pygame.draw.lines(surf, col, False, pts, 1)


def draw_panel(surf, st):
    px = CAM_W
    pygame.draw.rect(surf, PANEL,  (px, 0, PANEL_W, SCREEN_H))
    pygame.draw.line(surf, BORDER, (px, 0), (px, SCREEN_H), 2)
    x  = px + 12
    y  = 10
    bw = PANEL_W - 24

    surf.blit(F_TITLE.render("IMU RECORDER  v3", True, CYAN), (x, y)); y += 26
    pygame.draw.line(surf, BORDER, (px+6, y), (px+PANEL_W-6, y), 1);   y += 8

    elapsed  = st.get("elapsed", 0.0)
    post_n   = st.get("post_n", 0)
    sev      = st.get("severity", "none")
    impulse  = st.get("impulse", 0.0)
    rate     = _stats["collision"] / max(_stats["total"], 1)

    for k, v in [
        ("RUN",      f"{st.get('run_id', 1)}  /  {TOTAL_RUNS}"),
        ("ELAPSED",  f"{elapsed:.1f}s  /  {RUN_DURATION - WARMUP_SEC:.0f}s"),
        ("PRE BUF",  f"{st.get('pre_n', 0)}  /  {PRE_SAMPLES}"),
        ("POST BUF", f"{post_n}  /  {POST_SAMPLES}"),
        ("COL RATE", f"{rate:.0%}  ({_stats['collision']}/{max(_stats['total'],1)})"),
        ("SEVERITY", sev.upper()),
        ("IMPULSE",  f"{impulse:.0f}"),
    ]:
        surf.blit(F_LBL.render(f"{k:<10}", True, DIM),  (x, y))
        col_txt = {"MINOR": YELLOW, "MODERATE": ORANGE, "SEVERE": RED}.get(v, TXT)
        surf.blit(F_VAL.render(v, True, col_txt), (x+115, y))
        y += 17
    y += 6

    if st.get("warmup"):
        style_key = "warmup"
    elif st.get("fallen"):
        style_key = "fallen"
    elif sev != "none":
        style_key = sev
    else:
        style_key = "none"

    bc, btxt = SEVERITY_STYLE[style_key]
    pygame.draw.rect(surf, bc, (x, y, bw, 24), border_radius=4)
    bs = F_LBL.render(btxt, True, (0, 0, 0))
    surf.blit(bs, (x + (bw - bs.get_width()) // 2, y + 4)); y += 34

    pygame.draw.line(surf, BORDER, (px+6, y), (px+PANEL_W-6, y), 1); y += 8

    surf.blit(F_LBL.render("ACCELEROMETER  (m/s2)", True, CYAN), (x, y)); y += 18
    for v, l in zip(st.get("acc", [0, 0, 0]), ["X", "Y", "Z"]):
        draw_bar(surf, x, y, bw, 14, v, -40, 40, CYAN, f"acc_{l}"); y += 18
    y += 4
    surf.blit(F_MINI.render("history", True, DIM), (x, y)); y += 14
    for h in st.get("acc_h", [deque()] * 3):
        draw_sparkline(surf, x, y, bw, 22, h, CYAN); y += 25

    pygame.draw.line(surf, BORDER, (px+6, y), (px+PANEL_W-6, y), 1); y += 8

    surf.blit(F_LBL.render("GYROSCOPE  (rad/s)", True, PURPLE), (x, y)); y += 18
    for v, l in zip(st.get("gyro", [0, 0, 0]), ["X", "Y", "Z"]):
        draw_bar(surf, x, y, bw, 14, v, -9, 9, PURPLE, f"gyro_{l}"); y += 18
    y += 4
    surf.blit(F_MINI.render("history", True, DIM), (x, y)); y += 14
    for h in st.get("gyro_h", [deque()] * 3):
        draw_sparkline(surf, x, y, bw, 22, h, PURPLE); y += 25

    pygame.draw.line(surf, BORDER, (px+6, y), (px+PANEL_W-6, y), 1); y += 8
    for hint in ["W/S throttle/brake", "A/D steer  SPACE handbrake",
                 "R save+respawn  ESC quit"]:
        surf.blit(F_MINI.render(hint, True, DIM), (x, y)); y += 14


# ══════════════════════════════════════════
#  One run
# ══════════════════════════════════════════
def collect_run(run_id):
    global cam_surf, cam_ready

    ring_buf        = deque(maxlen=PRE_SAMPLES)
    full_buf        = []
    post_buf        = []
    col_time        = [-1.0]
    collecting_post = [False]
    severity        = ["none"]
    impulse_mag     = [0.0]
    fallen          = [False]
    sim_t0          = [None]

    last_imu     = {"acc": [0, 0, 0], "gyro": [0, 0, 0]}
    acc_h        = [deque(maxlen=80) for _ in range(3)]   # قلّلنا من 200
    gyro_h       = [deque(maxlen=80) for _ in range(3)]   # قلّلنا من 200
    throttle_ref = [0.0]

    def on_cam(img):
        global cam_surf, cam_ready
        arr      = np.frombuffer(img.raw_data, dtype=np.uint8)
        arr      = arr.reshape((img.height, img.width, 4))
        cam_surf  = pygame.surfarray.make_surface(
            arr[:, :, :3][:, :, ::-1].swapaxes(0, 1))
        cam_ready = True

    def on_imu(event):
        if sim_t0[0] is None:
            sim_t0[0] = event.timestamp
        elapsed_sim = event.timestamp - sim_t0[0]

        if elapsed_sim < WARMUP_SEC:
            return

        ts_ms = round((elapsed_sim - WARMUP_SEC) * 1000.0, 3)
        a = event.accelerometer
        g = event.gyroscope

        ax, ay, az, gx, gy, gz = add_motorcycle_noise(
            a.x, a.y, a.z, g.x, g.y, g.z, throttle_ref[0]
        )

        if not fallen[0] and abs(az) < 2.0 and abs(ax) > 5.0:
            fallen[0] = True
            print(f"  [RUN {run_id}] Motorcycle fallen at {elapsed_sim:.2f}s")

        row = [ts_ms, ax, ay, az, gx, gy, gz,
                0 if col_time[0] < 0 else 1,
                severity[0]]

        if col_time[0] < 0:
            ring_buf.append(row)
            full_buf.append(row)
        elif collecting_post[0] and len(post_buf) < POST_SAMPLES:
            post_buf.append(row)

        last_imu["acc"]  = [ax, ay, az]
        last_imu["gyro"] = [gx, gy, gz]
        for i in range(3):
            acc_h[i].append([ax, ay, az][i])
            gyro_h[i].append([gx, gy, gz][i])

    def on_col(event):
        if col_time[0] < 0:
            sev, mag       = classify_severity(event.normal_impulse)
            severity[0]    = sev
            impulse_mag[0] = mag
            col_time[0]    = event.timestamp - (sim_t0[0] or event.timestamp)
            collecting_post[0] = True
            print(f"  [RUN {run_id}] Collision @ {col_time[0]:.2f}s"
                    f" | impulse={mag:.0f} | severity={sev}")

    spawn_pts = world.get_map().get_spawn_points()
    vehicle   = world.spawn_actor(get_motorcycle_bp(), random.choice(spawn_pts))
    world.tick(); world.tick()

    npcs_v, walkers, ctrls = spawn_npcs()
    sensors = spawn_sensors(vehicle, on_cam, on_imu, on_col)

    steer   = 0.0
    running = True
    saved   = False
    render_tick = 0
    print(f"[RUN {run_id}] GO  |  R=save+respawn  |  ESC=quit")

    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                raise SystemExit
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    raise SystemExit

                if ev.key == pygame.K_r:
                    if fallen[0]:
                        print(f"  [RUN {run_id}] Fallen — respawn blocked")
                        continue
                    save_run(run_id, ring_buf, post_buf, severity[0])
                    saved = True

                    destroy_sensors(sensors)
                    try: vehicle.destroy()
                    except Exception: pass
                    destroy_npcs(npcs_v, walkers, ctrls)

                    vehicle = world.spawn_actor(
                        get_motorcycle_bp(), random.choice(spawn_pts))
                    world.tick()
                    npcs_v, walkers, ctrls = spawn_npcs()
                    sensors = spawn_sensors(vehicle, on_cam, on_imu, on_col)

                    ring_buf.clear(); full_buf.clear(); post_buf.clear()
                    col_time[0] = -1.0; collecting_post[0] = False
                    severity[0] = "none"; impulse_mag[0] = 0.0
                    fallen[0]   = False
                    sim_t0[0]   = None; steer = 0.0; saved = False
                    print(f"[RUN {run_id}] Respawned")

        throttle, brake, steer, hand_brake = get_manual_control(steer)
        throttle_ref[0] = throttle
        vehicle.apply_control(carla.VehicleControl(
            throttle=throttle, brake=brake,
            steer=steer, hand_brake=hand_brake,
        ))
        world.tick()

        snap      = world.get_snapshot().timestamp.elapsed_seconds
        elapsed   = max(0.0, snap - (sim_t0[0] or snap) - WARMUP_SEC)
        in_warmup = sim_t0[0] is None or (snap - sim_t0[0]) < WARMUP_SEC

        if col_time[0] >= 0 and len(post_buf) >= POST_SAMPLES:
            running = False
        elif sim_t0[0] is not None and snap - sim_t0[0] >= RUN_DURATION:
            running = False

        render_tick += 1
        if render_tick % RENDER_EVERY != 0:
            continue

        # render
        screen.fill(BG)
        if cam_ready and cam_surf:
            screen.blit(cam_surf, (0, 0))

            xi = 10
            for txt, col in [
                (f"THR: {throttle:.2f}", GREEN),
                (f"BRK: {brake:.2f}",   RED),
                (f"STR: {steer:+.2f}",  CYAN),
                ("HB: ON" if hand_brake else "HB: OFF",
                    RED if hand_brake else DIM),
            ]:
                s = F_LBL.render(txt, True, col)
                screen.blit(s, (xi, CAM_H - 28))
                xi += s.get_width() + 20

            if col_time[0] >= 0:
                sev_col = {"minor": YELLOW, "moderate": ORANGE,
                            "severe": RED}.get(severity[0], RED)
                ovl = pygame.Surface((CAM_W, 50), pygame.SRCALPHA)
                r, g_c, b = sev_col
                ovl.fill((r, g_c, b, 160))
                screen.blit(ovl, (0, 0))
                screen.blit(F_BIG.render(
                    f"!! {severity[0].upper()}  +{len(post_buf)/SAMPLE_RATE:.1f}s"
                    f"  imp={impulse_mag[0]:.0f}",
                    True, (0, 0, 0)), (12, 10))

            t_color = RED if elapsed > 12 else ORANGE if elapsed > 8 else GREEN
            screen.blit(F_BIG.render(f"{elapsed:.1f}s", True, t_color),
                        (CAM_W - 120, 10))
        else:
            screen.blit(F_TITLE.render("Waiting for camera...", True, DIM),
                        (CAM_W // 2 - 90, CAM_H // 2))

        draw_panel(screen, {
            "run_id":   run_id,
            "elapsed":  elapsed,
            "pre_n":    len(ring_buf),
            "post_n":   len(post_buf),
            "severity": severity[0],
            "impulse":  impulse_mag[0],
            "fallen":   fallen[0],
            "warmup":   in_warmup,
            "acc":      last_imu["acc"],
            "gyro":     last_imu["gyro"],
            "acc_h":    acc_h,
            "gyro_h":   gyro_h,
        })
        pygame.display.flip()
        clock_pg.tick(60)

    if not saved:
        save_run(run_id, ring_buf, post_buf, severity[0])

    destroy_sensors(sensors)
    try: vehicle.destroy()
    except Exception: pass
    destroy_npcs(npcs_v, walkers, ctrls)
    cam_surf  = None
    cam_ready = False


# ══════════════════════════════════════════
#  Main
# ══════════════════════════════════════════
def main():
    print("=" * 60)
    print("  CARLA IMU Recorder  v3")
    print(f"  {SAMPLE_RATE} Hz  |  run={RUN_DURATION}s"
            f"  warmup={WARMUP_SEC}s"
            f"  pre={PRE_COLLISION_SEC}s  post={POST_COLLISION_SEC}s")
    print(f"  Severity thresholds:"
            f"  minor < {IMPULSE_MINOR}"
            f"  moderate < {IMPULSE_MODERATE}"
            f"  severe >= {IMPULSE_MODERATE}")
    print(f"  Output: {OUTPUT_DIR}/")
    print(f"  Header: {CSV_HEADER}")
    print("=" * 60)
    try:
        for run_id in range(1, TOTAL_RUNS + 1):
            print(f"\n{'='*25}  RUN {run_id}/{TOTAL_RUNS}  {'='*25}")
            collect_run(run_id)
        print("\nALL RUNS DONE.")
        print(f"  total={_stats['total']}"
                f"  collision={_stats['collision']}"
                f"  minor={_stats['minor']}"
                f"  moderate={_stats['moderate']}"
                f"  severe={_stats['severe']}")
    except SystemExit:
        print("Stopped by user.")
    finally:
        cleanup()
        pygame.quit()
        print("World settings restored.")


if __name__ == "__main__":
    main()
