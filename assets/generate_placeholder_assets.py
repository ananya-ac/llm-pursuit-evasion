"""One-off generator for small placeholder images used by the vision-augmented
planner (Experiment II). No real drone/bird photography exists in this repo,
so these are simple PIL-drawn silhouettes: mechanical quadcopter shapes for
drones, organic winged shapes for birds -- distinct enough in silhouette for
a vision model to tell apart, without depending on any external asset source.

Run once (`python3 assets/generate_placeholder_assets.py`) to (re)populate
assets/drones/ and assets/birds/; the files it writes are checked in, so this
does not need to run again unless the asset set is intentionally changed.
"""

import math
import os

from PIL import Image, ImageDraw

SIZE = 128
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DRONE_DIR = os.path.join(OUT_DIR, "drones")
BIRD_DIR = os.path.join(OUT_DIR, "birds")


def _new_canvas():
    return Image.new("RGB", (SIZE, SIZE), color=(235, 235, 235))


def draw_drone(body_color, rotor_color, seed):
    img = _new_canvas()
    draw = ImageDraw.Draw(img)
    cx, cy = SIZE / 2, SIZE / 2
    arm_len = SIZE * 0.32
    rotor_r = SIZE * 0.11

    # Small per-variant jitter on arm angle so the four drone images aren't identical.
    base_angle = 45 + (seed * 7) % 20
    for k in range(4):
        angle = math.radians(base_angle + k * 90)
        ex = cx + arm_len * math.cos(angle)
        ey = cy + arm_len * math.sin(angle)
        draw.line([(cx, cy), (ex, ey)], fill=(60, 60, 60), width=4)
        draw.ellipse(
            [(ex - rotor_r, ey - rotor_r), (ex + rotor_r, ey + rotor_r)],
            fill=rotor_color,
            outline=(30, 30, 30),
            width=2,
        )

    body_r = SIZE * 0.13
    draw.ellipse(
        [(cx - body_r, cy - body_r), (cx + body_r, cy + body_r)],
        fill=body_color,
        outline=(20, 20, 20),
        width=2,
    )
    return img


def draw_bird(seed):
    img = _new_canvas()
    draw = ImageDraw.Draw(img)
    cx, cy = SIZE / 2, SIZE / 2
    wing_spread = SIZE * (0.36 + 0.04 * (seed % 3))
    body_color = (90, 70, 55)

    # Body: a small ellipse.
    body_w, body_h = SIZE * 0.14, SIZE * 0.09
    draw.ellipse(
        [(cx - body_w, cy - body_h), (cx + body_w, cy + body_h)],
        fill=body_color,
        outline=(40, 30, 20),
        width=2,
    )

    # Wings: two curved "V" strokes swept back, evoking a gliding bird silhouette.
    for side in (-1, 1):
        draw.line(
            [
                (cx, cy),
                (cx + side * wing_spread * 0.6, cy - SIZE * 0.10),
                (cx + side * wing_spread, cy - SIZE * 0.02),
            ],
            fill=(40, 30, 20),
            width=5,
            joint="curve",
        )

    # Small head/beak to break symmetry with drones.
    head_r = SIZE * 0.045
    hx, hy = cx + body_w * 0.9, cy - body_h * 0.3
    draw.ellipse([(hx - head_r, hy - head_r), (hx + head_r, hy + head_r)], fill=body_color)
    draw.polygon(
        [(hx + head_r, hy), (hx + head_r * 2.2, hy - head_r * 0.3), (hx + head_r, hy + head_r * 0.6)],
        fill=(230, 170, 40),
    )
    return img


def main():
    os.makedirs(DRONE_DIR, exist_ok=True)
    os.makedirs(BIRD_DIR, exist_ok=True)

    blue_colors = [(30, 90, 200), (20, 70, 170), (50, 110, 220)]
    red_colors = [(200, 40, 40), (170, 25, 25), (220, 60, 60)]

    for i, color in enumerate(blue_colors, start=1):
        draw_drone(body_color=color, rotor_color=(40, 40, 40), seed=i).save(
            os.path.join(DRONE_DIR, f"blue_{i}.png")
        )
    for i, color in enumerate(red_colors, start=1):
        draw_drone(body_color=color, rotor_color=(40, 40, 40), seed=i + 10).save(
            os.path.join(DRONE_DIR, f"red_{i}.png")
        )
    for i in range(1, 4):
        draw_bird(seed=i).save(os.path.join(BIRD_DIR, f"bird_{i}.png"))

    print(f"Wrote {len(blue_colors)} blue drone, {len(red_colors)} red drone, and 3 bird images.")


if __name__ == "__main__":
    main()
