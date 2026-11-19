"""Overlay + result card + H.264 export. Frames are matched to telemetry rows by CARLA frame id."""
import subprocess, textwrap, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib

_FD = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data/fonts/ttf")
COL = dict(PASS=(40, 170, 80), FAIL=(210, 50, 50), UNRESOLVED=(220, 160, 30), INVALID=(130, 130, 130))
PHASE_COL = dict(approach=(70, 140, 220), brake=(230, 120, 30), hold=(40, 170, 80))


def _font(sz, bold=False):
    return ImageFont.truetype(os.path.join(_FD, "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"), sz)


def overlay(img, row, t_cue):
    im = Image.fromarray(img).convert("RGBA")
    lay = Image.new("RGBA", im.size, (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
    d.rectangle([12, 12, 430, 190], fill=(0, 0, 0, 160))
    rel = row["sim_time"] - t_cue if t_cue is not None else float("nan")
    ph = row["phase"]
    f, fb = _font(24), _font(26, True)
    lines = [f"sim time  {row['sim_time']:7.2f} s   (cue {rel:+.2f} s)",
             f"speed     {row['speed']:6.2f} m/s",
             f"gap       {row['gap']:6.2f} m",
             f"a_req {row['a_req']:+5.2f}  accel {row['accel_long']:+5.2f} m/s2"]
    for i, s in enumerate(lines): d.text((24, 20 + 31 * i), s, font=f, fill=(255, 255, 255, 255))
    d.rounded_rectangle([24, 146, 300, 182], 8, fill=PHASE_COL[ph] + (255,))
    d.text((34, 148), f"PHASE: {ph.upper()}", font=_font(22, True), fill=(255, 255, 255, 255))
    if row["obs_delivered"] and rel < 1.0:
        d.text((450, 20), "BLOCKED-ROAD OBSERVATION DELIVERED", font=fb, fill=(255, 220, 60, 255))
    if row["contact_any"]:
        d.text((450, 56), "CONTACT", font=_font(40, True), fill=(255, 60, 60, 255))
    return np.array(Image.alpha_composite(im, lay).convert("RGB"))


def card(img, label, reason, stop_gap, min_gap, run_id):
    im = Image.fromarray(img).convert("RGBA")
    W, H = im.size
    lay = Image.new("RGBA", im.size, (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
    d.rectangle([0, 0, W, H], fill=(0, 0, 0, 170))
    d.rounded_rectangle([W // 2 - 480, H // 2 - 200, W // 2 + 480, H // 2 + 200], 20, fill=(25, 25, 30, 245), outline=COL[label] + (255,), width=6)
    d.text((W // 2, H // 2 - 150), label, font=_font(84, True), fill=COL[label] + (255,), anchor="mt")
    y = H // 2 - 40
    for ln in textwrap.wrap("Reason: " + reason, 62):
        d.text((W // 2 - 440, y), ln, font=_font(26), fill=(255, 255, 255, 255)); y += 34
    if stop_gap is not None: g = f"Stopping gap: {stop_gap:.2f} m   (min gap {min_gap:.2f} m)"
    else: g = f"Did not stop - minimum gap: {min_gap:.2f} m"
    d.text((W // 2 - 440, H // 2 + 130), g, font=_font(30, True), fill=(255, 255, 255, 255))
    d.text((W // 2 - 440, H // 2 + 170), run_id, font=_font(18), fill=(170, 170, 170, 255))
    return np.array(Image.alpha_composite(im, lay).convert("RGB"))


def export(path, frames, fps):
    h, w = frames[0].shape[:2]
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                          "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                          "-preset", "medium", "-movflags", "+faststart", path], stdin=subprocess.PIPE)
    for fr in frames: p.stdin.write(np.ascontiguousarray(fr).tobytes())
    p.stdin.close()
    if p.wait() != 0: raise RuntimeError("ffmpeg failed")


def overlay_lines(img, lines, phase, banners=()):
    """Generic overlay: text lines + phase pill + banners (list of (text, rgb))."""
    im = Image.fromarray(img).convert("RGBA")
    lay = Image.new("RGBA", im.size, (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
    f = _font(22)
    h = 14 + 29 * len(lines) + 46
    d.rectangle([12, 12, 520, 12 + h], fill=(0, 0, 0, 165))
    for i, s in enumerate(lines): d.text((24, 20 + 29 * i), s, font=f, fill=(255, 255, 255, 255))
    y = 20 + 29 * len(lines)
    d.rounded_rectangle([24, y, 330, y + 34], 8, fill=PHASE_COL.get(phase, (110, 110, 160)) + (255,))
    d.text((34, y + 3), f"PHASE: {phase.upper()}", font=_font(21, True), fill=(255, 255, 255, 255))
    for i, (txt, col) in enumerate(banners):
        d.text((560, 20 + 44 * i), txt, font=_font(30, True), fill=tuple(col) + (255,))
    return np.array(Image.alpha_composite(im, lay).convert("RGB"))


def card2(img, label, reason, detail, run_id):
    im = Image.fromarray(img).convert("RGBA")
    W, H = im.size
    lay = Image.new("RGBA", im.size, (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
    d.rectangle([0, 0, W, H], fill=(0, 0, 0, 170))
    d.rounded_rectangle([W // 2 - 520, H // 2 - 230, W // 2 + 520, H // 2 + 230], 20, fill=(25, 25, 30, 245), outline=COL[label] + (255,), width=6)
    d.text((W // 2, H // 2 - 190), label, font=_font(80, True), fill=COL[label] + (255,), anchor="mt")
    y = H // 2 - 85
    for ln in textwrap.wrap("Reason: " + reason, 66)[:5]:
        d.text((W // 2 - 480, y), ln, font=_font(25), fill=(255, 255, 255, 255)); y += 32
    y = H // 2 + 105
    for ln in textwrap.wrap(detail, 70)[:3]:
        d.text((W // 2 - 480, y), ln, font=_font(25, True), fill=(255, 255, 255, 255)); y += 32
    d.text((W // 2 - 480, H // 2 + 200), run_id, font=_font(17), fill=(170, 170, 170, 255))
    return np.array(Image.alpha_composite(im, lay).convert("RGB"))
