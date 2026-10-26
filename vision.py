"""Screen understanding for The Forest of Patience: finds the small branches and the character.

Pure functions on BGR numpy images (no Windows APIs) so they can be tested on a screenshot.
"""
import cv2
import numpy as np

# Brown of the round log under each branch (HSV, OpenCV ranges). Measured on the real map.
LOG_HSV_LO = (8, 120, 70)
LOG_HSV_HI = (22, 220, 170)

# Log disk diameter in screen pixels at 1x scale; the walkable bar sits a bit above the disk.
REF_DISK = 36
BAR_ABOVE_DISK = 15      # bar top is this many px (at 1x) above the disk's top
BAR_HALF_WIDTH = 27      # half length of the walkable bar (at 1x)

NAME_WHITE = 185         # name tag text brightness threshold


class Branch:
    __slots__ = ("cx", "top", "half", "disk")

    def __init__(self, cx, top, half, disk):
        self.cx, self.top, self.half, self.disk = cx, top, half, disk

    @property
    def left(self):
        return self.cx - self.half

    @property
    def right(self):
        return self.cx + self.half

    def __repr__(self):
        return f"Branch(cx={self.cx:.0f}, top={self.top:.0f})"


def find_branches(bgr, ignore=()):
    """Return branches (walkable bar center + top y), found by the brown log disks.

    `ignore` = list of (x, y, w, h) rectangles (UI windows) to skip.
    """
    small = cv2.resize(bgr, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOG_HSV_LO, LOG_HSV_HI)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        x, y, w, h, area = x * 2, y * 2, w * 2, h * 2, area * 4
        if not (20 <= w <= 140 and 20 <= h <= 140):
            continue
        if not (0.6 < w / h < 1.6) or area < 0.45 * w * h:
            continue
        cx = x + w / 2
        if any(rx <= cx <= rx + rw and ry <= y <= ry + rh for rx, ry, rw, rh in ignore):
            continue
        s = w / REF_DISK
        out.append(Branch(cx, y - BAR_ABOVE_DISK * s, BAR_HALF_WIDTH * s, w))
    if out:  # drop half-hidden logs (behind chat / UI) - they are much smaller than the rest
        med = float(np.median([b.disk for b in out]))
        out = [b for b in out if b.disk >= 0.75 * med]
    return out


def merge_platforms(branches):
    """Step 2 has rows of 3-4 logs = one wide platform. Merge neighbours on the same height."""
    bands = []   # logs on (almost) the same height
    for b in sorted(branches, key=lambda b: b.top):
        if bands and b.top - bands[-1][-1].top <= 0.3 * b.disk:
            bands[-1].append(b)
        else:
            bands.append([b])
    rows = []    # split each height band where the logs stop touching
    for band in bands:
        band.sort(key=lambda b: b.cx)
        rows.append([band[0]])
        for b in band[1:]:
            if b.cx - rows[-1][-1].cx <= 2.8 * b.disk:   # a log or two may be missed
                rows[-1].append(b)
            else:
                rows.append([b])
    # a long floor cut in two by a game window (toolbar) is still one floor
    joined = []
    for row in rows:
        prev = joined[-1] if joined else None
        if prev and len(prev) >= 4 and len(row) >= 4 and abs(prev[-1].top - row[0].top) <= 0.3 * row[0].disk:
            prev.extend(row)
        else:
            joined.append(row)
    rows = joined
    out = []
    for row in rows:
        if len(row) == 1:
            out.append(row[0])
            continue
        left, right = row[0].left, row[-1].right
        top = float(np.median([b.top for b in row]))
        out.append(Branch((left + right) / 2, top, (right - left) / 2, row[0].disk))
    return out


class Rope:
    __slots__ = ("x", "top", "bottom")

    def __init__(self, x, top, bottom):
        self.x, self.top, self.bottom = x, top, bottom

    def __repr__(self):
        return f"Rope(x={self.x:.0f}, {self.top:.0f}..{self.bottom:.0f})"


ROPE_HSV_LO = (55, 190, 110)
ROPE_HSV_HI = (70, 255, 255)


def find_ropes(bgr):
    """Climbable green vines: long, thin, vertical, bright green."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, ROPE_HSV_LO, ROPE_HSV_HI)
    m = cv2.dilate(m, np.ones((1, 9), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 1), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((120, 1), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(m)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if 10 <= w <= 40 and h >= 150 and area >= 8 * h:
            out.append(Rope(x + w / 2, float(y), float(y + h)))
    return out


def scale_of(branches):
    """Screen scale vs. the reference resolution, from the median disk size."""
    if not branches:
        return 1.0
    return float(np.median([b.disk for b in branches])) / REF_DISK


def name_tag_template(bgr, x, y, w=120, h=34):
    """Cut a binary name-tag template around (x, y) - the mouse position during calibration."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    b = (g > NAME_WHITE).astype(np.uint8) * 255
    x0, y0 = max(0, x - w // 2), max(0, y - h // 2)
    crop = b[y0:y0 + h, x0:x0 + w]
    ys, xs = np.nonzero(crop)
    if len(xs) < 10:
        return None
    # tighten to the text itself with a small margin
    crop = crop[max(0, ys.min() - 3):ys.max() + 4, max(0, xs.min() - 3):xs.max() + 4]
    return crop


def find_character(bgr, tag, min_score=0.6):
    """Return (x, feet_y, score) of the character using its name tag, or None."""
    if tag is None:
        return None
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    b = (g > NAME_WHITE).astype(np.uint8) * 255
    r = cv2.matchTemplate(b, tag, cv2.TM_CCOEFF_NORMED)
    _, score, _, (x, y) = cv2.minMaxLoc(r)
    if score < min_score:
        return None
    th, tw = tag.shape
    # the tag hangs right under the feet
    return x + tw / 2, y - 0.25 * th, score
