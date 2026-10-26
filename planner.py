"""Jump math: which branch to jump to, and the exact x to take off from so you land on its middle.

Physics is learned from your own jumps (JumpLearner), so it fits any resolution / server.
Pure Python - testable without the game.
"""
import math

# MapleStory defaults (pixels/sec at 1x scale) - used until a few real jumps were measured.
WALK_SPEED = 125.0
JUMP_SPEED = 555.0
GRAVITY = 2000.0


# Map scale (from the log size) -> physics scale. Measured on the map; JumpLearner refines it.
PHYS_PER_SCALE = 1.7


class Physics:
    def __init__(self, scale=1.0):
        scale *= PHYS_PER_SCALE
        self.vx = WALK_SPEED * scale
        self.vy = JUMP_SPEED * scale
        self.g = GRAVITY * scale

    @property
    def max_rise(self):
        return self.vy ** 2 / (2 * self.g)

    def air_time(self, rise):
        """Seconds from take-off until landing `rise` px higher (on the way down). None = too high."""
        d = self.vy ** 2 - 2 * self.g * rise
        if d < 0:
            return None
        return (self.vy + math.sqrt(d)) / self.g

    def arc(self, x, y, direction, steps=24, rise_floor=None):
        """Points of a jump arc from (x, y) - for drawing. Screen y grows downward."""
        t_end = self.air_time(rise_floor or 0) or 2 * self.vy / self.g
        pts = []
        for i in range(steps + 1):
            t = t_end * i / steps
            pts.append((x + direction * self.vx * t, y - (self.vy * t - self.g * t * t / 2)))
        return pts


def standing_on(branches, x, feet, scale):
    tol = 12 * scale
    for b in branches:
        if abs(feet - b.top) <= tol and b.left - 4 * scale <= x <= b.right + 4 * scale:
            return b
    return None


def pick_target(branches, current, phys, scale, prefer_dir=0):
    """Next branch to climb to: higher, reachable, closest. `prefer_dir` breaks ties (-1 left / 1 right)."""
    best, best_cost = None, None
    min_rise = 10 * scale
    for b in branches:
        if b is current:
            continue
        rise = current.top - b.top
        if rise < min_rise or rise > phys.max_rise * 0.92:
            continue
        t = phys.air_time(rise)
        gap = abs(b.cx - current.cx)
        # farthest we can travel: from the far edge of our branch at full air time
        if gap > phys.vx * t + current.half + b.half:
            continue
        cost = gap + 0.5 * rise
        if prefer_dir and (b.cx - current.cx) * prefer_dir < 0:
            cost += 1000
        if best is None or cost < best_cost:
            best, best_cost = b, cost
    return best


def plan_jump(current, target, phys, scale, latency=0.0):
    """Where to press jump. Returns dict(direction, takeoff_x, press_x, land_x, ok).

    direction 0 = straight up (target is right above). press_x includes your reaction / input latency.
    """
    rise = current.top - target.top
    t = phys.air_time(rise)
    if t is None:
        return None
    gap = target.cx - current.cx
    reach = phys.vx * t
    if abs(gap) < 6 * scale:
        return dict(direction=0, takeoff_x=target.cx, press_x=target.cx, land_x=target.cx, ok=True)
    d = 1 if gap > 0 else -1
    takeoff = target.cx - d * reach
    # can't take off outside our own branch - clamp, and say if we still land on the target
    margin = 3 * scale
    clamped = min(max(takeoff, current.left + margin), current.right - margin)
    land = clamped + d * reach
    ok = abs(land - target.cx) <= target.half - 2 * scale
    press = clamped - d * phys.vx * latency
    return dict(direction=d, takeoff_x=clamped, press_x=press, land_x=land, ok=ok)


class JumpLearner:
    """Watches the character in the air and fits the real walk speed / jump speed / gravity."""

    def __init__(self, phys):
        self.phys = phys
        self.samples = []   # (t, x, y) while airborne
        self.jumps = 0

    def feed(self, t, x, y, grounded):
        if not grounded:
            self.samples.append((t, x, y))
            return
        if len(self.samples) >= 6:
            self._fit()
        self.samples = []

    def _fit(self):
        s = self.samples
        t0 = s[0][0]
        ts = [p[0] - t0 for p in s]
        n = len(s)
        # horizontal: least squares slope of x(t)
        mt = sum(ts) / n
        mx = sum(p[1] for p in s) / n
        var = sum((t - mt) ** 2 for t in ts)
        if var <= 0:
            return
        vx = abs(sum((t - mt) * (p[1] - mx) for t, p in zip(ts, s)) / var)
        # vertical: y = a + b t + c t^2  (screen y down) -> vy = -b, g = 2c
        try:
            a, b, c = _quadfit(ts, [p[2] for p in s])
        except ZeroDivisionError:
            return
        vy, g = -b, 2 * c
        if g <= 0 or vy <= 0:
            return
        k = 0.3 if self.jumps else 0.6   # first real jump replaces most of the guess
        if vx > 0.2 * self.phys.vx:      # standing jumps tell nothing about walk speed
            self.phys.vx += k * (vx - self.phys.vx)
        self.phys.vy += k * (vy - self.phys.vy)
        self.phys.g += k * (g - self.phys.g)
        self.jumps += 1


def _quadfit(xs, ys):
    """Least squares y = a + b x + c x^2 (tiny normal-equations solve)."""
    s = [sum(x ** k for x in xs) for k in range(5)]
    t = [sum(y * x ** k for x, y in zip(xs, ys)) for k in range(3)]
    m = [[s[0], s[1], s[2]], [s[1], s[2], s[3]], [s[2], s[3], s[4]]]

    def det(a):
        return (a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
                - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
                + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))

    d = det(m)
    if abs(d) < 1e-12:
        raise ZeroDivisionError
    out = []
    for col in range(3):
        mc = [row[:] for row in m]
        for r in range(3):
            mc[r][col] = t[r]
        out.append(det(mc) / d)
    return out
