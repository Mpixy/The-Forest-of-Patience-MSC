"""The Forest of Patience helper - overlay + perfect jump assist for MapleStory (Windows).

Draws the middle of every small branch on top of the game, shows where to jump from,
and (optionally) presses the keys for you so every jump lands on the middle of the next branch.

Hotkeys (default, see config.json):
  F8   calibrate - put the mouse on your character's NAME TAG and press F8 (once)
  hold CAPS LOCK  full assist: walks to the perfect spot and jumps to the next branch
  F10  auto-jump on/off: you walk, the helper presses jump at the perfect moment
  F9   overlay on/off
  F12  quit
"""
import ctypes
import json
import os
import sys
import threading
import time
import tkinter as tk
from ctypes import wintypes

import cv2
import keyboard
import mss
import numpy as np
import pydirectinput

import planner
import vision

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "config.json")
TAG_PATH = os.path.join(HERE, "name_tag.png")

DEFAULTS = {
    "window_title": "MapleStory",
    "jump_key": "alt",
    "left_key": "left",
    "right_key": "right",
    "up_key": "up",
    "assist_hold_key": "caps lock",
    "calibrate_key": "f8",
    "overlay_key": "f9",
    "autojump_key": "f10",
    "quit_key": "f12",
    "latency_ms": 50,
    "runup_px": 10,
    "fps": 60,
}

TRANSPARENT = "#ff00ff"
user32 = ctypes.windll.user32
pydirectinput.PAUSE = 0
pydirectinput.FAILSAFE = False


def load_config():
    cfg = dict(DEFAULTS)
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg.update(json.load(f))
    else:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    return cfg


def make_dpi_aware():
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        user32.SetProcessDPIAware()


def find_game_rect(title):
    """Client area of the game window on screen: (left, top, width, height) or None."""
    found = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def cb(hwnd, _):
        if not user32.IsWindowVisible(hwnd):
            return True
        n = user32.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(n + 1)
        user32.GetWindowTextW(hwnd, buf, n + 1)
        if title.lower() in buf.value.lower() and "helper" not in buf.value.lower():
            found.append(hwnd)
        return True

    user32.EnumWindows(cb, 0)
    if not found:
        return None
    hwnd = found[0]
    r = wintypes.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(r))
    p = wintypes.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(p))
    if r.right <= 0 or r.bottom <= 0:
        return None
    return p.x, p.y, r.right, r.bottom


def cursor_pos():
    p = wintypes.POINT()
    user32.GetCursorPos(ctypes.byref(p))
    return p.x, p.y


class Keys:
    """Holds/taps game keys through DirectInput scan codes (what MapleStory listens to)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.held = set()
        self.lock = threading.Lock()

    def hold(self, key):
        with self.lock:
            if key not in self.held:
                pydirectinput.keyDown(key)
                self.held.add(key)

    def release(self, key):
        with self.lock:
            if key in self.held:
                pydirectinput.keyUp(key)
                self.held.discard(key)

    def release_all(self):
        for k in list(self.held):
            self.release(k)

    def walk(self, direction):
        left, right = self.cfg["left_key"], self.cfg["right_key"]
        if direction < 0:
            self.release(right)
            self.hold(left)
        elif direction > 0:
            self.release(left)
            self.hold(right)
        else:
            self.release(left)
            self.release(right)

    def tap(self, key, ms=60):
        def run():
            pydirectinput.keyDown(key)
            time.sleep(ms / 1000)
            pydirectinput.keyUp(key)
        threading.Thread(target=run, daemon=True).start()


class Helper:
    def __init__(self, cfg):
        self.cfg = cfg
        self.keys = Keys(cfg)
        self.tag = cv2.imread(TAG_PATH, cv2.IMREAD_GRAYSCALE) if os.path.exists(TAG_PATH) else None
        self.overlay_on = True
        self.autojump = False
        self.running = True
        self.phys = None
        self.learner = None
        self.rect = None
        self.last_frame = None
        self.prev_x = None
        self.last_jump = 0.0
        self.assisting = False
        self.message = ""

        self.root = tk.Tk()
        self.root.title("Forest Helper overlay")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-transparentcolor", TRANSPARENT)
        self.root.config(bg=TRANSPARENT)
        self.canvas = tk.Canvas(self.root, bg=TRANSPARENT, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.root.update_idletasks()
        self._make_click_through()

        keyboard.add_hotkey(cfg["calibrate_key"], self.calibrate)
        keyboard.add_hotkey(cfg["overlay_key"], self.toggle_overlay)
        keyboard.add_hotkey(cfg["autojump_key"], self.toggle_autojump)
        keyboard.add_hotkey(cfg["quit_key"], self.quit)
        self.sct = mss.mss()

    def _make_click_through(self):
        hwnd = user32.GetParent(self.root.winfo_id()) or self.root.winfo_id()
        GWL_EXSTYLE, WS_EX_LAYERED, WS_EX_TRANSPARENT, WS_EX_TOOLWINDOW = -20, 0x80000, 0x20, 0x80
        style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW)
        # keep our drawings out of the screenshots we analyse (Windows 10 2004+)
        user32.SetWindowDisplayAffinity(hwnd, 0x11)

    # ---------- hotkeys ----------
    def calibrate(self):
        if self.last_frame is None or self.rect is None:
            self.message = "Game window not found"
            return
        mx, my = cursor_pos()
        tag = vision.name_tag_template(self.last_frame, mx - self.rect[0], my - self.rect[1])
        if tag is None:
            self.message = "No name tag under the mouse - try again"
            return
        self.tag = tag
        cv2.imwrite(TAG_PATH, tag)
        self.message = "Name tag saved!"

    def toggle_overlay(self):
        self.overlay_on = not self.overlay_on

    def toggle_autojump(self):
        self.autojump = not self.autojump
        self.message = "Auto-jump ON" if self.autojump else "Auto-jump OFF"

    def quit(self):
        self.running = False

    # ---------- main loop ----------
    def tick(self):
        if not self.running:
            self.keys.release_all()
            self.root.destroy()
            return
        start = time.perf_counter()
        try:
            self.step()
        except Exception as e:  # never leave keys stuck down
            self.keys.release_all()
            self.message = f"Error: {e}"
        spent = (time.perf_counter() - start) * 1000
        self.root.after(max(1, int(1000 / self.cfg["fps"] - spent)), self.tick)

    def step(self):
        rect = find_game_rect(self.cfg["window_title"])
        if rect is None:
            self.rect = None
            self.canvas.delete("all")
            return
        if rect != self.rect:
            self.rect = rect
            self.root.geometry(f"{rect[2]}x{rect[3]}+{rect[0]}+{rect[1]}")
        shot = self.sct.grab({"left": rect[0], "top": rect[1], "width": rect[2], "height": rect[3]})
        frame = np.ascontiguousarray(np.asarray(shot)[:, :, :3])
        self.last_frame = frame
        now = time.perf_counter()

        branches = vision.find_branches(frame)
        scale = vision.scale_of(branches)
        platforms = vision.merge_platforms(branches)
        ropes = vision.find_ropes(frame)
        if self.phys is None and branches:
            self.phys = planner.Physics(scale)
            self.learner = planner.JumpLearner(self.phys)
        char = vision.find_character(frame, self.tag)

        cur = target = plan = None
        if char and self.phys:
            x, feet, _ = char
            cur = planner.standing_on(platforms, x, feet, scale)
            on_rope = any(abs(x - r.x) < 12 * scale and r.top - 10 * scale < feet < r.bottom + 30 * scale
                          for r in ropes) and cur is None
            # rope climbing is not a jump - don't learn physics from it
            self.learner.feed(now, x, feet, grounded=cur is not None or on_rope)
            if cur:
                target, plan = planner.pick_target(platforms, cur, self.phys, scale, ropes)
                if plan:
                    latency = self.cfg["latency_ms"] / 1000 + 1 / self.cfg["fps"]
                    plan["press_x"] = plan["takeoff_x"] - plan["direction"] * self.phys.vx * latency
            self.act(char, cur, plan, scale, now, on_rope)
            self.prev_x = x
        else:
            self.keys.release_all()

        self.draw(platforms, ropes, char, cur, target, plan, scale)

    def act(self, char, cur, plan, scale, now, on_rope=False):
        x = char[0]
        assist = keyboard.is_pressed(self.cfg["assist_hold_key"])
        if not assist and self.assisting:
            self.keys.release_all()
        self.assisting = assist
        jump_key = self.cfg["jump_key"]
        up_key = self.cfg["up_key"]
        cooled = now - self.last_jump > 0.35

        if cur is None:
            # on a rope: keep climbing (holding up at the top steps onto the platform).
            # in the air: keep holding the same arrow (MapleStory has no air control anyway)
            if assist and on_rope:
                self.keys.walk(0)
                self.keys.hold(up_key)
            return
        if up_key in self.keys.held and now - self.last_jump > 0.6:
            self.keys.release(up_key)   # landed after a climb / missed grab
        if assist and plan and plan.get("rope"):
            # walk under the rope, then jump while holding up to grab it
            if abs(x - plan["takeoff_x"]) > 5 * scale:
                self.keys.walk(1 if plan["takeoff_x"] > x else -1)
            else:
                self.keys.walk(0)
                if cooled:
                    self.keys.hold(up_key)
                    self.keys.tap(jump_key)
                    self.last_jump = now
            return
        if not plan:
            if assist:
                self.keys.walk(0)
            return

        d = plan["direction"]
        crossed = self.prev_x is not None and (
            (d > 0 and self.prev_x < plan["press_x"] <= x) or
            (d < 0 and self.prev_x > plan["press_x"] >= x))

        if assist:
            if d == 0:
                if abs(x - plan["takeoff_x"]) > 4 * scale:
                    self.keys.walk(1 if plan["takeoff_x"] > x else -1)
                else:
                    self.keys.walk(0)
                    if cooled:
                        self.keys.tap(jump_key)
                        self.last_jump = now
                return
            behind = (plan["press_x"] - x) * d   # >0 = still before the jump point
            if crossed and cooled:
                self.keys.walk(d)
                self.keys.tap(jump_key)
                self.last_jump = now
            elif behind < 0:                     # past the spot: back up for a run-up
                self.keys.walk(-d)
            elif behind < self.cfg["runup_px"] * scale and d not in self._walking():
                self.keys.walk(-d)               # too close to start walking - back up a bit more
            else:
                self.keys.walk(d)
        elif self.autojump and d != 0 and crossed and cooled:
            self.keys.tap(jump_key)
            self.last_jump = now

    def _walking(self):
        out = set()
        if self.cfg["left_key"] in self.keys.held:
            out.add(-1)
        if self.cfg["right_key"] in self.keys.held:
            out.add(1)
        return out

    # ---------- drawing ----------
    def draw(self, platforms, ropes, char, cur, target, plan, scale):
        c = self.canvas
        c.delete("all")
        if not self.overlay_on:
            return
        r = max(3, int(4 * scale))
        for b in platforms:
            c.create_line(b.left, b.top, b.right, b.top, fill="#00e5ff", width=2)
            c.create_line(b.cx, b.top - 10 * scale, b.cx, b.top + 6 * scale, fill="#ff2d2d", width=2)
            c.create_oval(b.cx - r, b.top - r, b.cx + r, b.top + r, fill="#ff2d2d", outline="")
        for rp in ropes:
            c.create_line(rp.x, rp.top, rp.x, rp.bottom, fill="#b388ff", width=2, dash=(6, 4))
        if isinstance(target, vision.Rope):
            c.create_line(target.x, target.top, target.x, target.bottom, fill="#ffd400", width=4)
            c.create_text(target.x, target.bottom + 16 * scale, text="CLIMB", fill="#ffd400",
                          font=("Segoe UI", max(9, int(10 * scale)), "bold"))
        elif target:
            c.create_line(target.left, target.top, target.right, target.top, fill="#ffd400", width=4)
            c.create_text(target.cx, target.top - 22 * scale, text="NEXT", fill="#ffd400",
                          font=("Segoe UI", max(9, int(10 * scale)), "bold"))
        if cur and plan:
            px = plan["takeoff_x"]
            color = "#39ff14" if plan["ok"] else "#ff8c00"
            c.create_line(px, cur.top - 40 * scale, px, cur.top + 4 * scale, fill=color, width=3)
            if not plan.get("rope"):
                pts = self.phys.arc(px, cur.top, plan["direction"], rise_floor=cur.top - target.top)
                for (x1, y1), (x2, y2) in zip(pts[::2], pts[1::2]):
                    c.create_line(x1, y1, x2, y2, fill=color, width=2)
        if char:
            x, feet, _ = char
            c.create_oval(x - r, feet - r, x + r, feet + r, outline="#ffffff", width=2)

        mode = "ASSIST" if self.assisting else ("AUTO-JUMP" if self.autojump else "overlay")
        lines = [f"Forest Helper - {mode}"]
        if self.tag is None:
            lines.append("Put the mouse on your NAME TAG and press " + self.cfg["calibrate_key"].upper())
        elif not char:
            lines.append("Character not found (name tag hidden?)")
        if self.learner:
            lines.append(f"jumps learned: {self.learner.jumps}")
        if self.message:
            lines.append(self.message)
        c.create_text(12, 12, anchor="nw", text="\n".join(lines), fill="#ffffff",
                      font=("Segoe UI", 11, "bold"))

    def run(self):
        self.root.after(100, self.tick)
        self.root.mainloop()


def main():
    if sys.platform != "win32":
        sys.exit("This helper runs on Windows (next to the game).")
    make_dpi_aware()
    Helper(load_config()).run()


if __name__ == "__main__":
    main()
