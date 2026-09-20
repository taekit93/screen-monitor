"""화면 위에 반투명 오버레이를 띄워 드래그로 영역을 고르는 도구."""

from __future__ import annotations

import tkinter as tk
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk

from capture import ScreenCapture


def select_region(
    master: tk.Misc,
    hint: str = "드래그해서 감지할 영역을 지정하세요",
) -> Optional[Tuple[int, int, int, int]]:
    """드래그로 고른 (x, y, w, h)를 절대 화면 좌표로 돌려준다. 취소하면 None."""
    cap = ScreenCapture()
    try:
        vx, vy, vw, vh = cap.virtual_screen()
        shot = cap.grab((vx, vy, vw, vh))
    finally:
        cap.close()

    dim = cv2.convertScaleAbs(shot, alpha=0.45, beta=0)
    rgb_dim = cv2.cvtColor(dim, cv2.COLOR_BGR2RGB)
    rgb_full = cv2.cvtColor(shot, cv2.COLOR_BGR2RGB)

    win = tk.Toplevel(master)
    win.overrideredirect(True)
    win.geometry(f"{vw}x{vh}+{vx}+{vy}")
    win.attributes("-topmost", True)
    win.configure(cursor="crosshair")

    canvas = tk.Canvas(win, width=vw, height=vh, highlightthickness=0, bd=0)
    canvas.pack(fill="both", expand=True)

    photo_dim = ImageTk.PhotoImage(Image.fromarray(rgb_dim))
    canvas.create_image(0, 0, anchor="nw", image=photo_dim)
    canvas.image = photo_dim  # GC 방지

    hint = canvas.create_text(
        vw // 2, 34,
        text=f"{hint}  ·  ESC 취소  ·  방향키로 1px 조정 후 Enter",
        fill="#ffffff", font=("맑은 고딕", 13, "bold"))
    canvas.create_rectangle(canvas.bbox(hint), fill="#000000", outline="",
                            tags="hintbg")
    canvas.tag_lower("hintbg", hint)

    state = {"x0": 0, "y0": 0, "x1": 0, "y1": 0, "drag": False,
             "result": None, "bright": None}

    rect = canvas.create_rectangle(0, 0, 0, 0, outline="#00ff88", width=2)
    label = canvas.create_text(0, 0, text="", fill="#00ff88",
                               font=("맑은 고딕", 11, "bold"), anchor="nw")

    def norm():
        x0, x1 = sorted((state["x0"], state["x1"]))
        y0, y1 = sorted((state["y0"], state["y1"]))
        return x0, y0, x1 - x0, y1 - y0

    def redraw():
        x, y, w, h = norm()
        canvas.coords(rect, x, y, x + w, y + h)
        canvas.coords(label, x, max(0, y - 22))
        canvas.itemconfigure(label, text=f"{w} x {h}   ({vx + x}, {vy + y})")
        # 선택 영역만 원래 밝기로 보여 준다
        if state["bright"] is not None:
            canvas.delete("bright")
        # 아주 큰 영역까지 매 프레임 다시 그리면 드래그가 버벅여서 테두리만 남긴다
        if 3 < w and 3 < h and w * h <= 1_500_000:
            crop = rgb_full[y:y + h, x:x + w]
            img = ImageTk.PhotoImage(Image.fromarray(crop))
            canvas.create_image(x, y, anchor="nw", image=img, tags="bright")
            state["bright"] = img
            canvas.tag_raise(rect)
            canvas.tag_raise(label)

    def on_down(e):
        state.update(x0=e.x, y0=e.y, x1=e.x, y1=e.y, drag=True)
        redraw()

    def on_move(e):
        if state["drag"]:
            state.update(x1=e.x, y1=e.y)
            redraw()

    def on_up(e):
        state["drag"] = False
        state.update(x1=e.x, y1=e.y)
        redraw()

    def finish(_=None):
        x, y, w, h = norm()
        if w >= 4 and h >= 4:
            state["result"] = (vx + x, vy + y, w, h)
        win.destroy()

    def cancel(_=None):
        state["result"] = None
        win.destroy()

    def nudge(dx, dy, resize=False):
        if resize:
            state["x1"] += dx
            state["y1"] += dy
        else:
            state["x0"] += dx
            state["y0"] += dy
            state["x1"] += dx
            state["y1"] += dy
        redraw()

    canvas.bind("<ButtonPress-1>", on_down)
    canvas.bind("<B1-Motion>", on_move)
    canvas.bind("<ButtonRelease-1>", on_up)
    win.bind("<Escape>", cancel)
    win.bind("<Return>", finish)
    win.bind("<Double-Button-1>", finish)
    for key, (dx, dy) in {"Left": (-1, 0), "Right": (1, 0),
                          "Up": (0, -1), "Down": (0, 1)}.items():
        win.bind(f"<{key}>", lambda e, d=(dx, dy): nudge(*d))
        win.bind(f"<Shift-{key}>", lambda e, d=(dx, dy): nudge(*d, resize=True))

    win.focus_force()
    win.grab_set()
    master.wait_window(win)
    return state["result"]


class ImagePicker(tk.Toplevel):
    """캡처한 영역을 확대해서 보여 주고 한 점(색 찍기) 또는 사각형(허용 영역)을 고르게 한다.

    미니맵은 화면에서 100px 남짓이라 화면 위에서 직접 고르기 어렵다.
    preset 을 주면 지금 설정된 사각형을 파란 점선으로 같이 보여 준다.
    """

    def __init__(self, master, bgr: np.ndarray, mode: str = "point", zoom: int = 6,
                 title: str = "", tip: str = "", preset=None):
        super().__init__(master)
        self.title(title or ("색 찍기" if mode == "point" else "영역 선택"))
        self.mode = mode
        self.bgr = bgr
        self.result = None

        h, w = bgr.shape[:2]
        self.zoom = max(1, min(zoom, max(1, 900 // max(1, w)), max(1, 700 // max(1, h))))
        big = cv2.resize(bgr, (w * self.zoom, h * self.zoom),
                         interpolation=cv2.INTER_NEAREST)
        self.photo = ImageTk.PhotoImage(
            Image.fromarray(cv2.cvtColor(big, cv2.COLOR_BGR2RGB)))

        tip = tip or ("아이콘 한가운데를 클릭하세요." if mode == "point"
                      else "원하는 범위를 드래그로 감싸세요.")
        tk.Label(self, text=tip, font=("맑은 고딕", 10)).pack(padx=8, pady=(8, 4))

        self.canvas = tk.Canvas(self, width=big.shape[1], height=big.shape[0],
                                highlightthickness=0, cursor="crosshair")
        self.canvas.pack(padx=8)
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        if preset:
            px, py, pw, ph = preset
            self.canvas.create_rectangle(px * self.zoom, py * self.zoom,
                                         (px + pw) * self.zoom, (py + ph) * self.zoom,
                                         outline="#4499ff", width=2, dash=(4, 3))
        self.rect = self.canvas.create_rectangle(0, 0, 0, 0,
                                                 outline="#00ff88", width=2)
        self.info = tk.Label(self, text="", font=("맑은 고딕", 9))
        self.info.pack(pady=(4, 8))

        self._p0 = None
        self.canvas.bind("<ButtonPress-1>", self._down)
        self.canvas.bind("<B1-Motion>", self._move)
        self.canvas.bind("<ButtonRelease-1>", self._up)
        self.canvas.bind("<Motion>", self._hover)
        self.bind("<Escape>", lambda e: self.destroy())
        self.transient(master)
        self.grab_set()

    def _img_xy(self, e):
        return (max(0, min(self.bgr.shape[1] - 1, e.x // self.zoom)),
                max(0, min(self.bgr.shape[0] - 1, e.y // self.zoom)))

    def _hover(self, e):
        x, y = self._img_xy(e)
        b, g, r = self.bgr[y, x]
        hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
        self.info.configure(
            text=f"({x}, {y})  RGB {r},{g},{b}   HSV {hsv[0]},{hsv[1]},{hsv[2]}")

    def _down(self, e):
        self._p0 = self._img_xy(e)
        if self.mode == "point":
            self.result = self._p0
            self.destroy()

    def _move(self, e):
        if self.mode != "rect" or self._p0 is None:
            return
        x1, y1 = self._img_xy(e)
        x0, y0 = self._p0
        self.canvas.coords(self.rect, min(x0, x1) * self.zoom, min(y0, y1) * self.zoom,
                           (max(x0, x1) + 1) * self.zoom, (max(y0, y1) + 1) * self.zoom)

    def _up(self, e):
        if self.mode != "rect" or self._p0 is None:
            return
        x1, y1 = self._img_xy(e)
        x0, y0 = self._p0
        x, y = min(x0, x1), min(y0, y1)
        w, h = abs(x1 - x0) + 1, abs(y1 - y0) + 1
        if w >= 3 and h >= 3:
            self.result = (x, y, w, h)
            self.destroy()
