"""미니맵 감시 알리미.

**미니맵 영역 하나**를 잡아 두고, 그 위에서 두 기능이 각자 따로 켜고 꺼진다.

  1. 룬 감지      룬 아이콘(보라 마름모)이 나타나면 소리로 알린다.
  2. 캐릭터 위치  내 캐릭터(노란 원)가 허용 영역을 벗어나거나 사라지면 알린다.

둘 다 "색 + 크기/모양 + 흰 테두리" 로 찾고, "같은 자리에서 N초에 걸쳐 계속 보일 것"
을 요구한다. 지형 얼룩은 1초를 못 버티기 때문에 이 조건 하나로 오탐이 걸러진다.
"""

from __future__ import annotations

import os
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

import config as cfgmod
from capture import ScreenCapture, enable_dpi_awareness
from detector import IconRule, draw_boxes
from notifier import CHAR_WAV, RUNE_WAV, Notifier, ensure_all
from region_select import ImagePicker, select_region
from watch import CharWatch, RuneWatch, inside

APP_DIR = os.path.dirname(os.path.abspath(__file__))
FONT = ("맑은 고딕", 9)
GREEN, YELLOW, BLUE, RED = (0, 255, 0), (0, 220, 255), (255, 160, 60), (0, 80, 255)


# --------------------------------------------------------------------------- #
# 감지 워커 스레드
# --------------------------------------------------------------------------- #
class Watcher(threading.Thread):
    """미니맵을 한 번 캡처해서 두 기능이 같이 본다. 결과는 큐로 GUI 에 보낸다."""

    def __init__(self, cfg: cfgmod.AppConfig, out_q: "queue.Queue"):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.q = out_q
        self.stop_event = threading.Event()

    def run(self) -> None:
        cap = None
        try:
            cap = ScreenCapture()
            self._loop(cap)
        except Exception as exc:  # pragma: no cover
            self.q.put(("error", f"감지 중 오류: {exc}"))
        finally:
            if cap is not None:
                cap.close()
            self.q.put(("stopped", None))

    # ------------------------------------------------------------------ #
    def _loop(self, cap: ScreenCapture) -> None:
        cfg = self.cfg
        interval = max(0.05, float(cfg.interval))
        region = tuple(cfg.region)
        allow = cfg.allow_box()

        rune_watch = RuneWatch(cfg.rune_hold, cfg.rune_repeat,
                               cfg.rune_repeat_interval) if cfg.rune_on else None

        char_on = bool(cfg.char_on and allow)
        char_watch = CharWatch(allow, cfg.char_hold, cfg.char_out_after,
                               cfg.char_lost_after,
                               cfg.char_repeat_interval) if char_on else None

        frames = 0
        t_stat = time.time()

        while not self.stop_event.is_set():
            t0 = now = time.time()
            try:
                img = cap.grab(region)
            except Exception as exc:
                self.q.put(("error", f"화면 캡처 실패: {exc}"))
                return

            rune_dets: list = []
            char_dets: list = []
            held: list = []
            me = None

            # 규칙과 시간 조건은 매번 다시 읽는다. 감지를 돌리는 중에 [세부…] 로
            # 값을 바꿔도 바로 먹히도록 (예전엔 중지했다 다시 시작해야 했다)
            if rune_watch is not None:
                rune_watch.hold = cfg.rune_hold
                rune_watch.repeat = cfg.rune_repeat
                rune_watch.repeat_interval = cfg.rune_repeat_interval
            if char_watch is not None:
                char_watch.hold = cfg.char_hold
                char_watch.out_after = cfg.char_out_after
                char_watch.lost_after = cfg.char_lost_after
                char_watch.repeat_interval = cfg.char_repeat_interval

            if rune_watch is not None:
                rune_dets = cfg.rune().detect(img)
                ev = rune_watch.update(rune_dets, now)
                held = rune_watch.held
                if ev is not None:
                    self.q.put(("event", ev))

            if char_watch is not None:
                char_dets = cfg.char().detect(img)
                ev = char_watch.update(char_dets, now)
                me = char_watch.me
                if ev is not None:
                    self.q.put(("event", ev))

            if cfg.show_preview:
                self.q.put(("preview", (img, rune_dets, char_dets, held, me, allow)))

            frames += 1
            if now - t_stat >= 2.0:
                self.q.put(("fps", frames / (now - t_stat)))
                frames, t_stat = 0, now

            time.sleep(max(0.0, interval - (time.time() - t0)))


# --------------------------------------------------------------------------- #
# 색 조건 세부 설정 창
# --------------------------------------------------------------------------- #
class RuleDialog(tk.Toplevel):
    """색/크기/모양 조건을 손으로 만지는 창. 평소에는 열 일이 없다."""

    ROWS = [
        ("h_min", "색조 H 최소", 0, 179, 1), ("h_max", "색조 H 최대", 0, 179, 1),
        ("s_min", "채도 S 최소", 0, 255, 1), ("s_max", "채도 S 최대", 0, 255, 1),
        ("v_min", "명도 V 최소", 0, 255, 1), ("v_max", "명도 V 최대", 0, 255, 1),
        ("min_area", "최소 픽셀 수", 1, 100000, 1),
        ("max_area", "최대 픽셀 수", 1, 100000, 1),
        ("min_side", "최소 변(px)", 1, 500, 1), ("max_side", "최대 변(px)", 1, 500, 1),
        ("max_aspect", "가로세로 비율 허용", 1.0, 5.0, 0.1),
        ("fill_min", "채움비 최소", 0.0, 1.0, 0.05),
        ("fill_max", "채움비 최대", 0.0, 1.0, 0.05),
        ("border_ratio", "밝은 테두리 비율", 0.0, 1.0, 0.02),
        ("white_ratio", "흰 테두리 비율", 0.0, 1.0, 0.02),
    ]

    def __init__(self, master, title: str, rule: dict, preset_key: str):
        super().__init__(master)
        self.title(title)
        self.resizable(False, False)
        self.result = None
        self.vars = {}

        top = ttk.Frame(self)
        top.grid(row=0, column=0, columnspan=4, sticky="w", padx=10, pady=(10, 4))
        ttk.Label(top, text="기본값으로 되돌리기", font=FONT).pack(side="left")
        ttk.Button(top, text=preset_key, width=18,
                   command=lambda: self._apply(cfgmod.PRESETS[preset_key])).pack(
            side="left", padx=6)

        for i, (key, label, lo, hi, step) in enumerate(self.ROWS):
            r, c = divmod(i, 2)
            f = ttk.Frame(self)
            f.grid(row=1 + r, column=c, sticky="w", padx=10, pady=2)
            var = (tk.DoubleVar if isinstance(step, float) and step < 1
                   else tk.IntVar)()
            var.set(rule.get(key, 0))
            self.vars[key] = var
            ttk.Label(f, text=label, width=16, font=FONT).pack(side="left")
            ttk.Spinbox(f, textvariable=var, from_=lo, to=hi, increment=step,
                        width=7).pack(side="left")

        f = ttk.Frame(self)
        f.grid(row=99, column=0, columnspan=2, sticky="e", padx=10, pady=10)
        ttk.Button(f, text="확인", command=self._ok).pack(side="left")
        ttk.Button(f, text="취소", command=self.destroy).pack(side="left", padx=6)

        self.transient(master)
        self.grab_set()

    def _apply(self, d: dict) -> None:
        for k, var in self.vars.items():
            if k in d:
                var.set(d[k])

    def _ok(self) -> None:
        out = {}
        for k, var in self.vars.items():
            try:
                out[k] = var.get()
            except tk.TclError:
                out[k] = getattr(IconRule(), k)
        self.result = out
        self.destroy()


# --------------------------------------------------------------------------- #
# 한 기능(룬 / 캐릭터)의 소리 설정 묶음
# --------------------------------------------------------------------------- #
class SoundRow:
    def __init__(self, parent, default_wav: str, notifier: Notifier):
        self.default = default_wav
        self.notifier = notifier
        self.path = tk.StringVar()
        self.beeps = tk.IntVar(value=2)

        f = ttk.Frame(parent)
        self.frame = f
        ttk.Label(f, text="소리", font=FONT).pack(side="left")
        ttk.Entry(f, textvariable=self.path, width=22).pack(side="left", padx=4)
        ttk.Button(f, text="찾기", width=5, command=self._pick).pack(side="left")
        ttk.Button(f, text="기본음", width=6, command=self._reset).pack(
            side="left", padx=2)
        ttk.Button(f, text="듣기", width=5, command=self.play).pack(side="left")
        ttk.Label(f, text="반복", font=FONT).pack(side="left", padx=(8, 2))
        ttk.Spinbox(f, textvariable=self.beeps, from_=1, to=10, increment=1,
                    width=3).pack(side="left")

    def _pick(self) -> None:
        p = filedialog.askopenfilename(
            title="알림음 선택",
            filetypes=[("WAV 파일", "*.wav"), ("모든 파일", "*.*")])
        if p:
            self.path.set(p)

    def _reset(self) -> None:
        self.path.set("")

    def play(self) -> None:
        self.notifier.play(self.path.get().strip(), int(self.beeps.get()),
                           fallback=self.default)


# --------------------------------------------------------------------------- #
# GUI
# --------------------------------------------------------------------------- #
class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("미니맵 감시 알리미")
        self.resizable(False, False)

        ensure_all()
        self.cfg = cfgmod.load()
        self.notifier = Notifier()
        self.q: "queue.Queue" = queue.Queue()
        self.watcher: Watcher | None = None
        self._preview_img = None
        self._rune_hits = 0
        self._char_hits = 0

        self._build()
        self._apply_cfg()
        self.after(60, self._pump)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.attributes("-topmost", bool(self.cfg.always_on_top))

    # ------------------------------------------------------------------ #
    # 화면 구성
    # ------------------------------------------------------------------ #
    def _build(self) -> None:
        root = ttk.Frame(self)
        root.grid(row=0, column=0, sticky="nsew")
        left = ttk.Frame(root)
        left.grid(row=0, column=0, sticky="n")
        right = ttk.Frame(root)
        right.grid(row=0, column=1, sticky="n", padx=(4, 10), pady=10)

        self._build_region(left)
        self._build_rune(left)
        self._build_char(left)
        self._build_common(left)
        self._build_right(right)

    # -- 미니맵 영역 (두 기능 공통) -------------------------------------- #
    def _build_region(self, parent) -> None:
        g = ttk.LabelFrame(parent, text="미니맵 영역 (두 기능이 같이 씁니다)")
        g.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))

        f = ttk.Frame(g)
        f.pack(anchor="w", padx=10, pady=(8, 8))
        ttk.Button(f, text="영역 지정", width=12,
                   command=self.on_region).pack(side="left")
        self.var_region = tk.StringVar(value="지정 안 됨")
        ttk.Label(f, textvariable=self.var_region, font=FONT).pack(
            side="left", padx=(10, 0))

    # -- 1. 룬 ---------------------------------------------------------- #
    def _build_rune(self, parent) -> None:
        g = ttk.LabelFrame(parent, text="1. 룬 감지")
        g.grid(row=1, column=0, sticky="ew", padx=10, pady=5)

        f = ttk.Frame(g)
        f.pack(anchor="w", padx=10, pady=(8, 2))
        self.var_rune_on = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="사용", variable=self.var_rune_on).pack(side="left")
        ttk.Button(f, text="지금 확인", width=10,
                   command=self.on_rune_check).pack(side="left", padx=(12, 4))
        ttk.Button(f, text="색 다시 잡기", width=12,
                   command=self.on_rune_pick).pack(side="left")
        ttk.Button(f, text="세부…", width=7,
                   command=self.on_rune_detail).pack(side="left", padx=4)

        f = ttk.Frame(g)
        f.pack(anchor="w", padx=10, pady=2)
        ttk.Label(f, text="이 시간 이상 계속 보이면 알림", font=FONT).pack(side="left")
        self.var_rune_hold = tk.DoubleVar(value=1.5)
        ttk.Spinbox(f, textvariable=self.var_rune_hold, from_=0.2, to=30,
                    increment=0.1, width=5).pack(side="left", padx=4)
        ttk.Label(f, text="초", font=FONT).pack(side="left")

        f = ttk.Frame(g)
        f.pack(anchor="w", padx=10, pady=2)
        self.var_rune_repeat = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="룬이 남아 있는 동안 다시 알림",
                        variable=self.var_rune_repeat).pack(side="left")
        self.var_rune_rep_iv = tk.DoubleVar(value=30.0)
        ttk.Spinbox(f, textvariable=self.var_rune_rep_iv, from_=2, to=600,
                    increment=5, width=5).pack(side="left", padx=4)
        ttk.Label(f, text="초마다", font=FONT).pack(side="left")

        self.rune_sound = SoundRow(g, RUNE_WAV, self.notifier)
        self.rune_sound.frame.pack(anchor="w", padx=10, pady=(2, 10))

    # -- 2. 캐릭터 ------------------------------------------------------ #
    def _build_char(self, parent) -> None:
        g = ttk.LabelFrame(parent, text="2. 캐릭터 위치 감지")
        g.grid(row=2, column=0, sticky="ew", padx=10, pady=5)

        f = ttk.Frame(g)
        f.pack(anchor="w", padx=10, pady=(8, 2))
        self.var_char_on = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="사용", variable=self.var_char_on).pack(side="left")
        ttk.Button(f, text="지금 확인", width=10,
                   command=self.on_char_check).pack(side="left", padx=(12, 4))
        ttk.Button(f, text="색 다시 잡기", width=12,
                   command=self.on_char_pick).pack(side="left")
        ttk.Button(f, text="세부…", width=7,
                   command=self.on_char_detail).pack(side="left", padx=4)

        f = ttk.Frame(g)
        f.pack(anchor="w", padx=10, pady=2)
        ttk.Button(f, text="허용 영역 지정", width=14,
                   command=self.on_char_allow).pack(side="left")
        self.var_char_allow = tk.StringVar(value="허용 영역 없음")
        ttk.Label(f, textvariable=self.var_char_allow, font=FONT).pack(
            side="left", padx=(10, 0))

        f = ttk.Frame(g)
        f.pack(anchor="w", padx=10, pady=2)
        ttk.Label(f, text="허용 영역 밖에", font=FONT).pack(side="left")
        self.var_char_out = tk.DoubleVar(value=3.0)
        ttk.Spinbox(f, textvariable=self.var_char_out, from_=0.2, to=120,
                    increment=0.5, width=5).pack(side="left", padx=4)
        ttk.Label(f, text="초 이상 있으면 알림", font=FONT).pack(side="left")

        f = ttk.Frame(g)
        f.pack(anchor="w", padx=10, pady=2)
        ttk.Label(f, text="캐릭터가", font=FONT).pack(side="left")
        self.var_char_lost = tk.DoubleVar(value=10.0)
        ttk.Spinbox(f, textvariable=self.var_char_lost, from_=0, to=300,
                    increment=1, width=5).pack(side="left", padx=4)
        ttk.Label(f, text="초 이상 안 보여도 알림 (0이면 안 알림)",
                  font=FONT).pack(side="left")

        f = ttk.Frame(g)
        f.pack(anchor="w", padx=10, pady=2)
        ttk.Label(f, text="돌아올 때까지", font=FONT).pack(side="left")
        self.var_char_rep_iv = tk.DoubleVar(value=10.0)
        ttk.Spinbox(f, textvariable=self.var_char_rep_iv, from_=2, to=600,
                    increment=1, width=5).pack(side="left", padx=4)
        ttk.Label(f, text="초마다 다시 알림", font=FONT).pack(side="left")

        self.char_sound = SoundRow(g, CHAR_WAV, self.notifier)
        self.char_sound.frame.pack(anchor="w", padx=10, pady=(2, 10))

    # -- 3. 동작 -------------------------------------------------------- #
    def _build_common(self, parent) -> None:
        g = ttk.LabelFrame(parent, text="3. 동작")
        g.grid(row=3, column=0, sticky="ew", padx=10, pady=5)

        f = ttk.Frame(g)
        f.pack(anchor="w", padx=10, pady=(8, 8))
        ttk.Label(f, text="감지 주기(초)", font=FONT).pack(side="left")
        self.var_interval = tk.DoubleVar(value=0.25)
        ttk.Spinbox(f, textvariable=self.var_interval, from_=0.05, to=5,
                    increment=0.05, width=5).pack(side="left", padx=4)
        self.var_top = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="항상 위", variable=self.var_top,
                        command=self._sync_top).pack(side="left", padx=(10, 0))
        self.var_preview = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="미리보기", variable=self.var_preview).pack(
            side="left", padx=(8, 0))

        f = ttk.Frame(parent)
        f.grid(row=4, column=0, sticky="ew", padx=10, pady=(0, 10))
        self.btn_start = ttk.Button(f, text="감지 시작", command=self.on_start)
        self.btn_start.pack(side="left")
        self.btn_stop = ttk.Button(f, text="중지", command=self.on_stop,
                                   state="disabled")
        self.btn_stop.pack(side="left", padx=6)
        ttk.Button(f, text="설정 저장", command=self.on_save).pack(side="right")

    # -- 오른쪽 --------------------------------------------------------- #
    def _build_right(self, parent) -> None:
        ttk.Label(parent, text="미리보기", font=FONT).pack(anchor="w")
        self.canvas = tk.Canvas(parent, width=330, height=210, bg="#1b1b1b",
                                highlightthickness=1, highlightbackground="#555")
        self.canvas.pack(pady=(2, 2))
        ttk.Label(parent, text="초록 = 룬,  노랑 = 캐릭터,  파랑 = 허용 영역",
                  font=FONT).pack(anchor="w")
        self.var_status = tk.StringVar(value="대기 중")
        ttk.Label(parent, textvariable=self.var_status, font=FONT).pack(
            anchor="w", pady=(4, 0))

        ttk.Label(parent, text="기록", font=FONT).pack(anchor="w", pady=(8, 0))
        self.log = tk.Text(parent, width=46, height=16, font=("Consolas", 9),
                           state="disabled", bg="#111", fg="#ddd")
        self.log.pack()

    # ------------------------------------------------------------------ #
    # 설정 <-> 위젯
    # ------------------------------------------------------------------ #
    def _apply_cfg(self) -> None:
        c = self.cfg
        self.var_rune_on.set(c.rune_on)
        self.var_rune_hold.set(c.rune_hold)
        self.var_rune_repeat.set(c.rune_repeat)
        self.var_rune_rep_iv.set(c.rune_repeat_interval)
        self.rune_sound.path.set(c.rune_wav)
        self.rune_sound.beeps.set(c.rune_beeps)

        self.var_char_on.set(c.char_on)
        self.var_char_out.set(c.char_out_after)
        self.var_char_lost.set(c.char_lost_after)
        self.var_char_rep_iv.set(c.char_repeat_interval)
        self.char_sound.path.set(c.char_wav)
        self.char_sound.beeps.set(c.char_beeps)

        self.var_interval.set(c.interval)
        self.var_top.set(c.always_on_top)
        self.var_preview.set(c.show_preview)
        self._labels()

    def _collect(self) -> cfgmod.AppConfig:
        c = self.cfg

        def num(var, fallback):
            try:
                return var.get()
            except tk.TclError:
                return fallback

        c.rune_on = bool(self.var_rune_on.get())
        c.rune_hold = float(num(self.var_rune_hold, 1.5))
        c.rune_repeat = bool(self.var_rune_repeat.get())
        c.rune_repeat_interval = float(num(self.var_rune_rep_iv, 30.0))
        c.rune_wav = self.rune_sound.path.get().strip()
        c.rune_beeps = int(num(self.rune_sound.beeps, 2))

        c.char_on = bool(self.var_char_on.get())
        c.char_out_after = float(num(self.var_char_out, 3.0))
        c.char_lost_after = float(num(self.var_char_lost, 10.0))
        c.char_repeat_interval = float(num(self.var_char_rep_iv, 10.0))
        c.char_wav = self.char_sound.path.get().strip()
        c.char_beeps = int(num(self.char_sound.beeps, 2))

        c.interval = float(num(self.var_interval, 0.25))
        c.always_on_top = bool(self.var_top.get())
        c.show_preview = bool(self.var_preview.get())
        return c

    def _labels(self) -> None:
        r = self.cfg.region
        self.var_region.set(
            f"x={r[0]}, y={r[1]}, 크기 {r[2]} x {r[3]}" if r else "지정 안 됨")
        a = self.cfg.allow_box()
        self.var_char_allow.set(
            f"{a[2]} x {a[3]} (왼쪽 위 {a[0]},{a[1]})" if a else "허용 영역 없음")

    def _sync_top(self) -> None:
        self.attributes("-topmost", bool(self.var_top.get()))

    # ------------------------------------------------------------------ #
    # 영역 지정
    # ------------------------------------------------------------------ #
    def on_region(self) -> None:
        self.withdraw()
        self.update_idletasks()
        time.sleep(0.25)
        try:
            r = select_region(self, "미니맵을 드래그로 감싸세요 (테두리 안쪽만)")
        finally:
            self.deiconify()
        if not r:
            return
        old = self.cfg.region
        self.cfg.region = list(r)
        if old and (old[2], old[3]) != (r[2], r[3]) and self.cfg.char_allow:
            self.cfg.char_allow = None
            self._log("영역 크기가 바뀌어 허용 영역을 지웠습니다. 다시 지정하세요.")
        self._labels()
        self._log(f"미니맵 영역: {r}")

    def on_char_allow(self) -> None:
        """확대된 미니맵 위에서 허용 범위를 드래그한다(작아서 화면에선 어렵다)."""
        img = self._grab()
        if img is None:
            return
        picker = ImagePicker(self, img, mode="rect", zoom=8,
                             title="허용 영역 지정",
                             tip="캐릭터가 여기 안에 있어야 정상인 범위를 드래그하세요."
                                 "  (파란 점선 = 지금 설정)",
                             preset=self.cfg.allow_box())
        self.wait_window(picker)
        if not picker.result:
            return
        self.cfg.char_allow = list(picker.result)
        self.var_char_on.set(True)
        self._labels()
        self._log(f"허용 영역: {tuple(picker.result)} (미니맵 영역 기준)")

    # ------------------------------------------------------------------ #
    # 확인 / 색 잡기 / 세부 설정
    # ------------------------------------------------------------------ #
    def _grab(self):
        if not self.cfg.region:
            messagebox.showwarning("영역 없음", "먼저 미니맵 영역을 지정하세요.")
            return None
        cap = ScreenCapture()
        try:
            return cap.grab(tuple(self.cfg.region))
        finally:
            cap.close()

    def on_rune_check(self) -> None:
        img = self._grab()
        if img is None:
            return
        dets = self._collect().rune().detect(img)
        self._draw(img, dets, [], [], None, self.cfg.allow_box())
        self._log(f"룬 확인: {len(dets)}개" +
                  (f"  (테두리 {dets[0].ring:.2f}, 채움 {dets[0].fill:.2f})"
                   if dets else "  — 지금은 룬이 없거나 색 조건이 안 맞습니다"))

    def on_char_check(self) -> None:
        img = self._grab()
        if img is None:
            return
        cfg = self._collect()
        dets = cfg.char().detect(img)
        allow = cfg.allow_box()
        self._draw(img, [], dets, [], dets[0] if dets else None, allow)
        msg = f"캐릭터 확인: {len(dets)}개"
        if dets and allow:
            msg += "  → 허용 영역 " + ("안" if inside(allow, dets[0].center) else "밖")
        elif dets:
            msg += "  (허용 영역이 아직 없습니다)"
        self._log(msg)

    def _pick_color(self, rule_key: str, what: str) -> None:
        """아이콘 한가운데를 클릭하면 색과 크기 조건을 그 아이콘에 맞춘다."""
        img = self._grab()
        if img is None:
            return
        picker = ImagePicker(self, img, mode="point", zoom=8,
                             title=f"{what} 색 다시 잡기",
                             tip=f"{what} 아이콘 한가운데를 클릭하세요.")
        self.wait_window(picker)
        if not picker.result:
            return
        x, y = picker.result
        rule = dict(getattr(self.cfg, rule_key))
        hsv = cv2.cvtColor(img[max(0, y - 1):y + 2, max(0, x - 1):x + 2],
                           cv2.COLOR_BGR2HSV).reshape(-1, 3)
        h, s, v = [int(round(float(c))) for c in hsv.mean(axis=0)]
        # 압축·안티앨리어싱 때문에 같은 아이콘도 프레임마다 색이 조금씩 흔들린다
        rule.update(h_min=max(0, h - 9), h_max=min(179, h + 9),
                    s_min=max(0, s - 60), s_max=255,
                    v_min=max(0, v - 60), v_max=255)
        msg = f"색 HSV({h},{s},{v})"

        probe = IconRule(**rule)
        mask = probe.build_mask(img)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        lid = int(labels[y, x])
        if lid > 0:
            bx, by, bw, bh, area = (int(t) for t in stats[lid])
            rule.update(min_area=max(5, int(area * 0.55)),
                        max_area=int(area * 4) + 10,
                        min_side=max(2, int(min(bw, bh) * 0.6)),
                        max_side=int(max(bw, bh) * 2.2) + 2)
            msg += f" / 크기 {bw}x{bh}({area}px)"
        else:
            msg += " / 크기는 그대로 (클릭한 점이 색 범위 밖입니다)"
        setattr(self.cfg, rule_key, rule)
        self._log(f"{what} 색 다시 잡기 — {msg}")

    def on_rune_pick(self) -> None:
        self._pick_color("rune_rule", "룬")

    def on_char_pick(self) -> None:
        self._pick_color("char_rule", "캐릭터")

    def _detail(self, rule_key: str, title: str, preset: str) -> None:
        dlg = RuleDialog(self, title, getattr(self.cfg, rule_key), preset)
        self.wait_window(dlg)
        if dlg.result:
            setattr(self.cfg, rule_key, {**getattr(self.cfg, rule_key), **dlg.result})
            live = " (감지 중이라 바로 적용됩니다)" if self.watcher and                 self.watcher.is_alive() else ""
            self._log(f"{title} 저장됨{live}")

    def on_rune_detail(self) -> None:
        self._detail("rune_rule", "룬 색 조건", "룬 (보라 마름모)")

    def on_char_detail(self) -> None:
        self._detail("char_rule", "캐릭터 색 조건", "내 캐릭터 (노란 원)")

    # ------------------------------------------------------------------ #
    # 시작 / 중지 / 저장
    # ------------------------------------------------------------------ #
    def on_save(self) -> None:
        cfgmod.save(self._collect())
        self._log("설정을 settings.json 에 저장했습니다.")

    def on_start(self) -> None:
        cfg = self._collect()
        if not cfg.region:
            messagebox.showwarning("영역 없음", "먼저 미니맵 영역을 지정하세요.")
            return
        if not (cfg.rune_on or cfg.char_on):
            messagebox.showwarning("켜진 기능 없음",
                                   "룬 감지나 캐릭터 위치 감지 중 하나는 켜야 합니다.")
            return
        if cfg.char_on and not cfg.allow_box():
            messagebox.showwarning("허용 영역 없음",
                                   "캐릭터 위치 감지를 쓰려면 허용 영역을 지정하세요.")
            return
        cfgmod.save(cfg)
        self._rune_hits = self._char_hits = 0
        self.watcher = Watcher(cfg, self.q)
        self.watcher.start()
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.var_status.set("감지 중…")
        on = [n for n, v in (("룬", cfg.rune_on), ("캐릭터", cfg.char_on)) if v]
        self._log(f"─ 감지 시작 ({' + '.join(on)}) ─")

    def on_stop(self) -> None:
        if self.watcher:
            self.watcher.stop_event.set()
        self.btn_stop.configure(state="disabled")

    def _on_close(self) -> None:
        try:
            cfgmod.save(self._collect())
        except Exception:
            pass
        if self.watcher:
            self.watcher.stop_event.set()
        self.destroy()

    # ------------------------------------------------------------------ #
    # 큐 처리
    # ------------------------------------------------------------------ #
    def _pump(self) -> None:
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "preview":
                    if self.var_preview.get():
                        self._draw(*payload)
                elif kind == "event":
                    self._on_event(payload)
                elif kind == "log":
                    self._log(payload)
                elif kind == "fps":
                    self.var_status.set(f"감지 중…  {payload:.1f} 회/초")
                elif kind == "error":
                    self._log("! " + str(payload))
                elif kind == "stopped":
                    self.btn_start.configure(state="normal")
                    self.btn_stop.configure(state="disabled")
                    self.var_status.set("정지됨")
                    self._log("─ 감지 중지 ─")
        except queue.Empty:
            pass
        self.after(60, self._pump)

    def _on_event(self, ev) -> None:
        if ev.kind == "rune":
            self._rune_hits += 1
            self.rune_sound.play()
            tag = "★ 룬 발견!" if ev.first else "★ 룬 계속 있음"
            self._log(f"{tag} ({ev.count}개, {ev.pos}, {ev.secs}초 유지)  "
                      f"누적 {self._rune_hits}회")
        elif ev.kind == "gone":
            self._log("룬이 사라졌습니다.")
        elif ev.kind == "back":
            self._log(f"캐릭터가 허용 영역으로 돌아왔습니다. {ev.pos}")
        elif ev.kind == "out":
            self._char_hits += 1
            self.char_sound.play()
            self._log(f"▲ 캐릭터가 허용 영역 밖! {ev.pos} ({ev.secs}초째)  "
                      f"누적 {self._char_hits}회")
        elif ev.kind == "lost":
            self._char_hits += 1
            self.char_sound.play()
            self._log(f"▲ 캐릭터가 안 보입니다 ({ev.secs}초째)  "
                      f"누적 {self._char_hits}회")

    # ------------------------------------------------------------------ #
    def _draw(self, bgr: np.ndarray, rune_dets, char_dets, held, me, allow) -> None:
        """미니맵 하나에 룬·캐릭터·허용 영역을 겹쳐 그린다."""
        view = draw_boxes(bgr, rune_dets, GREEN)
        if allow:
            x, y, w, h = allow
            cv2.rectangle(view, (x, y), (x + w - 1, y + h - 1), BLUE, 1)
        for t in held:
            cv2.drawMarker(view, t.center, GREEN, cv2.MARKER_TILTED_CROSS, 11, 1)
        if me is not None:
            cx, cy = me.center
            ok = allow is None or inside(allow, (cx, cy))
            cv2.drawMarker(view, (cx, cy), YELLOW if ok else RED,
                           cv2.MARKER_CROSS, 9, 1)

        h0, w0 = view.shape[:2]
        scale = min(330 / max(1, w0), 210 / max(1, h0))
        nw, nh = max(1, int(w0 * scale)), max(1, int(h0 * scale))
        interp = cv2.INTER_NEAREST if scale >= 1 else cv2.INTER_AREA
        view = cv2.resize(view, (nw, nh), interpolation=interp)
        self._preview_img = ImageTk.PhotoImage(
            Image.fromarray(cv2.cvtColor(view, cv2.COLOR_BGR2RGB)))
        self.canvas.delete("all")
        self.canvas.create_image(165, 105, image=self._preview_img)

    def _log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", f"[{time.strftime('%H:%M:%S')}] {text}\n")
        self.log.see("end")
        self.log.configure(state="disabled")


def _report_crash() -> None:
    """pythonw 로 띄우면 오류가 아무 데도 안 나온다. 파일로 남기고 창으로 알린다."""
    import traceback

    text = traceback.format_exc()
    try:
        with open(os.path.join(APP_DIR, "error.log"), "a", encoding="utf-8") as f:
            stamp = time.strftime("=== %Y-%m-%d %H:%M:%S ===")
            f.write(stamp + "\n" + text + "\n")
    except Exception:
        pass
    try:
        last = text.strip().splitlines()[-1]
        messagebox.showerror(
            "실행하지 못했습니다",
            last + "\n\n자세한 내용을 프로그램 폴더의 error.log 에 저장했습니다.")
    except Exception:
        pass


def main() -> None:
    try:
        enable_dpi_awareness()
        App().mainloop()
    except Exception:
        _report_crash()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
