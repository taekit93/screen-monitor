"""화면 캡처 유틸리티.

mss 인스턴스는 스레드마다 따로 만들어야 하므로, 이 클래스는 사용하는
스레드 안에서 생성해서 쓴다.
"""

from __future__ import annotations

import ctypes
import sys

import cv2
import mss
import numpy as np


def enable_dpi_awareness() -> None:
    """윈도우 디스플레이 배율(125%, 150% 등)이 걸려 있어도 실제 픽셀 좌표를 쓰도록 한다."""
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PER_MONITOR_AWARE
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


class ScreenCapture:
    def __init__(self) -> None:
        self._sct = mss.mss()

    def close(self) -> None:
        try:
            self._sct.close()
        except Exception:
            pass

    def virtual_screen(self):
        """모든 모니터를 감싸는 가상 화면 영역 (x, y, w, h)."""
        m = self._sct.monitors[0]
        return (m["left"], m["top"], m["width"], m["height"])

    def monitors(self):
        return [
            (m["left"], m["top"], m["width"], m["height"])
            for m in self._sct.monitors[1:]
        ]

    def grab(self, region):
        """region = (x, y, w, h) 절대 화면 좌표. BGR numpy 배열을 돌려준다."""
        x, y, w, h = region
        raw = self._sct.grab({"left": int(x), "top": int(y),
                              "width": int(w), "height": int(h)})
        arr = np.asarray(raw)  # BGRA
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
