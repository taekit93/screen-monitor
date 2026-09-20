"""알림음 재생.

기능이 둘이라 기본음도 둘이다. 소리만 듣고 어느 쪽인지 알 수 있게 다르게 만들었다.

  rune.wav  룬 발견     — 올라가는 맑은 두 음
  char.wav  캐릭터 이탈 — 내려가는 낮은 세 음 (경고에 가깝게)
"""

from __future__ import annotations

import math
import os
import struct
import sys
import threading
import wave

if sys.platform == "win32":
    import winsound
else:  # pragma: no cover - 참고용
    winsound = None

SOUND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sounds")
RUNE_WAV = os.path.join(SOUND_DIR, "rune.wav")
CHAR_WAV = os.path.join(SOUND_DIR, "char.wav")

# (주파수Hz, 길이ms, 뒤에 붙일 쉼ms)
RUNE_TUNE = [(988, 110, 45), (1319, 190, 0)]                   # B5 - E6
CHAR_TUNE = [(660, 140, 30), (554, 140, 30), (440, 300, 0)]    # E5 - C#5 - A4


def _write_tune(path: str, tune) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rate = 44100
    frames = bytearray()

    def tone(freq: float, ms: int, vol: float = 0.45):
        n = int(rate * ms / 1000)
        for i in range(n):
            # 앞뒤로 짧은 페이드를 줘서 딱딱거리는 소리를 없앤다
            env = min(1.0, i / (rate * 0.006), (n - i) / (rate * 0.02))
            s = math.sin(2 * math.pi * freq * i / rate)
            s += 0.3 * math.sin(4 * math.pi * freq * i / rate)
            v = int(max(-1.0, min(1.0, s / 1.3)) * env * vol * 32767)
            frames.extend(struct.pack("<h", v))

    def silence(ms: int):
        frames.extend(bytes(2 * int(rate * ms / 1000)))

    for freq, ms, gap in tune:
        tone(freq, ms)
        if gap:
            silence(gap)

    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(frames))
    return path


def ensure_rune_wav(path: str = RUNE_WAV) -> str:
    return path if os.path.isfile(path) else _write_tune(path, RUNE_TUNE)


def ensure_char_wav(path: str = CHAR_WAV) -> str:
    return path if os.path.isfile(path) else _write_tune(path, CHAR_TUNE)


def ensure_all() -> None:
    ensure_rune_wav()
    ensure_char_wav()


class Notifier:
    """소리를 비동기로 재생한다. 재생 중이면 겹치지 않게 건너뛴다."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._busy = False

    def play(self, wav_path: str, repeat: int = 1, gap_ms: int = 120,
             fallback: str = "") -> None:
        """wav_path 가 비었거나 파일이 없으면 fallback 을 쓴다."""
        with self._lock:
            if self._busy:
                return
            self._busy = True
        threading.Thread(target=self._play,
                         args=(wav_path, repeat, gap_ms, fallback),
                         daemon=True).start()

    def _play(self, wav_path: str, repeat: int, gap_ms: int,
              fallback: str) -> None:
        try:
            path = wav_path if wav_path and os.path.isfile(wav_path) else fallback
            if not path or not os.path.isfile(path):
                path = ensure_rune_wav()
            for i in range(max(1, repeat)):
                if winsound is not None:
                    # SND_ASYNC 를 쓰지 않아야 반복 재생이 순서대로 들린다
                    winsound.PlaySound(path, winsound.SND_FILENAME)
                if i + 1 < repeat:
                    threading.Event().wait(gap_ms / 1000.0)
        except Exception:
            if winsound is not None:
                try:
                    winsound.MessageBeep()
                except Exception:
                    pass
        finally:
            with self._lock:
                self._busy = False
