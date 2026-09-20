"""설정 저장/불러오기.

**미니맵 영역 하나**를 두 기능이 같이 쓴다.

  룬 감지        미니맵 영역 안에 룬 아이콘이 나타나면 알림
  캐릭터 위치    미니맵 영역에서 캐릭터를 찾아, 허용 영역을 벗어나면 알림
                 (허용 영역은 미니맵 영역 기준 상대 좌표)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional

from detector import IconRule

APP_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(APP_DIR, "settings.json")


# --------------------------------------------------------------------------- #
# 기본 색 조건 — 샘플 영상 2편(합계 3시간 40분)을 전부 돌려 가며 맞춘 값
# --------------------------------------------------------------------------- #
# 룬은 흰 테두리가 뚜렷해서 white_ratio 가 오탐을 거의 다 걸러 준다.
# 0.20 은 샘플 영상 3편으로 맞춘 값이다. 밝은 맵(잔잔한 해안가)에서는 미니맵
# 지형의 보라 얼룩이 옆 아이콘의 흰 테두리를 빌려 0.16 까지 올라오는데,
# 0.20 이면 그건 걸러지면서 진짜 룬 17곳은 하나도 안 놓친다.
RUNE_RULE = dict(
    h_min=135, h_max=155, s_min=90, s_max=255, v_min=150, v_max=255,
    min_area=18, max_area=250, min_side=4, max_side=22, max_aspect=1.6,
    fill_min=0.40, fill_max=1.0,
    border_ratio=0.15, border_s_max=100, border_v_min=140,
    white_ratio=0.20, white_s_max=60, white_v_min=185, min_count=1)

# 캐릭터 아이콘은 훨씬 작아서 흰 테두리가 1px 띠에 거의 안 잡힌다
# (샘플 영상에서 중앙값 0.04). 그래서 white_ratio 는 끄고 밝기만 본다.
CHAR_RULE = dict(
    h_min=22, h_max=38, s_min=120, s_max=255, v_min=170, v_max=255,
    min_area=12, max_area=250, min_side=3, max_side=22, max_aspect=1.6,
    fill_min=0.40, fill_max=1.0,
    border_ratio=0.20, border_s_max=100, border_v_min=140,
    white_ratio=0.0, white_s_max=60, white_v_min=185, min_count=1)

PRESETS = {
    "룬 (보라 마름모)": RUNE_RULE,
    "내 캐릭터 (노란 원)": CHAR_RULE,
}


def _rule(d: dict) -> IconRule:
    r = IconRule()
    for k, v in (d or {}).items():
        if hasattr(r, k):
            setattr(r, k, v)
    return r


@dataclass
class AppConfig:
    # --- 공통 미니맵 영역 ------------------------------------------------ #
    region: Optional[List[int]] = None               # [x, y, w, h] 화면 좌표

    # --- 1. 룬 감지 ------------------------------------------------------ #
    rune_on: bool = True
    rune_rule: dict = field(default_factory=lambda: dict(RUNE_RULE))
    rune_hold: float = 1.5        # 이 시간에 걸쳐 계속 보여야 알림(초)
    rune_wav: str = ""
    rune_beeps: int = 2
    rune_repeat: bool = True      # 룬이 남아 있는 동안 다시 알릴지
    rune_repeat_interval: float = 30.0

    # --- 2. 캐릭터 위치 감지 --------------------------------------------- #
    char_on: bool = False
    char_allow: Optional[List[int]] = None           # 미니맵 영역 기준 상대 좌표
    char_rule: dict = field(default_factory=lambda: dict(CHAR_RULE))
    char_hold: float = 0.6        # 캐릭터로 인정할 최소 지속 시간(초)
    char_out_after: float = 3.0   # 허용 영역 밖에 이만큼 있으면 알림(초)
    char_lost_after: float = 10.0  # 이만큼 안 보이면 알림(초). 0이면 알리지 않음
    char_wav: str = ""
    char_beeps: int = 2
    char_repeat_interval: float = 10.0  # 돌아올 때까지 다시 알리는 간격(초)

    # --- 동작 ------------------------------------------------------------ #
    interval: float = 0.25        # 감지 주기(초)
    always_on_top: bool = True
    show_preview: bool = True

    # ------------------------------------------------------------------ #
    def rune(self) -> IconRule:
        return _rule(self.rune_rule)

    def char(self) -> IconRule:
        return _rule(self.char_rule)

    def allow_box(self):
        """허용 영역을 미니맵 영역 안으로 잘라서 돌려준다. 없으면 None."""
        if not (self.region and self.char_allow):
            return None
        _, _, rw, rh = self.region
        x, y, w, h = self.char_allow
        x, y = max(0, min(int(x), rw - 1)), max(0, min(int(y), rh - 1))
        w, h = max(1, min(int(w), rw - x)), max(1, min(int(h), rh - y))
        return (x, y, w, h)


def load(path: str = CONFIG_PATH) -> AppConfig:
    cfg = AppConfig()
    if not os.path.isfile(path):
        return cfg
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return cfg

    # 영역을 룬/캐릭터 따로 두던 시절의 설정도 읽어 준다
    if "region" not in data:
        for old in ("rune_region", "char_region"):
            if data.get(old):
                data["region"] = data[old]
                break

    for k, v in data.items():
        if not hasattr(cfg, k):
            continue
        if k in ("rune_rule", "char_rule") and isinstance(v, dict):
            base = dict(RUNE_RULE if k == "rune_rule" else CHAR_RULE)
            base.update({kk: vv for kk, vv in v.items() if hasattr(IconRule(), kk)})
            setattr(cfg, k, base)
        else:
            setattr(cfg, k, v)
    return cfg


def save(cfg: AppConfig, path: str = CONFIG_PATH) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)
