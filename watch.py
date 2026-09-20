"""시간 판정.

색만 보면 어느 맵에나 잠깐씩 아이콘처럼 보이는 지형 얼룩이 있다.
샘플 영상에서 재 보니 이런 얼룩은 **길어야 0.8초**면 사라지는데,
진짜 룬은 화면에 뜨면 **몇 초에서 몇 분까지** 같은 자리에 머문다.

그래서 "한 번 보였다" 가 아니라 "같은 자리에서 N초 이상 계속 보인다" 를 조건으로 쓴다.
이 한 가지로 지형 오탐이 사실상 사라진다.

아래 RuneWatch / CharWatch 가 알림을 낼지 말지를 전부 판단한다.
GUI(main.py)와 검증 도구(tools/analyze_video.py)가 똑같이 이 클래스를 쓴다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from detector import Detection

# 룬이 잠깐 안 잡혀도(압축 노이즈로 깜빡인다) 이 시간 안에 다시 보이면
# "아직 그 룬이 있다" 로 본다. 샘플 영상에서 같은 룬이 17초나 안 잡히다가
# 다시 보인 적이 있어 넉넉히 잡았다.
RUNE_FORGET = 25.0
# 단, 같은 룬인지는 시간이 아니라 **자리**로 가른다. 이만큼 떨어진 곳에
# 나타나면 다른 룬으로 보고 새로 알린다 (먹자마자 딴 데 뜨는 경우).
RUNE_SAME_TOL = 10

# 캐릭터를 같은 아이콘으로 볼 한 프레임당 이동 허용치(px).
# 감지 주기 0.25초 기준 초당 80px 까지 따라간다.
CHAR_TOL = 20

# 아이콘이 잠깐 안 잡혀도 "같은 것이 계속 있다" 로 이어 주는 시간(초).
# 룬은 배경이 밝은 맵에서 3초에 한두 번씩만 잡히기도 해서 넉넉히 준다.
# 지형 얼룩은 어차피 한두 번 반짝이고 끝이라 이걸 늘려도 통과하지 못한다.
RUNE_GRACE = 3.0
# 캐릭터 쪽은 이 시간만큼 '실종' 판정이 늦어지므로 짧게 잡는다.
CHAR_GRACE = 1.0


@dataclass
class Track:
    """같은 자리에서 이어지고 있는 아이콘 하나."""

    x: int
    y: int
    w: int
    h: int
    since: float           # 처음 보인 시각
    last: float            # 마지막으로 보인 시각
    hits: int = 1          # 실제로 보인 횟수

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)

    @property
    def span(self) -> float:
        """처음 보인 때부터 마지막으로 **보인** 때까지의 길이.

        '지금까지 흐른 시간'(now - since)이 아니라는 게 중요하다. 그걸 쓰면
        잠깐 반짝한 지형 얼룩도 grace 동안 살아 있는 사이에 시간이 차 버린다.
        """
        return self.last - self.since


class Tracker:
    """프레임마다 감지 결과를 넣으면 같은 자리에서 이어지는 아이콘을 추적한다.

    tol   같은 아이콘으로 볼 중심 이동 허용치(px)
    grace 잠깐 안 보여도 끊긴 걸로 치지 않는 시간(초).
          영상 압축 탓에 아이콘이 프레임마다 깜빡이듯 사라져서 필요하다.
    """

    def __init__(self, tol: int = 6, grace: float = 0.6) -> None:
        self.tol = tol
        self.grace = grace
        self.tracks: List[Track] = []

    def update(self, dets: List[Detection], now: float) -> List[Track]:
        used = [False] * len(self.tracks)
        for d in dets:
            cx, cy = d.center
            best, best_dist = -1, None
            for i, t in enumerate(self.tracks):
                if used[i]:
                    continue
                tx, ty = t.center
                dist = max(abs(cx - tx), abs(cy - ty))
                if dist <= self.tol and (best_dist is None or dist < best_dist):
                    best, best_dist = i, dist
            if best >= 0:
                t = self.tracks[best]
                t.x, t.y, t.w, t.h = d.x, d.y, d.w, d.h
                t.last = now
                t.hits += 1
                used[best] = True
            else:
                self.tracks.append(Track(d.x, d.y, d.w, d.h, now, now))
                used.append(True)      # 방금 만든 것과 또 짝짓지 않도록

        self.tracks = [t for t in self.tracks if now - t.last <= self.grace]
        return self.tracks

    def held_tracks(self, now: float, hold: float,
                    min_hits: int = 3) -> List[Track]:
        """hold 초에 걸쳐 실제로 min_hits 번 이상 보인 것만 (먼저 나타난 순서대로)."""
        out = [t for t in self.tracks if t.span >= hold and t.hits >= min_hits]
        out.sort(key=lambda t: t.since)
        return out

    def nearest(self, now: float, hold: float,
                to: Optional[Tuple[int, int]]) -> Optional[Track]:
        """hold 를 넘긴 것 중 to 에 가장 가까운 것. to 가 없으면 가장 오래 버틴 것."""
        cand = self.held_tracks(now, hold)
        if not cand:
            return None
        if to is None:
            return cand[0]
        return min(cand, key=lambda t: (t.center[0] - to[0]) ** 2 +
                                       (t.center[1] - to[1]) ** 2)


def inside(box: Tuple[int, int, int, int], pt: Tuple[int, int]) -> bool:
    x, y, w, h = box
    return x <= pt[0] < x + w and y <= pt[1] < y + h


# --------------------------------------------------------------------------- #
# 알림 판단
# --------------------------------------------------------------------------- #
@dataclass
class Event:
    """알릴 일이 생겼다는 신호."""

    kind: str                       # rune / out / lost / gone / back
    pos: Optional[Tuple[int, int]] = None
    secs: float = 0.0
    first: bool = True              # 새로 생긴 일인지, 이어지는 중 재알림인지
    count: int = 1

    @property
    def is_alert(self) -> bool:
        """소리를 낼 일인지 (gone/back 은 기록에만 남긴다)."""
        return self.kind in ("rune", "out", "lost")


class RuneWatch:
    """룬이 hold 초 이상 계속 보이면 알린다. 남아 있는 동안 주기적으로 재알림."""

    def __init__(self, hold: float, repeat: bool = True,
                 repeat_interval: float = 30.0) -> None:
        self.hold = hold
        self.repeat = repeat
        self.repeat_interval = repeat_interval
        # 룬은 한 자리에 박혀 있으므로 이동 허용치는 좁게, 깜빡임은 넉넉히
        self.tracker = Tracker(tol=6, grace=RUNE_GRACE)
        self.held: List[Track] = []
        self._alerted = 0.0
        self._alerted_pos: Optional[Tuple[int, int]] = None
        self._last_seen = 0.0

    def update(self, dets: List[Detection], now: float) -> Optional[Event]:
        self.tracker.update(dets, now)
        self.held = self.tracker.held_tracks(now, self.hold)
        if self.held:
            t = self.held[0]
            self._last_seen = now
            same = (self._alerted and self._alerted_pos is not None and
                    max(abs(t.center[0] - self._alerted_pos[0]),
                        abs(t.center[1] - self._alerted_pos[1])) <= RUNE_SAME_TOL)
            first = not same
            due = (self.repeat and same and
                   now - self._alerted >= max(2.0, self.repeat_interval))
            if first or due:
                self._alerted, self._alerted_pos = now, t.center
                return Event("rune", t.center, round(t.span, 1), first,
                             len(self.held))
        elif self._alerted and now - self._last_seen >= RUNE_FORGET:
            self._alerted, self._alerted_pos = 0.0, None
            return Event("gone")
        return None


class CharWatch:
    """캐릭터를 쫓아가며 허용 영역을 벗어나거나 사라지면 알린다."""

    def __init__(self, allow: Tuple[int, int, int, int], hold: float,
                 out_after: float, lost_after: float,
                 repeat_interval: float = 10.0) -> None:
        self.allow = allow
        self.hold = hold
        self.out_after = out_after
        self.lost_after = lost_after
        self.repeat_interval = repeat_interval
        # 캐릭터는 돌아다닌다. 샘플 영상에서 1초 사이 최대 142px 까지 튀었다
        # (텔레포트/맵 이동). 좁게 잡으면 추적이 끊겨 엉뚱한 '실종' 알림이 난다.
        self.tracker = Tracker(tol=CHAR_TOL, grace=CHAR_GRACE)
        self.me: Optional[Track] = None
        self.state = "ok"           # ok / out / lost
        self._pos = None
        self._last_seen = None
        self._out_since = None
        self._alerted = 0.0

    def update(self, dets: List[Detection], now: float) -> Optional[Event]:
        if self._last_seen is None:
            self._last_seen = now
        self.tracker.update(dets, now)
        self.me = self.tracker.nearest(now, self.hold, self._pos)

        if self.me is not None:
            self._pos = self.me.center
            self._last_seen = now
            if inside(self.allow, self._pos):
                was = self.state
                self.state, self._out_since, self._alerted = "ok", None, 0.0
                return Event("back", self._pos) if was != "ok" else None
            if self._out_since is None:
                self._out_since = now
            gone = now - self._out_since
            was = self.state          # _due 가 상태를 바꾸므로 미리 챙겨 둔다
            if gone >= self.out_after and self._due(now, "out"):
                return Event("out", self._pos, round(gone, 1), was != "out")
            return None

        self._out_since = None
        gone = now - self._last_seen
        was = self.state
        if self.lost_after > 0 and gone >= self.lost_after and self._due(now, "lost"):
            return Event("lost", None, round(gone, 1), was != "lost")
        return None

    def _due(self, now: float, state: str) -> bool:
        if self.state != state:
            self.state, self._alerted = state, now
            return True
        if now - self._alerted >= max(2.0, self.repeat_interval):
            self._alerted = now
            return True
        return False
